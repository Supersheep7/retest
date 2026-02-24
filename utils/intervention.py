import torch as t
import tqdm.auto as tqdm
from tqdm import tqdm
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer
from jaxtyping import Float
from typing import List, Tuple
import numpy as np
import functools
from .funcs import force_format

'''
= = = = = = = = = = = = = = = = Intervention = = = = = = = = = = = = = = = =
'''

def generate(model, prompt, max_length=50, temperature=0.0, top_k=None):
    with t.no_grad():
        model.tokenizer.padding_side = "left"
        tokens = model.to_tokens(prompt)
        generated_tokens = tokens.clone()
        batch_size = generated_tokens.size(0)

        finished = t.zeros(batch_size, dtype=t.bool, device=generated_tokens.device)

        for _ in range(max_length):
            logits = model(generated_tokens)  # (B, L, V)
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits /= temperature

            if top_k is not None:
                top_k_vals, _ = t.topk(next_token_logits, top_k, dim=-1)
                thresh = top_k_vals[:, -1].unsqueeze(1)
                next_token_logits = t.where(
                    next_token_logits < thresh,
                    t.full_like(next_token_logits, -float("inf")),
                    next_token_logits,
                )

            probs = t.nn.functional.softmax(next_token_logits, dim=-1)

            if temperature > 0:
                next_token = t.multinomial(probs, 1)              # (B, 1)
            else:
                next_token = t.argmax(probs, dim=-1, keepdim=True)

            next_token = t.where(
                finished.unsqueeze(1),
                t.full_like(next_token, model.tokenizer.eos_token_id),
                next_token,
            )

            generated_tokens = t.cat([generated_tokens, next_token], dim=1)

            finished |= (next_token.squeeze(1) == model.tokenizer.eos_token_id)
            if finished.all():
                break

        outputs = []
        prompt_len = tokens.size(1)
        for i in range(batch_size):
            outputs.append(
                model.tokenizer.decode(generated_tokens[i, prompt_len:])
            )

    return outputs

def mask_top_k(activation_accuracies: np.array,
                activation_directions: np.array,
                K: int = 1
                ) -> Tuple[List[Tuple], List[np.array]]:

    """
    Takes a tensor of head accuracies and a tensor of head directions,
    returns a list of the top K heads in coordinate form and their corresponding directions.

    E.g. with K = 1
    returns top_k_indices = [(10, 12)]
    returns top_k_directions = [activation_directions[10][12]] = [direction vector of head 12 in layer 10]
    """

    assert activation_accuracies.shape == activation_directions.shape[:2], "Shape mismatch between activation_accuracies and activation_directions"
    assert K <= activation_accuracies.numel(), "K is larger than the number of available heads"

    # Get the indices of the top K heads
    head_accuracies_flattened = activation_accuracies.flatten()
    flat_indices = np.argsort(head_accuracies_flattened)[-K:]
    row_indices, col_indices = np.unravel_index(flat_indices, activation_accuracies.shape)
    top_k_indices = list(zip(row_indices, col_indices)) # (list of tuples of (layer, head) indices)

    # Get the corresponding directions
    top_k_directions = []
    top_k_directions = [activation_directions[layer, head] for layer, head in top_k_indices]

    return top_k_indices, top_k_directions

def compute_attention_sign_mask(model: HookedTransformer,
                                head_directions: t.Tensor,
                                resid_mid_directions: t.Tensor
                                ) -> t.Tensor:
    
    head_directions = force_format(head_directions, format='tensor')
    resid_mid_directions = force_format(resid_mid_directions, format='tensor')
    n_layers, n_heads, d_head = head_directions.shape

    signed_directions = head_directions.clone()

    flipped = 0

    for l in range(n_layers):
        w_mid = resid_mid_directions[l]  # (d_model,)
        
        # full W_O for this layer
        W_O = model.blocks[l].attn.W_O    # (n_heads*d_head, d_model) or (n_heads, d_head, d_model)

        if W_O.ndim == 3 and W_O.shape[0] == n_heads:
            # per-head layout (LLaMA style)
            for h in range(n_heads):
                d_z = signed_directions[l, h]
                W_O_h = W_O[h]  # (d_head, d_model)
                w_head = W_O_h @ w_mid
                dotted = t.dot(d_z.float(), w_head.float())
                print(dotted)
                if dotted < 0:
                    flipped += 1
                    signed_directions[l, h] = -d_z

        elif W_O.ndim == 2 and W_O.shape[0] == n_heads * d_head:
            # flattened layout (GPT style)
            for h in range(n_heads):
                d_z = signed_directions[l, h]
                h_start = h * d_head
                h_end = h_start + d_head
                W_O_h = W_O[h_start:h_end, :]  # (d_head, d_model)
                w_head = W_O_h @ w_mid
                dotted = t.dot(d_z.float(), w_head.float())
                print(dotted)
                if dotted < 0:
                    flipped += 1
                    signed_directions[l, h] = -d_z

        else:
            raise ValueError(f"Unexpected W_O shape {W_O.shape}")
        
    print()
    print(f"Flipped {flipped} heads")
    print()

    take = input("Use signed directions? (y/n): ")
    if take.lower() != 'y':
        signed_directions = head_directions.clone() # override

    return signed_directions
    
def set_intervention_hooks(model: HookedTransformer,
                           top_k_indices: List[Tuple],
                           top_k_directions: List[t.Tensor],
                           last_positions,
                           alpha: float = 1,
                           verbose: bool = False
                           ) -> List:

    """
    Sets the intervention hooks for the top K heads.
    """

    def steering_hook(z: Float[t.Tensor, "d_batch seq_len n_head d_head"],
                      hook: HookPoint,
                      head_idx: int,
                      head_direction: t.Tensor,
                      last_positions,
                      alpha: float = 1):

        """
        Steers the model by returning a modified activations tensor,
        with some multiple of the steering vector added to the top K heads.
        """
        assert head_direction.shape == z.shape[-1:], f"Shape mismatch: {head_direction.shape} vs {z.shape[-1:]}"
        # Steer only the d_head corresponding to the given head_index
        batch_idx = t.arange(z.shape[0], device=z.device)

        head_direction = head_direction / head_direction.norm()
        head_direction_clamped = head_direction * alpha

        z[batch_idx, last_positions, head_idx, :] += head_direction_clamped.half()

        return z

    model.reset_hooks()
    half = True if next(model.parameters(), None).dtype == t.float16 else False

    if half:
        # Set half precision for the steering
        model.add_hook("hook_embed", lambda tensor, hook: tensor.half())
    if verbose:
        print(f"Setting hooks for top {len(top_k_indices)} heads:")
    for (layer, head), direction in zip(top_k_indices, top_k_directions):
        layer = int(layer)
        head = int(head)
        if half:
            direction = direction.clone().detach().half()
        steering = functools.partial(steering_hook, head_idx=head, head_direction=direction, last_positions=last_positions, alpha=alpha)
        model.add_hook(f"blocks.{layer}.attn.hook_z", steering)

    return model

def full_intervention(model: HookedTransformer,
                      activation_accuracies: np.array,
                      activation_directions: np.array,
                      last_positions,
                      K: int = 1,
                      alpha: int = 1,
                      verbose: bool = False) -> HookedTransformer:

    """
    Full intervention function that sets the hooks for the top K heads and returns the model with the hooks set.
    """

    if K > activation_accuracies.numel():
        raise ValueError(f"K ({K}) cannot be larger than the number of heads ({activation_accuracies.numel()})")

    # Get the top K heads and their directions
    top_k_indices, top_k_directions = mask_top_k(activation_accuracies, activation_directions, K)

    top_k_indices = force_format(top_k_indices, format='tensor')
    top_k_directions = force_format(top_k_directions, format='tensor')

    # Set the intervention hooks for the top K heads
    model = set_intervention_hooks(model, top_k_indices=top_k_indices, top_k_directions=top_k_directions, alpha=alpha, verbose=verbose, last_positions=last_positions)

    return model

def intervention_on_residual(
                            model: HookedTransformer,
                            activation_accuracies: np.array,
                            activation_directions: np.array,
                            last_positions,
                            k: int = -1,
                            alpha: int = 1,
                            top_features: np.array = None,
                            verbose: bool = False
                            ) -> HookedTransformer:

    if k > len(activation_accuracies):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({activation_accuracies.numel()})")

    top_k_indices = np.argsort(activation_accuracies)[-k:][::-1]
    top_k_directions = [t.as_tensor(activation_directions[i]) for i in top_k_indices]

    def steering_residual_hook(
                      resid: Float[t.Tensor, "n_batch n_seq d_model"],
                      hook: HookPoint,
                      direction: t.Tensor,
                      last_positions,
                      alpha: float = 1,
                      top_features: int = None
                      ):

        assert direction.shape == resid.shape[-1:], f"Shape mismatch: {direction.shape} vs {resid.shape[-1:]}"
        batch_idx = t.arange(resid.shape[0], device=resid.device)

        direction = direction / direction.norm()
        direction_clamped = direction * alpha
        resid[batch_idx, last_positions, :] += direction_clamped.half()
        return resid

    model.reset_hooks()
    # Set half precision for the steering
    model.add_hook("hook_embed", lambda tensor, hook: tensor.half())
    for layer, direction in zip(top_k_indices, top_k_directions):        
        direction = direction.clone().detach().half()
        steering = functools.partial(steering_residual_hook, direction=direction, alpha=alpha, last_positions=last_positions, top_features=top_features)
        model.add_hook(f"blocks.{layer}.hook_resid_post", steering)

    return model


'''
= = = = = = = = = = = = = = = = Evaluation = = = = = = = = = = = = = = = =
'''

''' *** Parameter Sweep ***'''

def parameter_sweep(model_baseline: HookedTransformer,
                    prompts: List[str],
                    activation_accuracies,
                    activation_directions,
                    ks : List = [1, 2, 3, 4, 5],
                    alphas: List = [1, 2, 3, 4, 5],
                    metric: str = 'boolprobs',
                    verbose: bool = True,
                    shots: List = None,
                    labels: List[int] = None,
                    attn: bool = True,
                    model_family: str = 'llama'
                    ) -> np.array:

    with t.no_grad():

        metrics = np.zeros((len(ks), len(alphas)))

        if metric == 'boolprobs':
            prob_diff = np.zeros((len(ks), len(alphas)))

        model_baseline.reset_hooks()
        model_baseline.add_hook("hook_embed", lambda tensor, hook: tensor.half())

        for num_k, k in tqdm(enumerate(ks)):
                tqdm.write(f"Steering top {k} activations")
                for num_alpha, alpha in tqdm(enumerate(alphas)):
                    tqdm.write(f"With strength {alpha}")
                    model_baseline.reset_hooks()
                    if labels is None:
                            raise ValueError("Labels must be provided for truth assignment evaluation.")
                    metrics[num_k, num_alpha], prob_diff[num_k, num_alpha] = mass_truth_assignment_eval(model_baseline, 
                                                                                                        prompts, 
                                                                                                        attn=attn, 
                                                                                                        shots=shots, 
                                                                                                        labels=labels,
                                                                                                        activation_accuracies=activation_accuracies,
                                                                                                        activation_directions=activation_directions,
                                                                                                        k=k,
                                                                                                        alpha=alpha,
                                                                                                        verbose=verbose,
                                                                                                        model_family=model_family
                                                                                                        )
        if metric == 'boolprobs':
            return metrics, prob_diff
        else:
            return metrics

''' *** P(True)-P(False) ***'''

def mass_truth_assignment_eval(
              model_baseline: HookedTransformer,
              statements: List[str],
              labels: List[int],
              activation_accuracies, activation_directions, k, alpha, verbose,
              true_tokens: List[str] = ['true','Ġtrue','True','ĠTrue','▁true','▁True'],
              false_tokens: List[str] = ['false','Ġfalse','False','ĠFalse','▁false','▁False'],
              attn = True,
              shots: List[str] = None,
              batch_size: int = 100,
              model_family: str = 'llama'
              ) -> float:

    """
    Evaluates the model on a set of prompts and labels, returning the average P(True) - P(False) score.
    """

    assert len(statements) == len(labels), "Number of prompts and labels must match"

    true_token_ids = [model_baseline.tokenizer.convert_tokens_to_ids(token) for token in true_tokens if token in model_baseline.tokenizer.get_vocab()]
    false_token_ids = [model_baseline.tokenizer.convert_tokens_to_ids(token) for token in false_tokens if token in model_baseline.tokenizer.get_vocab()]
    pad = model_baseline.tokenizer.pad_token_id

    total_metric = 0.0
    total_prob_diff = 0.0
    for i in range(0, len(statements), batch_size):
        batch_statements = statements[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        print("Prompting in the style of", model_family)
        if model_family == 'llama':
            batch_prompts = [
                f"Determine whether the following statement is factually correct. Respond with exactly one of: True, False, Unknown. Answer Unknown unless you are certain.\n{stmt}\nAnswer:".rstrip()
                for stmt in batch_statements
            ]
        elif model_family == 'gemma':
            batch_prompts = [
                f"Determine whether the following statement is factually correct. Respond with exactly one of: True, False, Unknown. Answer Unknown unless you are certain. \nStatement: {stmt} Answer:".rstrip()
                for stmt in batch_statements
            ]
        elif model_family == 'gpt':
            batch_prompts = [
                f"Determine whether the following statement is factually correct. Respond with exactly one of: True, False, Unknown. Answer Unknown unless you are certain.\n{stmt}\nAnswer:"
                for stmt in batch_statements
            ]


        tokens = model_baseline.to_tokens(batch_prompts)
        last_positions = (tokens != pad).sum(dim=1) - 1     # Change to - 0 for GPT-style models

        if attn:
            model = full_intervention(model_baseline, activation_accuracies=activation_accuracies, activation_directions=activation_directions, K=k, alpha=alpha, verbose=verbose, last_positions=last_positions)
        else:
            model = intervention_on_residual(model=model_baseline, activation_accuracies=activation_accuracies, activation_directions=activation_directions, k=k, alpha=alpha, verbose=verbose, last_positions=last_positions)
        with t.no_grad():
            with t.amp.autocast(device_type='cuda', dtype=t.float16):
                logits = model(tokens)

        log_probs = t.nn.functional.log_softmax(logits, dim=-1)
        seq_lens = (tokens != pad).sum(dim=1)
        last_positions = seq_lens - 1               # Change to - 0 for GPT-style models
        
        for j, label in enumerate(batch_labels):
            j_pos = last_positions[j].item()

            # log-probs at final position
            lp = log_probs[j, j_pos]          # shape: [vocab_size]

            # top-5
            topk_logp, topk_ids = t.topk(lp, k=5)

            topk_probs = topk_logp.exp()
            topk_tokens = model.tokenizer.convert_ids_to_tokens(topk_ids.tolist())

            print("Top-5 tokens:")
            for tok, p in zip(topk_tokens, topk_probs.tolist()):
                print(f"  {tok!r}: {p:.6f}")
            log_p_true = t.logsumexp(log_probs[j, last_positions[j], true_token_ids], dim=0).item()
            log_p_false = t.logsumexp(log_probs[j, last_positions[j], false_token_ids], dim=0).item()
            j_pos = last_positions[j].item()
            most_probable_token_id = t.argmax(log_probs[j, j_pos]).item()
            most_probable_token = model.tokenizer.convert_ids_to_tokens([most_probable_token_id])[0]
            most_probable_prob = log_probs[j, j_pos, most_probable_token_id].exp().item()
            THRESHOLD = 0.2 if model_family in ['llama','gpt'] else 0.15

            print(f"Prompt: {batch_prompts[j]}")
            print(f"P(True): {np.exp(log_p_true):.6f}, P(False): {np.exp(log_p_false):.6f}")

            if most_probable_token_id not in true_token_ids + false_token_ids or most_probable_prob < THRESHOLD:
                print(f"Unsure! Answer: {most_probable_token}")
                successful = 0
            else:
                successful = int(int(log_p_true >= log_p_false) != label)

            prob_diff = np.exp(log_p_true) - np.exp(log_p_false) if label == 0 else np.exp(log_p_false) - np.exp(log_p_true)
            total_metric += successful
            total_prob_diff += prob_diff

    n = len(statements)
    tot_perc = total_metric / n
    tot_pd = total_prob_diff / n

    return tot_perc,  t.tensor(tot_pd).half().item()
