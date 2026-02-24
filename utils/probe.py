import sys
from pathlib import Path
from sklearn.metrics import accuracy_score
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import einops
import copy
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .processing import get_activations
from jaxtyping import Float
from typing import Tuple, List
import pickle
from tqdm import tqdm
import einops
import re
from sklearn.linear_model import LogisticRegression
from .intervention import generate
import matplotlib.pyplot as plt
from .funcs import *
from sklearn.model_selection import StratifiedKFold
from src.cfg import load_cfg
from sklearn.isotonic import IsotonicRegression
cfg = load_cfg()
probe_cfg = cfg["probe"]
device = t.device("cuda" if t.cuda.is_available() else "cpu")

'''
Here we have the three probes that we will deploy to test for internal representation of belief
Logreg is a simple logistic regressor
MMP is a mass-mean probe as described in Marks & Tegmark 2023
Neural is a tentative copy of SAPLMA as described in Azaria & Mitchell 2023
'''

''' MMP class adapted from https://github.com/saprmarks/geometry-of-truth/blob/main/probes.py '''

''' Part of the following code is adapted from https://github.com/collin-burns/discovering_latent_knowledge/blob/main/CCS.ipynb by Burns et al. 2022 '''

from sklearn.base import BaseEstimator, ClassifierMixin

class MassMeanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, direction, inv_cov):

        self.direction = direction
        self.device = self.direction.device
        self.inv_cov = inv_cov

    def fit(self, X=None, y=None):
        # no training needed; direction is precomputed
        return self

    def predict_proba(self, X):
        if not isinstance(X, t.Tensor):
            X_t = t.from_numpy(X).to(self.device).float()
        else:
            X_t = X.to(self.device).float()
        proj = X_t.float() @ self.inv_cov.float() @ self.direction.float()
        print(proj.mean().item(), proj.std().item())
        probs = t.sigmoid(proj).cpu().numpy()
        return np.column_stack([1 - probs, probs])  # shape (n_samples, 2)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

class MMP(nn.Module):

    def __init__(self, direction, covariance, inv=None, atol=1e-3):
        super().__init__()
        self.direction = direction.to(device)

        if inv is None:
            inv_32 = t.linalg.pinv(covariance.float(), hermitian=True, atol=atol)
            self.inv = nn.Parameter(inv_32.to(dtype=t.float16), requires_grad=False).to(device)
        else:
            self.inv = nn.Parameter(inv, requires_grad=False).to(device)

    def forward(self, x, iid=True, project=False):
        inv = self.inv.half()
        direction = self.direction.half()
        x = x.to(device).half()
        
        if iid:
            projection = (x @ inv @ direction).unsqueeze(1)
        else:
            projection = (x @ direction).unsqueeze(1)

        return projection if project else t.sigmoid(projection)
        
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, d//2)
        self.linear2 = nn.Linear(d//2, d//4)
        self.linear3 = nn.Linear(d//4, d//8)
        self.linear4 = nn.Linear(d//8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = t.relu(x)
        x = self.linear2(x)
        x = t.relu(x)
        x = self.linear3(x)
        x = t.relu(x)
        x = self.linear4(x)
        return t.sigmoid(x)

class Probe(object):

    def __init__(self, probe_cfg):

        # probe config
        self.seed = cfg["common"]["seed"]
        self.var_normalize = probe_cfg["var_normalize"]
        self.dropout = probe_cfg["dropout"]
        self.direction_type = probe_cfg["direction_type"]
        self.epochs = probe_cfg["epochs"]
        self.lr = probe_cfg["lr"]
        self.verbose = probe_cfg["verbose"]
        self.device = probe_cfg["device"]
        self.batch_size = probe_cfg["batch_size"]
        self.weight_decay = probe_cfg["weight_decay"]
        self.max_iter = probe_cfg["max_iter"]
        self.C = float(probe_cfg["C"])
        self.probe_type = probe_cfg["probe_type"]
        self.control = probe_cfg["control"]

        # probe
        self.probe = None
        self.best_probe = None
        self.train_loader = None
        self.test_loader = None
        self.direction = None
        self.covariance = None
        self.std = None

    def initialize_direction(self, direction_type, full_dataset=True, whitened=False):
        
        if direction_type == 'mmp':
            """
            We take the mass mean of positive and negative activations and subtract them to get the direction.
            """
            if full_dataset:
                data_for_mm = t.cat([self.unnormalized_X_train, self.unnormalized_X_test], dim=0).to(device)
                labels_for_mm = t.cat([self.y_train, self.y_test], dim=0)
            else:
                data_for_mm = t.tensor(self.unnormalized_X_train, device=device)
                labels_for_mm = t.tensor(self.y_train, device=device)
            pos_acts, neg_acts = data_for_mm[labels_for_mm == 1], data_for_mm[labels_for_mm == 0]
            pos_mean, neg_mean = pos_acts.float().mean(0), neg_acts.float().mean(0)
            direction = pos_mean - neg_mean
            centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            cov = centered_data.t() @ centered_data / centered_data.shape[0]
            if whitened:
                inv_cov = t.linalg.pinv(cov.float(), hermitian=True, atol=1e-6).to(device)
                inv_cov = inv_cov.half()
                self.direction = nn.Parameter(inv_cov @ direction, requires_grad=True)
            else:
                self.direction = direction

            self.covariance = cov

        elif direction_type == None: 
            self.direction = None
        else:
            with t.no_grad():
              theta = self.best_probe.coef_[0] if direction_type == 'logistic' else self.best_probe[0].weight[0].cpu().numpy()
              self.direction = nn.Parameter(t.tensor(theta, dtype=t.float, requires_grad=True, device=self.device).squeeze(0))

    def initialize_probe(self, override_probe_type=None):

        if override_probe_type is not None:
            self.probe_type = override_probe_type

        if self.probe_type == "mmp":
            # We need the direction and covariance in advance for the MMP probe 
            self.initialize_direction('mmp', full_dataset=False)
            self.probe = MMP(direction=self.direction, covariance=self.covariance)
        elif self.probe_type == "logistic_regression":
            self.probe = LogisticRegression(max_iter=self.max_iter, solver="lbfgs", C=self.C, random_state=self.seed, n_jobs=-1)
        else:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
            dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            self.train_loader = DataLoader(dataset, batch_size=self.batch_size if self.batch_size > 0 else len(dataset), shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            if self.probe_type == "linear_layer":
                self.probe = nn.Sequential(
                    nn.Linear(self.input_dim, 1, bias=True),
                    nn.Sigmoid()
                )
            if self.probe_type == "mlp":
                self.probe = MLPProbe(self.input_dim)

            self.probe.to(self.device)

    def normalize(self, X_train, X_test):
        train_mean = X_train.mean(dim=0, keepdim=True)
        train_std = X_train.std(dim=0, keepdim=True, unbiased=False)

        # avoid zero or NaN std
        train_std = torch.where(train_std == 0, torch.ones_like(train_std), train_std)
        train_std = torch.nan_to_num(train_std, nan=1.0)

        normalized_X_train = (X_train - train_mean)
        normalized_X_test  = (X_test - train_mean)

        if self.var_normalize:
            normalized_X_train /= train_std
            normalized_X_test  /= train_std

        return normalized_X_train, normalized_X_test

    def train(self):
        """
        Does a single training run of epochs 
        """

        if self.probe_type == "logistic_regression":
            X_train, y_train = force_format(self.X_train, self.y_train, format='numpy', device=None)
            self.initialize_probe()
            self.probe.fit(X_train, y_train)
            y_pred = self.probe.predict(X_train)
            acc = accuracy_score(y_pred, y_train)
            self.best_probe = copy.deepcopy(self.probe)

        elif self.probe_type == "mmp":
            self.initialize_probe()

        else:
            self.initialize_probe()        
            # set up optimizer
            optimizer = t.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            best_loss = float('inf')
            epoch_losses = []
            accuracies = []

            # Start training
            for epoch in range(self.epochs):
                self.probe.train()
                epoch_loss = 0.0  # Initialize epoch loss

                for x_batch, labels_batch in self.train_loader:
                    # probe
                    p = self.probe(x_batch).squeeze(-1)
                    # get the corresponding loss
                    loss = self.get_loss(p, labels_batch)

                    # update the parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    avg_loss = epoch_loss / len(self.train_loader)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_probe = copy.deepcopy(self.probe)

                epoch_losses.append(avg_loss)  # Track epoch loss for plotting
                self.probe.eval()  # Set the model to evaluation mode
                with t.no_grad():
                    current_acc = self.get_acc()
                    accuracies.append(current_acc)
            
            if self.verbose:
                # Plot the training and validation loss
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
                plt.plot(range(1, len(accuracies) + 1), accuracies, label='Online Accuracy', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Loss Over Epochs Accuracy at the last step: {current_acc}')
                plt.legend()
                plt.grid(True)
                plt.show()

    def predict(self, X, proba=False):
        probe = self.probe
        if self.probe_type == 'logistic_regression':
            if proba:
                projections = probe.predict_proba(X)
            else:
                projections = probe.predict(X)
        elif self.probe_type == 'mmp':
            if proba:
                projections = probe(X, iid=True).detach().cpu().numpy()
            else: 
                # TO DO (If needed): implement prediction
                projections = probe(X, iid=True).detach().cpu().numpy()
        return projections
    
    def save_best_probe(self,
                        filename: str
    ) -> None:
        """
        Save the best trained probe to a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.best_probe, f)
        print(f"Best probe saved to {filename}")

    def get_direction(self,
                      std: bool = False
        ) -> t.Tensor:
        '''
        Gets the chosen direction for steering. 
        '''
        self.initialize_direction(self.direction_type)
        direction = self.direction.clone().detach()

        if std: 
            std = self.get_std()
            return std * direction

        return direction
    
    def get_std(self) -> t.Tensor:
        '''
        For steering.
        '''
        if not isinstance(self.X_train, t.Tensor):
            self.X_train = t.tensor(self.X_train, dtype=t.float, device=self.device)
            self.X_test = t.tensor(self.X_test, dtype=t.float, device=self.device)
        full_dataset = t.cat([self.X_train, self.X_test], dim=0)
        project_on_direction = full_dataset.float() @ self.direction
        self.std = project_on_direction.std(dim=0, keepdim=True)

        return self.std

class SupervisedProbe(Probe):

    def __init__(self,
                 X_train: Float[t.Tensor, "n_data d_activation"],
                 X_test: Float[t.Tensor, "n_data d_activation"],
                 y_train: Float[t.Tensor, "n_data"],
                 y_test: Float[t.Tensor, "n_data"],
                 probe_cfg
                 ):
        super().__init__(probe_cfg=probe_cfg)
        self.input_dim = X_train.shape[-1]
        self.unnormalized_X_train = X_train
        self.unnormalized_X_test = X_test
        X_train, X_test = self.normalize(X_train, X_test) if self.var_normalize else (X_train, X_test)
        self.X_train, self.X_test, self.y_train, self.y_test = force_format(X_train, X_test, y_train, y_test, format='tensor', device=self.device)

        """
        Shuffle the labels if control is True. This is done to create a control condition for the probe.
        """
        if self.control:
            np.random.shuffle(self.y_train)
            if self.probe_type == "mmp":
                np.random.shuffle(self.y_test)

    def get_loss(self,
                 p: Float[t.Tensor, "batch"],
                 labels: Float[t.Tensor, "batch"]
    ) -> Float[t.Tensor, "batch"]:

        return nn.functional.binary_cross_entropy(p, labels)

    def get_acc(self) -> Float:
        '''
        Returns accuracy for the best probe trained on a specific activation
        '''
        if self.probe_type == "logistic_regression":
            # We just call sklearn's predict
            X_test, y_test = force_format(self.X_test, self.y_test, format='numpy', device=None)
            predictions = self.probe.predict(X_test)
            acc = (predictions == y_test).mean()
        elif self.probe_type == 'mmp':
            # We just call a forward
            predictions = self.probe(self.X_test, iid=True).squeeze(-1).detach().cpu().numpy().round() # Only one probe
            acc = (predictions == self.y_test).mean()
            print(acc)

        else:
            with t.no_grad():
                correct = 0
                total = 0
                for x_batch, labels_batch in self.test_loader:
                    predictions = self.best_probe(x_batch).squeeze(-1).round()
                    correct += (predictions == labels_batch).sum().item()
                    total += labels_batch.size(0)
                acc = (correct / total)

        return acc

def probe_sweep(list_of_datasets: List,
                labels: t.Tensor
                ) -> Tuple:
    '''
    Runs a probe sweep on a list of activations

    Takes:
        list: a list of activations (if supervised); a list of tuples (activations0, activations1)
        labels: a tensor of labels of shape = list[0].shape (supervised) or list['pos'][0].shape
        probe_cfg: config object for the probe

    Returns: a list of accuracies for the list of activations; a list of vectors for steering; a list of best_probes for keeping them
    '''
    accuracies = []
    directions = []
    best_probes = []
    labels = einops.rearrange(labels, 'n b -> (n b)')

    print("Starting probe sweep...")

    for dataset in tqdm(list_of_datasets, desc="lrs/heads", disable=not probe_cfg["verbose"]):

        dataset = einops.rearrange(dataset, 'n b d -> (n b) d')
        
        # Combined stratification key: domain x label
        strat_key = [f"{f}_{l}" for f, l in zip(filenames, labels)]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=probe_cfg["seed"])
        
        fold_accuracies = []
        fold_directions = []
        fold_probes = []
        
        for train_idx, test_idx in skf.split(dataset, strat_key):
            X_train, X_test = dataset[train_idx], dataset[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            probe = SupervisedProbe(X_train=X_train, y_train=y_train,
                                    X_test=X_test, y_test=y_test,
                                    probe_cfg=probe_cfg)
            probe.train()
            fold_accuracies.append(probe.get_acc())
            if probe_cfg["direction_type"] is not None:
                fold_directions.append(probe.get_direction(std=probe_cfg["with_std"]))
            fold_probes.append(probe.best_probe)

        accuracies.append(fold_accuracies)
        if probe_cfg["direction_type"] is not None:
            directions.append(fold_directions)
        best_probes.append(fold_probes)

    return (accuracies, directions, best_probes)

class Estimator:
    def __init__(self, estimator_name: str, model, best_layer=None, modality='residual'):
        self.estimator_name = estimator_name
        self.model = model
        self.best_layer = best_layer
        self.modality = modality
        self.train_data = None 
        self.probe = None
        self.logic = None
        self.context = None
        self.context_self = None
        self.ir = None
        self.mmp_direction = None
        self.mmp_scaler = None

    def set_logic(self, logic: str):
        self.logic = logic

    def set_train_data(self, train_data: list):
        self.train_data = train_data

    def set_context(self,   context: str = None, 
                            shots: List[str] = None,
                            context_self: str = None,
                            shots_self: List[str] = None):

        full_context = ""
        if shots is not None:
            for shot in reversed(shots):
                full_context = full_context + f"\n\n{shot}"
        if context is not None:
            full_context = context + "\n\n" + full_context

        self.context = full_context

        full_context_self = ""
        if shots is not None:   
            for shot in reversed(shots_self):
                full_context_self = full_context_self + f"\n\n{shot}"
        if context is not None:
            full_context_self = context_self + "\n\n" + full_context_self
        
        self.context_self = full_context_self

    def logits_evaluate(self, data: list, batch_size=100) -> np.ndarray:
        
        model = self.model
        true_tokens = ['true','True','TRUE','Ġtrue','ĠTrue','ĠTRUE','▁true','▁True','▁TRUE']
        false_tokens = ['false','False','FALSE','Ġfalse','ĠFalse','ĠFALSE','▁false','▁False','▁FALSE']

        true_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in true_tokens if token in model.tokenizer.get_vocab()]
        false_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in false_tokens if token in model.tokenizer.get_vocab()]

        selected_ids = true_ids + false_ids
        pad = model.tokenizer.pad_token_id
        probas = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            prompts = [
                f"{self.context}{statement} This statement is:"
                for statement in batch
            ]
            tokens = model.to_tokens(prompts)
            with t.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(tokens)
            log_probs = t.nn.functional.log_softmax(logits, dim=-1)
            seq_lens = (tokens != pad).sum(dim=1)
            last_positions = seq_lens - 1 # Change to 0 for gpt-style models

            for j, statement in enumerate(batch):
                j_pos = last_positions[j].item()

                # log-probs at final position
                lp = log_probs[j, j_pos]          # shape: [vocab_size]

                # top
                topk_logp, topk_ids = t.topk(lp, k=10)

                topk_probs = topk_logp.exp()
                topk_tokens = model.tokenizer.convert_ids_to_tokens(topk_ids.tolist())
                print("Top 10")
                for tok, p in zip(topk_tokens, topk_probs.tolist()):
                    print(f"  {tok!r}: {p:.6f}")
                log_p_true = t.logsumexp(log_probs[j, last_positions[j], true_ids], dim=0).item()
                log_p_false = t.logsumexp(log_probs[j, last_positions[j], false_ids], dim=0).item()
                p_true = np.exp(log_p_true)
                p_false = np.exp(log_p_false)
                total_mass = p_true + p_false                     
                low_conf = total_mass < 0.3
                if low_conf:
                    print(
                        f"Warning: Low confidence in prediction for statement: "
                        f"'{statement}'. P(True)+P(False)={total_mass:.4f}"
                    )
                probs = p_true / total_mass
                probas.append(probs)
        
        return t.tensor(probas)

    def self_evaluate(self, data: list, batch_size=100) -> np.ndarray:
        model = self.model
        probas = []

        for i in tqdm(range(0, len(data), batch_size),
                    desc="Evaluating statements, self-reporting"):
            batch = data[i:i + batch_size]

            prompts = [
                f"{self.context_self}Statement: {s}\nP(True):"
                for s in batch
            ]

            with t.no_grad(), t.autocast('cuda'):
                answers = generate(
                    model=model,
                    prompt=prompts,        # batched
                    temperature=0,
                    max_length=10
                )
            for statement, answer in zip(batch, answers):
                match = re.search(r'\d+\.\d+', answer)
                if match:
                    probas.append(float(match.group()))
                else:
                    print(
                        f"Warning: No confidence score found for statement: "
                        f"'{statement}'. Answer: {answer}"
                    )
                    probas.append(0.5)

        return t.tensor(probas)
    
    def train_estimator(self):
        if self.estimator_name in ['logistic_regression', 'mmp']:
            # train probe 
            print("Training estimator...")
            data = self.train_data
            activations, labels = get_activations(self.model, data, self.modality, focus=self.best_layer)
            activations = activations[f'blocks.{self.best_layer}.hook_resid_post'] if self.modality == 'residual' else activations[f'blocks.{self.best_layer[0]}.attn.hook_z']
            if self.modality == 'heads':
                heads = decompose_mha(activations)
                activations = heads[self.best_layer[1]]
            X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.1, random_state=42)
            X_train = einops.rearrange(X_train, 'n b d -> (n b) d').detach().cpu().numpy()
            y_train = einops.rearrange(y_train, 'n b -> (n b)').detach().cpu().numpy()
            X_test = einops.rearrange(X_test, 'n b d -> (n b) d').detach().cpu().numpy()
            y_test = einops.rearrange(y_test, 'n b -> (n b)').detach().cpu().numpy()

            if self.estimator_name == 'logistic_regression':

                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(
                        max_iter=1000,
                        n_jobs=-1,
                        solver="lbfgs",
                        multi_class="auto"
                    ))
                ])
                print("Train start...")
                clf.fit(X_train, y_train)

            else:
                print("MMP...")
                # X_train: shape (n_samples, d), y_train: 0/1 labels
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                X_t = t.from_numpy(X_train_scaled).float()  # or keep it as tensor

                # compute covariance (d x d)
                cov = t.cov(X_t.T)  

                # compute pseudo-inverse
                atol = 1e-3  
                inv_cov = t.linalg.pinv(cov, hermitian=True, atol=atol).float()
                inv_cov = inv_cov.float()

                pos_mask = t.from_numpy(y_train).bool()
                neg_mask = ~pos_mask
                X_pos = X_t[pos_mask]
                X_neg = X_t[neg_mask]
                direction = (X_pos.mean(dim=0) - X_neg.mean(dim=0)).half()

                clf = MassMeanClassifier(direction, inv_cov)
                self.mmp_scaler = scaler
                X_test = scaler.transform(X_test)
            
            # evaluate
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print("Accuracy on train dataset:", acc)

            probas = clf.predict_proba(X_train)[:, 1]
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probas, y_train)

            self.probe = clf
            self.ir = ir

        else:
            raise ValueError(f"Unsupported estimator: {self.estimator_name}")

    def extract_proba(self, data, batch_size=100) -> np.ndarray:
        
        if self.estimator_name in ['logistic_regression', 'mmp']:
            probe = self.probe 
            ir = self.ir
            data = (data, None) # Useful for get_activations loop
            activations, labels = get_activations(self.model, data, self.modality, focus=self.best_layer)
            activations = activations[f'blocks.{self.best_layer}.hook_resid_post'] if self.modality == 'residual' else activations[f'blocks.{self.best_layer[0]}.attn.hook_z']
            if self.modality == 'heads':
                heads = decompose_mha(activations)
                activations = heads[self.best_layer[1]]
            X = einops.rearrange(activations, 'n b d -> (n b) d')
            if self.estimator_name == 'mmp':
                X = self.mmp_scaler.transform(X.detach().cpu().numpy())  
            probas = probe.predict_proba(X)[:, 1]
            pseudo_probs = ir.transform(probas)
            return t.tensor(pseudo_probs)

        elif self.estimator_name == 'logits':
            return self.logits_evaluate(data)
        
        elif self.estimator_name == 'self_report':
            return self.self_evaluate(data)

        else:
            raise ValueError(f"Unsupported estimator: {self.estimator_name}")