import sys
from pathlib import Path
import torch
import einops
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.cfg import load_cfg
from datetime import datetime
from collections import defaultdict
cfg = load_cfg()

import torch

def get_orthogonal_direction(v, eps=1e-8):
    r = torch.randn_like(v)
    # orthogonalize r w.r.t. v
    v_norm_sq = (v * v).sum()
    r_orth = r - (r * v).sum() / (v_norm_sq + eps) * v
    # norm-match
    r_orth = r_orth / (r_orth.norm() + eps) * v.norm()
    return r_orth


def force_format(*items, format='tensor', device=None):

    def to_numpy(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
        elif isinstance(item, np.ndarray):
            return item
        elif isinstance(item, list):
            return np.array([to_numpy(x) for x in item])
        else:  # scalar
            return item
    
    if device is None:
        device = cfg["common"]["device"]
    
    results = []

    for item in items:
        if format == 'tensor':
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item)  
            if isinstance(item, list):
                item = to_numpy(item) 
                item = torch.from_numpy(item)
            elif not isinstance(item, torch.Tensor):
                item = torch.tensor(item)
            if device is not None:
                item = item.to(device)
            item = item.half()

        elif format == 'numpy':
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if not isinstance(item, np.ndarray):
                item = np.array(item)
            item = item.astype(np.float16)

        else:
            raise ValueError(f"Unsupported format: {format}")

        results.append(item)

    return results if len(results) > 1 else results[0]

def decompose_mha(mha_batch):

    decomposed = einops.rearrange(mha_batch, 'n_batch batch_size n_head d_head -> n_head n_batch batch_size d_head')
    
    return [decomposed[i] for i in range(decomposed.shape[0])]

def get_top_entries(accuracies, n=5):

    accuracies = np.asarray(accuracies)

    # --- 1D ---
    if accuracies.ndim == 1:
        flat_indices = np.argpartition(accuracies, -n)[-n:]
        top_indices = flat_indices[np.argsort(-accuracies[flat_indices])]
        top_values = accuracies[top_indices]
        return top_indices, top_values

    # --- 2D ---
    elif accuracies.ndim == 2:
        flat_indices = np.argpartition(accuracies.flatten(), -n)[-n:]
        coords = np.array(np.unravel_index(flat_indices, accuracies.shape)).T
        sorted_coords = coords[np.argsort(-accuracies[tuple(coords.T)])]
        top_values = accuracies[tuple(sorted_coords.T)]
        return sorted_coords, top_values

    else:
        raise ValueError("Input must be 1D or 2D.")

def nested_dict():
        return defaultdict(nested_dict)

def save_results(item, datatype, model, modality='residual', k=0, alpha=0, direction='', notes=""):

    base_dir = ROOT / "full_results"
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    filename_map = {
        "accuracies": f"accuracies_{modality}",
        "directions": f"directions_{modality}",
        "probes": f"probes_{modality}",
        "intervention_scores": f"intervention_scores_{modality}",
        "intervention_sweep": f"intervention_sweep_{modality}",
        "uniformity": f"uniformity_{modality}",
        "coherence_scores": f"coherence_scores",
        "viz": f"acts_for_viz"
    }

    if datatype not in filename_map:
        raise ValueError(f"Unsupported datatype: {datatype}")

    if datatype == "intervention_scores":
        path = base_dir / model / "USE" / (f"{filename_map[datatype]}"f"{notes}")
    elif datatype == "coherence_scores":
        path = base_dir / model / "COHERENCE" / (filename_map[datatype]+notes)
    elif datatype == "uniformity":
        path = base_dir / model / "UNIFORMITY" / (filename_map[datatype]+notes)
    elif datatype == "viz":
        path = base_dir / model / "ACTIVATIONS" / (f"{filename_map[datatype]}_{modality}")
    else:       # Accuracies, directions, probes
        path = base_dir / model / "ACCURACY" / (filename_map[datatype]+notes)

    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base_dir / f"{path.stem}_{ts}{path.suffix}"

    try:
        if isinstance(item, torch.Tensor):
            item_to_save = item.detach().cpu()
        else:
            item_to_save = item
    except Exception:
        item_to_save = item
        
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(item_to_save, path)

    return path