import argparse
import os
import random
import torch as t
import numpy as np
from huggingface_hub import login
import subprocess
from cfg import load_cfg
from experiments import EXPERIMENTS
cfg = load_cfg()

def main():

    print()
    print("Welcome to the experiments for our project!")
    print()

    # Check CUDA availability once
    cuda_available = t.cuda.is_available()
    if cuda_available:
        print(f"CUDA available")
        print(t.cuda.get_device_name(0))  

    else:
        raise RuntimeError("CUDA must be available for running experiments")
    
    parser = argparse.ArgumentParser(description="Run your pipeline with a config + seed")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--exp", choices=EXPERIMENTS.keys(), required=True,
                        help="Experiment to run")
    parser.add_argument("--set", nargs="*", help="Overrides like train.lr=1e-4")
    parser.add_argument("--hf_token", help="Hugging Face access token")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--seed", type=int, default=666, help="Random seed")
    args = parser.parse_args()
    token = args.hf_token or os.getenv("HF_TOKEN") or getattr(cfg, "hf_token", None)
    if not token:
        raise ValueError("No Hugging Face token provided. "
                        "Set --hf_token, env var HF_TOKEN, or cfg.hf_token.")
    login(token)  # authenticates the session


    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model
    print(f"Running {args.exp} on {model_name} with {args.config} (seed={args.seed})")
    EXPERIMENTS[args.exp](model_name=model_name)

if __name__ == "__main__":
    main()
