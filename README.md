# Paper Title: LLM Beliefs Are in Their Heads

Implementation and experiments for our paper on belief-like representations in LLMs.

## Requirements

### GPU 

Since this experiment runs language models up to 9B in 16fp, a GPU with enough dedicated VRAM is required for replication. The experiment will not start if CUDA is not available. 

### Environment Setup

To install the requirements, activate the environment through conda or mamba. Installation with conda can take several minutes.

```bash
conda env create -f env.yml
conda activate beliefs_llms
```

Then install the remaining requirements through pip

```bash
pip install -r requirements.txt
```

Please set your huggingface token in configs/default.yaml or by adding the corresponding flag when running the experiments.

## Data

Our analysis uses the True/False dataset from S. Marks and M. Tegmark, “The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets,” 2024, doi: https://doi.org/10.48550/arXiv.2310.06824. The dataset is included in our submission as a separate archive "Data.zip".

Once you download the dataset, integrate it with the repository so to see the structure:
```
data/
├── datasets
├──── coherence
├──── true_false
├──── curated_dataset_full.pkl
├──directions
├──── gemma 
├──── gemma_instruct 
├──── gpt-j 
└──── llama
└──── llama_instruct
```

## Reproducing Paper Results

All experiments will be run from the same script by flagging --model "model_name" and --exp "experiment_name". More specifics about the experiment (residual/heads, domains, etc...) will be asked during the run.

**Note**: Each experiment takes approximately 10-40 mins on a single A100 GPU depending on the model size.

### CLI arguments

The script runs on three main arguments: --model; --exp; --hf_token. You can enter your hf_token on configs/default.yml, otherwise please call it through --hf_token.
The --model argument can be used to choose the model to analyze. The values that it takes are:

`gpt`: GPT2-Large 

`pythia`: Pythia-6.9B

`pythia_deduped`: Pythia-6.9B-Deduped

`yi`: Yi-6B

`yi_instruct`: Yi-6B-chat

`gpt-j`: GPT-J-6B

`llama`: Llama-3.1-8B

`llama_instruct`: Llama-3.1-8B-instruct

`gemma`: Gemma-2-9B

`gemma_instruct`: Gemma-2-9B-instruct

### Experiment 1 (Accuracy)
```bash
python src/main.py --model "model_name" --exp "accuracy" --hf_token "put_token_here (optional)"
```

### Experiment 2 (Use)
```bash
python src/main.py --model "model_name" --exp "intervention" --hf_token "put_token_here (optional)"
```

### Experiment 3 (Coherence - Probabilistic)
```bash
python src/main.py --model "model_name" --exp "coherence" --hf_token "put_token_here (optional)"
```

### Experiment 3/4 (Coherence - Logic; Uniformity)
```bash
python src/main.py --model "model_name" --exp "uniformity" --hf_token "put_token_here (optional)"
```

## Activations

We provide a script to access PCA'd and projected activations as seen in Figure 3.
```bash
python src/main.py --model "model_name" --exp "visualization" --hf_token "put_token_here (optional)"
```
## Results Files

Pre-computed results from our paper are in `full_results/`. These are PyTorch-serialized files containing NumPy arrays, lists or Tensors depending on the experiment. 

We provide a simple notebook `scores.ipynb` to access the results and their relative scoring in an easy way.

## Repository Structure
```
├── configs/                      # Contains cfg yaml
├── data/                         # Datasets should go here
├── full_results/                 # Pre-computed results from paper
├── src/                          # Main experiment runner
├────── cfg.py
├────── coherence_experiments.py
├────── experiments.py
├────── main.py
├── utils/                        # Utilities for the experiment
├────── funcs.py
├────── get_model.py
├────── intervention.py
├────── probe.py
├────── processing.py
├────── viz.py
├── environment.yml               # Conda environment
├── README.md                     # This file
├── requirements.txt              # Pip requirements
└── scores.ipynb                  # Notebook to reproduce the scores
```

## Attributions

Parts of the code were adapted from:

S. Marks and M. Tegmark, “The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets,” 2024, doi: https://doi.org/10.48550/arXiv.2310.06824; Code available at https://github.com/saprmarks/geometry-of-truth | Used with permission for research purposes; copyright retained by original authors.

K. Li, O. Patel, F. Viégas, H. Pfister, and M. Wattenberg, “Inference-Time Intervention: Eliciting Truthful Answers from a Language Model,” 2024, doi: https://doi.org/10.48550/arXiv.2306.03341; Code available at https://github.com/likenneth/honest_llama | MIT License Copyright (c) 2023 Kenneth Li

This experiment used **TransformerLens** as an interpretability tool [(Nanda & Bloom, 2022)](https://github.com/TransformerLensOrg/TransformerLens) | MIT License Copyright (c) 2022 TransformerLensOrg

## Contact

For questions or issues please contact the submission's authors.