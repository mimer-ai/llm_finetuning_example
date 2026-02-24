# LLM Finetune (Dolly 15k)

This package fine-tunes **Qwen/Qwen2.5-1.5B-Instruct** using **LoRA** on a
subset of **databricks-dolly-15k** (CC BY-SA 3.0). It includes SLURM scripts
for Leonardo, LUMI, MeluXina. Use `tools/prep_dolly.py` to download and convert
Dolly to JSONL. This is meant to be an example on how to run LLM finetuning
jobs on different EuroHPC clusters.

## Quick start

On LUMI and Leonardo, the modules contain all the needed packages and are
called accordingly from the submission script. In particular, on LUMI the
relevant software can be accessed with

```bash
module use /appl/local/csc/modulefiles
ml pytorch
```

and on Leonardo:

```bash
ml profile/deeplrn cineca-ai
```

On Leonardo, models and datasets need to be prefetched from the login node.
Running `python tools/prep_dolly.py` will download the dataset to the right
folder. To download the model, HF cli can be used: `hf download
Qwen/Qwen2.5-1.5B-Instruct`. 
LUMI has internet access so the model can be downloaded on the fly, but the
script still expects training and validation datasets to be prefetched for
compatibility with Leonardo.

The training can then be run with:

```bash
sbatch run_leonardo.slurm # same thing for run_lumi.slurm and run_meluxina.slurm
```

## Licensing
- Dataset: databricks-dolly-15k (CC BY-SA 3.0) — commercial use allowed with attribution and share-alike. See dataset card. 
- Code: Apache‑style MIT/Apache compatible packages.
