#!/bin/bash
#SBATCH -J qwen15b_lora_dolly_lumi
#SBATCH -A project_465002387
#SBATCH -p dev-g
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
#source .venv/bin/activate

export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

export NCCL_DEBUG=WARN
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

singularity run $SIF python tools/prep_dolly.py --max_train_samples 10000 --max_val_samples 1000

singularity run $SIF accelerate launch   --multi_gpu   --num_processes ${SLURM_GPUS}   --num_machines 1   src/train.py   --bf16   --output_dir outputs/qwen15b_lora_dolly_lumi   --per_device_batch_size 2   --epochs 1   --num_workers 8
