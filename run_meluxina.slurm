#!/bin/bash -l
#SBATCH -J qwen15b_lora_dolly_meluxina
#SBATCH -A <account>
#SBATCH -p gpu
#SBATCH --qos test # This is default for production jobs
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

module load PyTorch
source .venv/bin/activate

export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME


export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0


python tools/prep_dolly.py --max_train_samples 8000 --max_val_samples 800

accelerate launch   --multi_gpu   --num_processes ${SLURM_GPUS}   --num_machines 1   src/train.py   --bf16   --output_dir outputs/qwen15b_lora_dolly_meluxina   --per_device_batch_size 2   --epochs 1   --num_workers 8
