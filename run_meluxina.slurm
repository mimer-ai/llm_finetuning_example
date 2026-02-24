#!/bin/bash
#SBATCH -J qwen15b_lora_dolly_meluxina
#SBATCH -A <account>
#SBATCH -p <gpu_partition>
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 00:35:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
# module load CUDA/12.1
# module load PyTorch/2.2.0
source .venv/bin/activate

export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + RANDOM % 20000))

python tools/prep_dolly.py --max_train_samples 8000 --max_val_samples 800

accelerate launch   --multi_gpu   --num_processes ${SLURM_GPUS_ON_NODE}   --num_machines 1   src/train.py   --bf16   --output_dir outputs/qwen15b_lora_dolly_meluxina   --per_device_batch_size 6   --epochs 1   --num_workers 8
