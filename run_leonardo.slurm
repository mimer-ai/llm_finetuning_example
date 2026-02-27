#!/bin/bash
#SBATCH -J finetune_test
#SBATCH -A EUHPC_D29_075
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

#source .venv/bin/activate
module purge
ml profile/deeplrn cineca-ai/4.3.0

export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

# Rely on OMP parallelism
export TOKENIZERS_PARALLELISM=false

export NCCL_DEBUG=WARN
#export NCCL_ASYNC_ERROR_HANDLING=1
#export CUDA_LAUNCH_BLOCKING=0

#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$((12000 + RANDOM % 20000))

# Prepare Dolly 15k (subset ~10k train / 1k val for ~0.5h)
#python tools/prep_dolly.py --max_train_samples 10000 --max_val_samples 1000

accelerate launch   --multi_gpu   --num_processes ${SLURM_GPUS}   --num_machines 1   src/train.py   --bf16   --output_dir outputs/qwen15b_lora_dolly_leonardo   --per_device_batch_size 2   --epochs 1   --num_workers 8 --max_seq_len 256
