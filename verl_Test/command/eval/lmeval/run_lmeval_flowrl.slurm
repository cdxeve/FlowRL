#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=xuekai_lmeval_flowrl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Unset AMD GPU variable to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Run the evaluation script from lm-evaluation-harness
cd /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/lm-evaluation-harness
bash benchmark_scripts/vllm_eval_flowrl_standalone.sh
