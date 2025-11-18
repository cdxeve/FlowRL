#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=xuekai_ablation_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
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

# Run the evaluation script
cd /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test
bash command/eval/ablation/eval_only.sh
