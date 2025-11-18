#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=xuekai_lmeval_ppo
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/lm-evaluation-harness/logs/%j.out
#SBATCH --error=/mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/lm-evaluation-harness/logs/%j.err

# Change to lm-evaluation-harness directory
cd /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/lm-evaluation-harness

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
echo "Working Directory: $(pwd)"
echo "=========================================="

# Run the evaluation script using absolute path
bash /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/command/eval/lmeval/vllm_eval_ppo_standalone.sh
