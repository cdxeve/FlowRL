#!/bin/bash
set -x

# Fix CUDA multiprocessing issue with vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model path - PPO
MODEL_PATH="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/merged_model/ppo_qwen7b_tr-dapo_0602_step200_merged"

# GPU configuration
TENSOR_PARALLEL_SIZE=1  # Number of GPUs to split model across
DATA_PARALLEL_SIZE=8    # Number of model replicas

# Tasks to evaluate (comma-separated)
TASKS="gpqa,mmlu"

# Output directory
OUTPUT_DIR="./results/ppo"

# Run evaluation (reduced gpu_memory_utilization to 0.6 to leave room for evaluation)
python -m lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},dtype=auto,gpu_memory_utilization=0.6,data_parallel_size=${DATA_PARALLEL_SIZE},max_model_len=8192 \
    --tasks ${TASKS} \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}
