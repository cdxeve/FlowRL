#!/bin/bash
set -x

# Fix CUDA multiprocessing issue with vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Original FlowRL model (with proj_z)
SOURCE_MODEL="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200"

# Converted model (without proj_z for vLLM compatibility)
MODEL_PATH="${SOURCE_MODEL}_no_proj_z"

# Tasks to evaluate (comma-separated)
TASKS="gpqa,mmlu"

# Output directory
OUTPUT_DIR="./results/flowrl"

# Convert model to remove proj_z if not already done
if [ ! -d "${MODEL_PATH}" ]; then
    echo "=========================================="
    echo "Converting FlowRL model to remove proj_z"
    echo "=========================================="
    python /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/command/eval/remove_proj_z.py \
        "${SOURCE_MODEL}" \
        "${MODEL_PATH}"
    echo "Conversion complete!"
    echo ""
else
    echo "Using existing converted model: ${MODEL_PATH}"
fi

# GPU configuration
TENSOR_PARALLEL_SIZE=1  # Number of GPUs to split model across
DATA_PARALLEL_SIZE=8    # Number of model replicas

# Run evaluation using vLLM (much faster than HuggingFace)
python -m lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},dtype=auto,gpu_memory_utilization=0.6,data_parallel_size=${DATA_PARALLEL_SIZE},max_model_len=8192 \
    --tasks ${TASKS} \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}
