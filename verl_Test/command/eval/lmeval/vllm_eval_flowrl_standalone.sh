#!/bin/bash
set -x

# Model path - FlowRL (uses custom architecture with proj_z)
MODEL_PATH="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200"

# Tasks to evaluate (comma-separated)
TASKS="gpqa,mmlu"

# Output directory
OUTPUT_DIR="./results/flowrl"

# Transformers optimizations
export TOKENIZERS_PARALLELISM=false

# Run evaluation using HuggingFace Transformers (vLLM doesn't support FlowRL's custom architecture)
# Optimizations: parallelism, larger batch size, use_accelerate for multi-GPU
python -m lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True,parallelize=True,use_accelerate=True,max_length=8192 \
    --tasks ${TASKS} \
    --batch_size 4 \
    --output_path ${OUTPUT_DIR}
