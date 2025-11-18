#!/bin/bash
set -x

# Model path - FlowRL (uses custom architecture with proj_z)
MODEL_PATH="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200"

# Tasks to evaluate (comma-separated)
TASKS="gpqa,mmlu"

# Output directory
OUTPUT_DIR="./results/flowrl"

# Run evaluation using HuggingFace (vLLM doesn't support FlowRL's custom architecture)
# Note: HF is slower than vLLM but works with custom model modifications like proj_z
python -m lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True \
    --tasks ${TASKS} \
    --batch_size 1 \
    --output_path ${OUTPUT_DIR}
