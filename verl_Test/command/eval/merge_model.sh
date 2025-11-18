#!/bin/bash
set -x

BACKEND="fsdp"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/checkpoints/FlowRL/math/7B/flowrl_qwen_7b_1116_ablation_avg_reward_z"
TARGET_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/merged_models/flowrl_qwen_7b_1116_ablation_avg_reward_z"

PYTHONPATH=. python scripts/model_merger.py merge \
  --backend $BACKEND \
  --local_dir $LOCAL_DIR \
  --target_dir $TARGET_DIR

