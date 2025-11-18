#!/bin/bash
set -x

echo "=========================================="
echo "Starting evaluation for ablation experiments"
echo "=========================================="

# Evaluation for ablation_z step150
echo "Evaluating: ablation_z_step150"
OUTPUT_DIR_Z="/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/flowrl_qwen_7b_1116_ablation_z_step150"

python3 -m recipe.r1.main_eval \
    data.path=$OUTPUT_DIR_Z/test-output-16.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func

echo "Completed: ablation_z_step150"
echo ""

# Evaluation for ablation_avg_reward_z step100
echo "Evaluating: ablation_avg_reward_z_step100"
OUTPUT_DIR_AVG="/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/flowrl_qwen_7b_1116_ablation_avg_reward_z_step100"

python3 -m recipe.r1.main_eval \
    data.path=$OUTPUT_DIR_AVG/test-output-16.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func

echo "Completed: ablation_avg_reward_z_step100"
echo ""

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
