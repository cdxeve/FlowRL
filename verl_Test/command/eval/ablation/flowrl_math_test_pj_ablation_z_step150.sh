#!/bin/bash
set -x

# Merge model first
BACKEND="fsdp"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/checkpoints/FlowRL/math/7B/flowrl_qwen_7b_1116_ablation_z/global_step_150/actor"
TARGET_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/merged_models/flowrl_qwen_7b_1116_ablation_z_step150"

PYTHONPATH=. python scripts/model_merger.py merge \
  --backend $BACKEND \
  --local_dir $LOCAL_DIR \
  --target_dir $TARGET_DIR

# Test configuration
MODEL_PATH=/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/merged_models/flowrl_qwen_7b_1116_ablation_z_step150
OUTPUT_DIR=/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/flowrl_qwen_7b_1116_ablation_z_step150
DATA_PATH=/mnt/petrelfs/linzhouhan/xuekaizhu/data/math_data/test.parquet

n_gpus_per_node=8

# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=16 \
    data.output_path=$OUTPUT_DIR/test-output-16.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m recipe.r1.main_eval \
    data.path=$OUTPUT_DIR/test-output-16.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func