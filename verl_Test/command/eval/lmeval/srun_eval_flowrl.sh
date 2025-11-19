#!/bin/bash
# Run FlowRL evaluation with srun (interactive, see output directly)
# Usage from verl_Test/: sh command/eval/lmeval/srun_eval_flowrl.sh

srun --partition=plm \
     --nodes=1 \
     --gres=gpu:8 \
     --cpus-per-task=64 \
     bash -c 'cd /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/lm-evaluation-harness && bash /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/command/eval/lmeval/vllm_eval_flowrl_standalone.sh'
