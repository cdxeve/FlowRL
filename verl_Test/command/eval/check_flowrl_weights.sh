#!/bin/bash

# Check FlowRL model for proj_z parameters

echo "Checking original FlowRL model..."
python /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/command/eval/check_model_parameters.py \
    /mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200 \
    proj_z

echo ""
echo "========================================"
echo ""

echo "Checking converted model (if exists)..."
if [ -d "/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200_no_proj_z" ]; then
    python /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_Test/command/eval/check_model_parameters.py \
        /mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200_no_proj_z \
        proj_z
else
    echo "Converted model not found yet."
fi
