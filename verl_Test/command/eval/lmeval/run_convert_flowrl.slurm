#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=xuekai_convert_flowrl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

cd /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL

bash verl_Test/command/eval/convert_flowrl.sh
