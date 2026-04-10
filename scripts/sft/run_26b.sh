#!/bin/bash
# Llama-2-26B SFT (随机初始化, 8卡)

PROJECT_ROOT=/root/paddlejob/workspace/env_run/liuyi39/hygon_2030/llama3_xpu_pretrain
cd $PROJECT_ROOT

source /root/paddlejob/workspace/env_run/liuyi39/hygon_2030/venv/bin/activate

# 8卡配置
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_TIMEOUT=2000
export BKCL_SOCKET_IFNAME=eth0
export PYTHONUNBUFFERED=1

# 启动训练
paddleformers-cli train configs/sft/train_26b.yaml > logs/sft/26b_train.log 2>&1

echo "Llama-2-26B SFT completed."
