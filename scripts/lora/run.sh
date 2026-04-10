#!/bin/bash
PROJECT_ROOT=/root/paddlejob/workspace/env_run/liuyi39/hygon_2030/llama3_xpu_pretrain
cd $PROJECT_ROOT

# 激活虚拟环境
source /root/paddlejob/workspace/env_run/liuyi39/hygon_2030/venv/bin/activate

# XPU 配置 (4卡)
export XPU_VISIBLE_DEVICES="0,1,2,3"
export BKCL_TIMEOUT=1000
export BKCL_SOCKET_IFNAME=eth0
export PYTHONUNBUFFERED=1

# 分布式配置
paddleformers-cli train configs/lora/train.yaml > logs/lora/train.log 2>&1