# Llama-2 XPU 训练项目

在国产 XPU（昆仑芯 P800）上验证大模型训练能力。

---

## 环境搭建

### 1. 克隆 PaddleFormers

```bash
cd /root/paddlejob/workspace/env_run/liuyi39/hygon_2030
git clone https://github.com/PaddlePaddle/PaddleFormers.git
```

### 2. 安装依赖

```bash
cd PaddleFormers
pip install -e .
```

### 3. 设置环境变量

```bash
export PYTHONPATH=/root/paddlejob/workspace/env_run/liuyi39/hygon_2030/PaddleFormers:$PYTHONPATH
```

---

## 数据准备

### 预训练数据

```bash
cd /root/paddlejob/workspace/env_run/liuyi39/hygon_2030/llama3_xpu_pretrain
mkdir -p data/pt

# 下载 OpenWebText 100k 数据集
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin -O data/pt/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx -O data/pt/llama_openwebtext_100k.idx
```

### SFT 数据

```bash
mkdir -p data/sft

# 下载 school_math 数据集
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/SFT/school_math_0.25M.tar.gz -O data/sft/school_math_0.25M.tar.gz
tar -xf data/sft/school_math_0.25M.tar.gz -C data/sft/
```

---

## 模型准备

### 下载 Llama-2-13B

```bash
mkdir -p models/Llama-2-13b

# 从 HuggingFace 下载（需先安装 git-lfs）
# git lfs install
# git clone https://huggingface.co/unsloth/Llama-2-13b models/Llama-2-13b
```

或通过其他方式将模型文件放入 `models/Llama-2-13b/` 目录。

### 构建 26B 模型

26B 模型通过修改 13B 的 config.json 深度翻倍得到：

```bash
mkdir -p models/Llama-2-26B
cp models/Llama-2-13b/*.json models/Llama-2-13b/*.model models/Llama-2-26B/

# 修改 config.json: num_hidden_layers 从 40 改为 80
# 详见 configs/pt/train_26b.yaml
```

---

## 快速开始

### 检查 XPU 状态

```bash
xpu-smi
```

### 启动训练

```bash
# 激活环境
source /root/paddlejob/workspace/env_run/liuyi39/hygon_2030/venv/bin/activate
export PYTHONPATH=/root/paddlejob/workspace/env_run/liuyi39/hygon_2030/PaddleFormers:$PYTHONPATH

# PT 预训练 - 13B (4卡)
bash scripts/pt/run.sh

# PT 预训练 - 26B (8卡)
bash scripts/pt/run_26b.sh

# SFT 全量微调 (4卡)
bash scripts/sft/run.sh

# LoRA 微调 (4卡)
bash scripts/lora/run.sh
```

---

## 项目结构

```
.
├── configs/              # 训练配置
│   ├── pt/               # 预训练配置
│   ├── sft/              # SFT 配置
│   └── lora/             # LoRA 配置
├── scripts/              # 启动脚本
├── docs/                 # 文档
├── models/               # 模型目录 (gitignore)
├── data/                 # 数据目录 (gitignore)
├── checkpoints/          # 检查点 (gitignore)
└── logs/                 # 日志 (gitignore)
```

---

## 训练监控

```bash
# 查看日志
tail -f logs/pt/train.log

# 查看 XPU 使用
watch -n 1 xpu-smi
```

---

## 注意事项

1. **XPU占用**：训练前确认 XPU 空闲
2. **显存限制**：26B 需要 8 卡
3. **数据缓存**：修改 split 配置后需清理 `data/pt/index-cache/`

---

## 参考资料

- [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers)
- [千亿模型扩张指南](docs/scaling_to_110b.md)
