# Llama-2 XPU 训练项目

在国产 XPU（昆仑芯 P800）上验证大模型训练能力。

---

## 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source /root/paddlejob/workspace/env_run/liuyi39/hygon_2030/venv/bin/activate

# 设置环境变量
export PYTHONPATH=/root/paddlejob/workspace/env_run/liuyi39/hygon_2030/PaddleFormers:$PYTHONPATH
```

### 2. 启动训练

#### PT 预训练 - 13B (4卡)
```bash
bash scripts/pt/run.sh
```

#### PT 预训练 - 26B (8卡)
```bash
bash scripts/pt/run_26b.sh
```

#### SFT 全量微调 (4卡)
```bash
bash scripts/sft/run.sh
```

#### LoRA 微调 (4卡)
```bash
bash scripts/lora/run.sh
```

---

## 项目结构

```
.
├── configs/              # 训练配置
│   ├── pt/               # 预训练配置
│   │   ├── train.yaml        # 13B 基准配置
│   │   └── train_26b.yaml    # 26B 2倍扩张配置
│   ├── sft/              # SFT 配置
│   │   └── train.yaml
│   └── lora/             # LoRA 配置
│       ├── train.yaml
│       └── export.yaml
├── scripts/              # 启动脚本
│   ├── pt/
│   │   ├── run.sh
│   │   └── run_26b.sh
│   ├── sft/
│   │   └── run.sh
│   └── lora/
│       └── run.sh
├── docs/                 # 文档
│   └── scaling_to_110b.md   # 千亿模型扩张指南
├── models/               # 模型目录 (gitignore)
│   ├── Llama-2-13b/
│   └── Llama-2-26B/
├── data/                 # 数据目录 (gitignore)
│   ├── pt/               # 预训练数据
│   └── sft/              # SFT数据
├── checkpoints/          # 检查点 (gitignore)
└── logs/                 # 日志 (gitignore)
```

---

## 配置说明

### 并行策略

| 模型 | 并行配置 | 卡数 |
|-----|---------|------|
| 13B | TP=2, PP=2, Sharding=stage1 | 4 |
| 26B | TP=4, PP=2, Sharding=stage1 | 8 |

### 关键参数

```yaml
# 数据
max_seq_len: 4096          # 序列长度
dataset_type: pretrain      # 数据类型 (pretrain/erniekit)

# 训练
max_steps: 100              # 最大步数
learning_rate: 1e-4         # 学习率
gradient_accumulation_steps: 4

# 并行
tensor_model_parallel_size: 2
pipeline_model_parallel_size: 2
sharding: stage1

# 显存优化
recompute_granularity: full
bf16: true
```

---

## 模型扩张

### 26B 模型构建

基于13B深度翻倍（40层→80层）：

```json
{
  "num_hidden_layers": 80,    // 40 × 2
  "hidden_size": 5120,        // 不变
  "intermediate_size": 13824, // 不变
  "num_attention_heads": 40   // 不变
}
```

参数量：~26B (13B × 2)

### 更多扩张方案

参见 [docs/scaling_to_110b.md](docs/scaling_to_110b.md)

---

## 验证检查

### 训练前检查

```bash
# 1. 检查XPU状态
xpu-smi

# 2. 清理残留进程
pkill -9 -f paddleformers-cli

# 3. 清理数据缓存
rm -rf ./data/pt/index-cache/
```

### 训练中监控

```bash
# 查看日志
tail -f logs/pt/train.log

# 查看XPU使用
watch -n 1 xpu-smi
```

### 训练结果

```bash
# 查看训练结果
cat checkpoints/pt/llama2-13b-pretrain/all_results.json

# 示例输出：
# {
#     "train_loss": 8.496,
#     "train_runtime": 857.77,
#     "train_steps_per_second": 0.1166
# }
```

---

## 注意事项

1. **XPU占用**：训练前确认XPU空闲，避免与他人冲突
2. **显存限制**：26B需要8卡，确保有足够资源
3. **数据缓存**：修改split配置后需清理index-cache
4. **评估策略**：评估时需确保验证集数据足够

---

## 当前状态

- ✅ PT 预训练 (13B) - 已跑通
- ✅ PT 预训练 (26B) - 配置完成，待验证
- ✅ SFT 全量微调 - 已跑通
- ✅ LoRA 微调 - 已跑通
- ⏳ 等比扩缩验证 - 进行中

---

## 参考资料

- [千亿模型扩张指南](docs/scaling_to_110b.md)
- [PaddleFormers 文档](https://github.com/PaddlePaddle/PaddleFormers)
