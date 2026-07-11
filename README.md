# SimpleGPTModel

这是我读《从零构建大模型》（*Build a Large Language Model From Scratch*）照着书上代码和B站教程视频实现的一个 GPT-2 小语言模型，也是我上传到 GitHub 的第一个项目。以下的README由Kimi生成。

## 已完成的内容

| 章节 | 内容 |
|------|------|
| 第 5 章 | 实现完整 GPT-2 架构，加载 OpenAI 预训练权重，文本生成（贪婪解码 + temperature/top-k 采样） |
| 第 6 章 | 垃圾邮件分类微调（参数高效微调：替换输出头为 2 分类，仅解冻最后 Transformer 块 + 输出头 + 层归一化） |
| 第 7 章 | 指令微调（Instruction Fine-tuning）：基于 355M 模型，使用 Alpaca 格式数据集，实现任务理解与指令遵循 |

---

## 核心模块说明（`simple_gpt/`）

本项目从零手写了 GPT-2 的完整架构，没有使用 Hugging Face Transformers。以下是各模块的作用：

### `config.py` — 模型配置

定义模型的超参数：

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # 词汇表大小
    "context_length": 256,    # 上下文长度
    "emb_dim": 768,           # 嵌入维度
    "n_heads": 12,            # 注意力头数
    "n_layers": 12,           # Transformer 块层数
    "drop_rate": 0.1,         # Dropout 概率
    "qkv_bias": False         # QKV 是否使用偏置
}
```

另外定义了 `model_configs`（不同体量模型的超参数）和 `BASE_CONFIG`（用于下游任务微调的基础配置，`context_length=1024`，`qkv_bias=True`，`drop_rate=0.0`）。

### `components.py` — Transformer 核心组件

从零实现的 5 个核心组件：

| 类 | 作用 |
|----|------|
| `MultiHeadAttention` | 多头自注意力，含因果掩码（Causal Mask），支持 `qkv_bias` |
| `LayerNorm` | 层归一化，含可学习的 `scale`（γ）和 `shift`（β）参数 |
| `GELU` | GELU 激活函数（使用 tanh 近似版本） |
| `FeedForward` | 前馈网络：Linear(emb_dim → 4×emb_dim) → GELU → Linear(4×emb_dim → emb_dim) |
| `TransformerBlock` | 完整的 Transformer 块：Norm → 多头注意力 → 残差连接 → Norm → FFN → 残差连接 |

### `model.py` — GPT 主模型

`GPTModel` 类，按顺序拼接：

```
Token Embedding → Positional Embedding → Dropout
→ TransformerBlock × n_layers → LayerNorm → Linear(输出到 vocab_size)
```

`forward` 返回 shape 为 `[batch_size, seq_len, vocab_size]` 的 logits。

### `tokenizer.py` — 文本与 Token ID 互转

```python
def text_to_token_ids(text, tokenizer):
    # 将字符串转为 shape [1, num_tokens] 的张量

def token_ids_to_text(token_ids, tokenizer):
    # 将 token ID 张量解码回字符串
```

使用 OpenAI 的 `tiktoken`（`gpt2` 编码）作为分词器。

### `generation.py` — 贪婪解码生成

`generate_text_simple(model, idx, max_new_tokens, context_size)`：
- 每步取 logits 中概率最大的 token（`argmax`）
- 自动截断超过 `context_size` 的上下文
- 确定性输出，结果可复现

### `new_generation_temp_topk.py` — 高级采样生成

`generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None)`：
- `temperature=0.0`：贪婪解码（与上面一致）
- `temperature>0`：对 logits 除以 temperature 后做 softmax，再按概率采样
- `top_k`：只保留概率最高的前 k 个 token，其余设为 `-inf`
- `eos_id`：遇到结束符 `<|endoftext|>` 时提前停止生成

### `data_loader.py` — 训练数据加载

```python
class GPTDatasetV1(Dataset):
    # 使用滑动窗口将长文本切分为 (input, target) 对

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128):
    # 创建 DataLoader，stride 控制滑动窗口步长
```

### `training.py` — 训练工具函数

| 函数 | 作用 |
|------|------|
| `calc_loss_batch` | 计算单个批次的交叉熵损失 |
| `calc_loss_loader` | 计算数据加载器上指定批次的平均损失 |
| `evaluate_model` | 在训练集和验证集上评估损失 |
| `train_model_simple` | 完整的训练循环，含定期评估和生成样本打印 |
| `generate_and_print_sample` | 每轮训练结束后打印一段生成文本，直观观察训练效果 |

### `__init__.py` — 包入口

导出所有常用类和函数，方便 `from simple_gpt import ...` 一键导入。

---

## 权重加载（`loadPretrainingWeights/`）

### `gpt_download.py`

下载并加载 OpenAI GPT-2 官方权重：
- 先检查本地 `gpt2/{model_size}/` 目录是否已有完整权重文件
- **缺什么下载什么**：只下载缺失的文件，已存在且大小一致则跳过
- 使用 `tqdm` 显示下载进度
- 支持 124M / 355M / 774M / 1558M 四种模型

### `load_weights.py`

将 TensorFlow 格式的权重逐层映射到 PyTorch `GPTModel`：
- 拆分 `c_attn` 组合权重为独立的 Q、K、V 矩阵（注意转置）
- 加载前馈网络两层、两个 LayerNorm、输出投影层的参数
- 输出头 `out_head` 与词嵌入 `tok_emb` **共享权重**（GPT-2 原始设计）

---

## 各章节脚本与运行方式

### 第 5 章：加载预训练权重并生成文本

```bash
python src/ch5_load_pre_trained_model.py
```

作用：加载 GPT-2（默认 124M）预训练权重，输入 `"Every effort moves you"`，使用贪婪解码和 temperature/top-k 采样生成文本。

关键代码片段：

```python
from simple_gpt.model import GPTModel
from simple_gpt.tokenizer import text_to_token_ids, token_ids_to_text
from simple_gpt.new_generation_temp_topk import generate

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)  # 加载预训练权重
model.eval()

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=1024,
    top_k=50,
    temperature=1.5
)
print(token_ids_to_text(token_ids, tokenizer))
```

### 第 6 章：垃圾邮件分类微调

#### 数据预处理

```bash
python src/ch6_spam_data_preprocess.py
```

作用：读取 `SMSSpamCollection.tsv`，平衡 ham/spam 数量，随机划分为 train/val/test CSV 文件。

#### 训练微调模型

```bash
python src/ch6_train_for_spam.py
```

作用：
1. 加载 GPT-2 124M 预训练权重
2. 冻结所有参数
3. **替换输出头**：`model.out_head = nn.Linear(768, 2)`（ham=0, spam=1）
4. **解冻**最后一个 Transformer 块 + 最终层归一化 + 输出头
5. 使用 AdamW 优化器训练 5 轮
6. 保存微调权重到 `gpt2/spam_classifier/review_classifier.pth`

关键代码片段：

```python
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换输出头为 2 分类
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)

# 选择性解冻
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# ...
```

#### 加载并测试分类模型

```bash
python src/ch6_load_spam_model.py
```

作用：加载 `review_classifier.pth`，对测试邮件文本进行 spam / not spam 判断。

关键代码片段：

```python
model.load_state_dict(torch.load("./gpt2/spam_classifier/review_classifier.pth"))
model.eval()

# 取最后一个 token 的输出 logits 做分类
text = "You are a winner you have been selected..."
input_ids = tokenizer.encode(text) + [pad_token_id] * (max_length - len(input_ids))
logits = model(input_tensor)[:, -1, :]  # 最后一帧
predicted = torch.argmax(logits, dim=-1).item()  # 0=ham, 1=spam
```

### 第 7 章：指令微调

#### 数据格式化与划分

```bash
python src/ch7_instruction_data_process.py
```

作用：下载 `instruction-data.json`，将每条数据格式化为：

```
Below is an instruction that describes a task. Write a response...

### Instruction:
{instruction}

### Input:
{input}   # 可选

### Response:
{output}
```

并划分为训练集（85%）、验证集（5%）、测试集（10%）。

#### 指令微调训练

```bash
python src/ch7_train_for_instruction.py
```

作用：
1. 加载 **GPT-2 Medium (355M)** 预训练权重
2. 使用 `InstructionDataset` + `custom_collate_fn`（只对第一个 `<|endoftext|>` 计算损失，其余 padding 用 `-100` 忽略）
3. 全参数微调 2 轮，学习率 `5e-5`
4. 保存微调权重到 `gpt2/355M/gpt2-medium355M-sft.pth`

关键代码片段：

```python
# 自定义 collate_fn：忽略多余的 padding token 损失
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index  # 只保留第一个 eos，其余忽略

# 训练
CHOOSE_MODEL = "gpt2-medium (355M)"
settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
# ...
```

#### 加载指令微调模型并测试

```bash
python src/ch7_load_instruction_model.py
```

作用：加载 `gpt2-medium355M-sft.pth`，对 5 个测试用例（翻译、问答、总结、代码生成、常识推理）进行推理，输出模型回复。

关键代码片段：

```python
model.load_state_dict(torch.load("./gpt2/355M/gpt2-medium355M-sft.pth"))
model.eval()

prompt = format_input({"instruction": "...", "input": "..."})
token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer),
    max_new_tokens=256,
    context_size=1024,
    eos_id=50256  # 遇到 <|endoftext|> 停止
)
response = token_ids_to_text(token_ids, tokenizer)[len(prompt):].strip()
```

---

## 项目结构

```
SimpleGPTModel/
├── data/                           # 训练数据
│   ├── sms+spam+collection/        # 垃圾邮件分类数据集
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── instruction-data.json      # 指令微调数据集（第7章）
│   ├── alice.txt
│   ├── pride_and_prejudice.txt
│   ├── sherlock.txt
│   └── the-verdict.txt
│
├── loadPretrainingWeights/        # 预训练权重加载
│   ├── gpt_download.py            # 下载/加载 OpenAI GPT-2 权重（支持断点续传）
│   └── load_weights.py            # TF → PyTorch 权重映射
│
├── simple_gpt/                    # 核心模型代码包（从零手写）
│   ├── __init__.py                # 包入口，导出常用 API
│   ├── components.py              # Transformer 组件（注意力、归一化、FFN）
│   ├── config.py                  # 超参数配置（GPT_CONFIG_124M、model_configs）
│   ├── data_loader.py             # 滑动窗口数据集 + DataLoader
│   ├── generation.py              # 贪婪解码文本生成
│   ├── model.py                   # GPTModel 主模型
│   ├── new_generation_temp_topk.py # temperature + top-k 采样生成
│   ├── tokenizer.py               # 文本 ↔ token ID 互转
│   └── training.py               # 损失计算、训练循环、评估
│
├── src/                           # 各章节主脚本
│   ├── ch5_load_pre_trained_model.py   # 第5章：加载预训练权重并生成
│   ├── ch5_train.py                    # 第5章：从头训练
│   ├── ch6_spam_data_preprocess.py    # 第6章：垃圾邮件数据预处理
│   ├── ch6_train_for_spam.py          # 第6章：垃圾邮件分类微调训练
│   ├── ch6_load_spam_model.py         # 第6章：加载微调模型并分类测试
│   ├── ch7_instruction_data_process.py # 第7章：指令数据格式化与划分
│   ├── ch7_train_for_instruction.py   # 第7章：指令微调训练
│   └── ch7_load_instruction_model.py  # 第7章：加载指令模型并测试
│
├── scripts/                       # 工具脚本
│   ├── generate_text.py
│   └── test_components.py
│
├── tests/                         # 单元测试
│   ├── test_ch4_generate_text.py
│   ├── test_ch5_cross_entropy.py
│   └── test_ch5_untrained_generation.py
│
├── the_pre_trained_model.py      # 早期入口（第5章前）
├── train.py                      # 早期训练入口
├── .gitignore                    # 忽略 gpt2/、.idea/、__pycache__/
└── README.md                     # 本文件
```

---

## 权重文件说明

GPT-2 预训练权重文件体积很大，**未提交到 GitHub**，需要自行准备：

### 自动下载（推荐）

`loadPretrainingWeights/gpt_download.py` 已恢复下载逻辑：
- 检查 `gpt2/{model_size}/` 目录下的 7 个必需文件
- 只下载缺失的文件，已存在且大小一致则跳过
- 在国内网络环境下可能连接 Azure Blob Storage 较慢，如超时可手动下载

### 手动下载

从 OpenAI 官方存储桶下载对应模型的 7 个文件，放置到 `gpt2/124M/` 或 `gpt2/355M/` 等目录：
```
checkpoint
encoder.json
hparams.json
model.ckpt.data-00000-of-00001
model.ckpt.index
model.ckpt.meta
vocab.bpe
```

### 微调权重保存位置

| 任务 | 保存路径 |
|------|---------|
| 垃圾邮件分类 | `gpt2/spam_classifier/review_classifier.pth` |
| 指令微调 | `gpt2/355M/gpt2-medium355M-sft.pth` |

---

## 技术栈

- **Python 3**
- **PyTorch** — 深度学习框架
- **tiktoken** — OpenAI GPT-2 分词器
- **TensorFlow** — 仅用于读取 OpenAI 原始 TF checkpoint 格式权重
- **pandas** — 垃圾邮件数据处理
- **numpy** — 数值计算
