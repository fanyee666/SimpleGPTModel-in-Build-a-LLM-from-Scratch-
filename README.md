# SimpleGPTModel

这是我读《从零构建大模型》（*Build a Large Language Model From Scratch*）照着书上代码和B站教程视频实现的一个 GPT-2 小语言模型，也是我上传到 GitHub 的第一个项目。

## 已完成的内容

### 第 5 章：加载预训练权重与文本生成
- 实现了完整的 GPT-2 模型架构（嵌入层、位置编码、Transformer 块、输出头）
- 从 OpenAI 官方加载 GPT-2 预训练权重（124M 参数）
- 支持贪婪解码和 temperature + top-k 采样生成文本

### 第 6 章：垃圾邮件分类微调（Spam Classification）
- 使用 SMS Spam Collection 数据集进行二分类微调
- 将 GPT-2 输出头替换为 2 分类输出（ham / spam）
- 采用**参数高效微调**：冻结大部分层，仅解冻最后一个 Transformer 块 + 最终层归一化 + 输出头
- 训练完成后保存微调权重，支持独立加载与推理
- 实现了分类测试函数，可直接对邮件文本进行 spam / not spam 判断

### 第 7 章：指令微调数据准备（Instruction Fine-tuning）
- 下载并加载指令数据集（`instruction-data.json`）
- 实现数据格式化函数：将 JSON 转换为 `### Instruction`、`### Input`、`### Response` 格式
- 完成训练 / 验证 / 测试集划分（85% / 5% / 10%）
- 为后续指令微调（Instruction Fine-tuning）做准备

## 运行

### 加载预训练权重并生成文本
```bash
python src/ch5_load_pre_trained_model.py
```

### 垃圾邮件分类训练（微调）
```bash
python src/ch6_train_for_spam.py
```

### 加载微调后的分类模型并测试
```bash
python src/ch6_load_spam_model.py
```

### 指令数据格式化与划分
```bash
python src/ch7_instruction_data_process.py
```

**注意**：运行前需要先将 GPT-2 权重文件放到 `gpt2/124M/` 目录下（权重文件太大没有放进仓库）。`gpt_download.py` 已恢复自动下载逻辑：检查本地缺失的文件，只下载缺失部分，文件齐全时直接加载。如果网络连接 Azure Blob Storage 超时，仍需手动下载后放置到对应目录。

## 项目结构

```
SimpleGPTModel/
├── data/                           # 训练数据
│   ├── sms+spam+collection/        # 垃圾邮件分类数据集（train/val/test CSV）
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── instruction-data.json      # 指令微调数据集（第7章）
│   ├── alice.txt                   # 《爱丽丝梦游仙境》
│   ├── pride_and_prejudice.txt    # 《傲慢与偏见》
│   ├── sherlock.txt               # 《福尔摩斯探案集》
│   └── the-verdict.txt            # 《裁决》短篇（书中示例语料）
│
├── loadPretrainingWeights/        # 预训练权重加载相关
│   ├── gpt_download.py            # 下载/加载 OpenAI GPT-2 官方权重（支持断点续传）
│   └── load_weights.py            # 将 TensorFlow 格式的权重映射到 PyTorch 模型
│
├── simple_gpt/                    # 核心模型代码包
│   ├── __init__.py
│   ├── components.py              # Transformer 核心组件（多头注意力、层归一化、前馈网络等）
│   ├── config.py                  # 模型配置（GPT_CONFIG_124M、BASE_CONFIG 等超参数）
│   ├── data_loader.py             # 数据加载器
│   ├── generation.py              # 基础文本生成函数（贪婪解码）
│   ├── model.py                   # GPTModel 主模型定义
│   ├── new_generation_temp_topk.py  # 带 temperature + top-k 采样的生成函数
│   ├── tokenizer.py               # 文本与 token ID 互转工具
│   └── training.py               # 训练相关函数（损失计算、评估等）
│
├── src/                           # 各章节主脚本
│   ├── ch5_load_pre_trained_model.py   # 第5章：加载预训练权重并生成文本
│   ├── ch5_train.py                    # 第5章：训练脚本
│   ├── ch6_spam_data_preprocess.py    # 第6章：垃圾邮件数据预处理与划分
│   ├── ch6_train_for_spam.py          # 第6章：垃圾邮件分类微调训练
│   ├── ch6_load_spam_model.py         # 第6章：加载微调模型并测试分类
│   └── ch7_instruction_data_process.py # 第7章：指令数据格式化与训练验证测试划分
│
├── scripts/                       # 脚本工具
│   ├── generate_text.py           # 文本生成脚本
│   └── test_components.py         # 组件测试脚本
│
├── tests/                         # 单元测试
│   ├── test_ch4_generate_text.py
│   ├── test_ch5_cross_entropy.py
│   └── test_ch5_untrained_generation.py
│
├── the_pre_trained_model.py      # 早期入口：加载预训练权重并生成文本
├── train.py                      # 早期训练入口脚本
├── .gitignore                    # 忽略 gpt2/、.idea/、.vscode/、__pycache__/
└── README.md                     # 本文件
```

## 技术栈

- Python 3
- PyTorch
- tiktoken（OpenAI 的分词器）
- TensorFlow（仅用于读取 OpenAI 原始的 TF checkpoint 格式权重）
- pandas（数据处理）
- numpy
