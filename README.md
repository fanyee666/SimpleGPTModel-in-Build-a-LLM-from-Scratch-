# SimpleGPTModel

这是我读《从零构建大模型》照着书上代码和B站教程视频弄的一个简单的 GPT-2 小语言模型，也是我上传到 GitHub 的第一个项目。

## 运行

运行 `the_pre_trained_model.py` 即可加载预训练权重并生成文本。

**注意**：运行前需要先将 GPT-2 权重文件放到 `gpt2/124M/` 目录下（权重文件太大没有放进仓库里）。原始的自动下载代码被我注释掉了，你可以手动下载权重文件后放置到对应目录，然后直接运行。

随机数种子可以去掉以生成不同的文本，修改 `temperature`、`top_k`、`max_new_tokens` 等参数也可以生成风格各异的文本。

这个小语言模型就在我的电脑上搭建好了。

## 项目结构

```
SimpleGPTModel/
├── data/                       # 训练数据（语料文本）
│   ├── alice.txt               # 《爱丽丝梦游仙境》
│   ├── pride_and_prejudice.txt # 《傲慢与偏见》
│   ├── sherlock.txt            # 《福尔摩斯探案集》
│   └── the-verdict.txt         # 《裁决》短篇（书中示例语料）
│
├── loadPretrainingWeights/     # 预训练权重加载相关
│   ├── gpt_download.py         # 下载 OpenAI GPT-2 官方权重（已注释掉下载逻辑，优先本地加载）
│   └── load_weights.py         # 将 TensorFlow 格式的权重映射到 PyTorch 模型
│
├── scripts/                    # 脚本工具
│   ├── generate_text.py        # 文本生成脚本
│   └── test_components.py      # 组件测试脚本
│
├── simple_gpt/                 # 核心模型代码包
│   ├── __init__.py
│   ├── components.py           # Transformer 核心组件（多头注意力、层归一化、前馈网络等）
│   ├── config.py               # 模型配置（GPT_CONFIG_124M 等超参数）
│   ├── data_loader.py          # 数据加载器
│   ├── generation.py           # 基础文本生成函数
│   ├── model.py                # GPTModel 主模型定义
│   ├── new_generation_temp_topk.py  # 带 temperature + top-k 采样的生成函数
│   ├── tokenizer.py            # 文本与 token ID 互转工具
│   └── training.py             # 训练相关函数
│
├── tests/                      # 单元测试
│   ├── test_ch4_generate_text.py
│   ├── test_ch5_cross_entropy.py
│   └── test_ch5_untrained_generation.py
│
├── the_pre_trained_model.py    # 主入口：加载预训练权重并生成文本
├── train.py                    # 训练入口脚本
└── .gitignore                  # 忽略 gpt2/、.idea/、.vscode/、__pycache__/
```

## 技术栈

- Python 3
- PyTorch
- tiktoken（OpenAI 的分词器）
- TensorFlow（仅用于读取 OpenAI 原始的 TF checkpoint 格式权重）
