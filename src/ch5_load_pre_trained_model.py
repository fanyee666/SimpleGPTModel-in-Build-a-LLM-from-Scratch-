"""
加载 OpenAI GPT-2 预训练权重并生成文本的脚本。

本文件演示如何：
1. 根据配置创建 GPTModel 实例
2."""
# 加载 OpenAI GPT-2 预训练权重并生成文本的脚本。

# 本文件演示如何：
# 1. 根据配置创建 GPTModel 实例
# 2. 从本地加载 OpenAI 官方 GPT-2 预训练权重（gpt_download.py 已修改为
#    优先检查本地文件，齐全时直接加载，跳过网络下载）
# 3. 将权重映射到自定义模型架构
# 4. 使用 top-k + temperature 采样生成文本
"""
3. 将权重映射到自定义模型架构
4. 使用 top-k + temperature 采样生成文本
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #把上级目录加载进来
from simple_gpt import config
from simple_gpt.model import GPTModel
from loadPretrainingWeights.gpt_download import download_and_load_gpt2
from loadPretrainingWeights.load_weights import load_weights_into_gpt
from simple_gpt.tokenizer import token_ids_to_text, text_to_token_ids
from simple_gpt.new_generation_temp_topk import generate
import torch
import tiktoken

model_configs = config.model_configs
GPT_CONFIG_124M = config.GPT_CONFIG_124M

# 选择要加载的模型名称，并根据该模型更新基础配置
model_name = "gpt2-small (124M)"  # 当前示例使用 124M 参数的最小模型
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

# 注意：这里必须覆盖 context_length 为 1024（OpenAI GPT-2 的真实上下文长度），
# 因为基础配置中默认是 256，不足以匹配预训练权重。
# 同时必须启用 qkv_bias=True，因为原始 GPT-2 的注意力层使用了偏置项。
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# 使用合并后的配置实例化模型
gpt = GPTModel(NEW_CONFIG)

# 将模型设置为评估模式，禁用 Dropout 等训练特有行为
# 注意：行尾的分号是多余的，Python 不需要分号作为语句结束符
gpt.eval();

# 优先使用 GPU，否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从本地加载 GPT-2 预训练权重。
# gpt_download.py 已修改为优先检查本地文件，若 gpt2/124M/ 目录下
# 所有必需文件齐全则直接加载，不再发起网络请求。
# 如需重新下载权重，可恢复 gpt_download.py 中注释掉的下载逻辑。
# settings 包含模型超参数（如 n_layer, n_head 等），
# params 包含实际的权重张量。
# 注意：settings 变量被解构但后续未被使用，可以考虑用 _ 忽略。
# settings 包含模型超参数（如 n_layer, n_head 等），
# params 包含实际的权重张量。
# 注意：settings 变量被解构但后续未被使用，可以考虑用 _ 忽略。
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# 将下载的 TensorFlow 权重逐层映射到 PyTorch 模型中
load_weights_into_gpt(gpt, params)

# 将模型移动到目标设备（GPU/CPU）
# 注意：行尾的分号是多余的
gpt.to(device);

# 固定随机种子以保证生成结果可复现
torch.manual_seed(123)

# 初始化 GPT-2 的 tiktoken 分词器（与 OpenAI 预训练时使用的编码一致）
tokenizer = tiktoken.get_encoding("gpt2")

# 将起始文本编码为 token ID 并送入设备，然后调用生成函数
input_text = "The weather is good, let's"

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=15,           # 最多生成 15 个新 token
    context_size=NEW_CONFIG["context_length"],  # 模型最大上下文长度（1024）
    top_k=50,                    # 仅保留概率最高的前 50 个 token
    temperature=1.5              # 温度 >1 使分布更平坦，输出更多样化
)

# 将生成的 token ID 序列解码回文本并输出
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
