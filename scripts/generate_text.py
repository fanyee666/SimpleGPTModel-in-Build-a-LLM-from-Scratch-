"""
生成文本脚本
使用训练好的 GPT 模型生成文本

用法:
    python scripts/generate_text.py
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import tiktoken

from simple_gpt.config import GPT_CONFIG_124M
from simple_gpt.model import GPTModel
from simple_gpt.tokenizer import text_to_token_ids, token_ids_to_text
from simple_gpt.generation import generate_text_simple


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    # 加载训练好的权重（如果存在）
    model_path = os.path.join(project_root, "checkpoints", "gpt_model_trained.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型: {model_path}")
    else:
        print("未找到训练好的模型，使用随机初始化的模型生成文本")

    model.eval()

    # 生成文本
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"
    context_size = GPT_CONFIG_124M["context_length"]

    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(f"\n生成文本:\n{generated_text}")


if __name__ == "__main__":
    main()
