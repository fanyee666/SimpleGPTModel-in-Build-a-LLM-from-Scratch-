"""
第四章测试：生成文本（未训练模型）
验证未训练模型的输出效果
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import tiktoken
from simple_gpt.config import GPT_CONFIG_124M
from simple_gpt.model import GPTModel
from simple_gpt.generation import generate_text_simple
from simple_gpt.tokenizer import text_to_token_ids, token_ids_to_text


def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 禁用 dropout

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_tensor = text_to_token_ids(start_context, tokenizer)

    print(f"\n{'='*50}\n{' '*22}INPUT\n{'='*50}")
    print(f"Input text: {start_context}")
    print(f"Encoded shape: {encoded_tensor.shape}")

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    decoded_text = token_ids_to_text(out, tokenizer)

    print(f"\n{'='*50}\n{' '*22}OUTPUT\n{'='*50}")
    print(f"Output text: {decoded_text}")
    print(f"Output length: {out.shape[1]} tokens")


if __name__ == "__main__":
    main()
