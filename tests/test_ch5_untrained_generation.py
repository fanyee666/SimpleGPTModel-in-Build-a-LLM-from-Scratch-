"""
第五章测试：未训练模型生成文本
演示未训练模型输出的随机性
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
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:")
    try:
        print(token_ids_to_text(token_ids, tokenizer))
    except UnicodeEncodeError:
        # Windows终端编码问题，安全输出
        print(token_ids_to_text(token_ids, tokenizer).encode('ascii', 'replace').decode('ascii'))



if __name__ == "__main__":
    main()
