"""
分词器工具模块
提供文本与 token ID 之间的转换功能
"""

import torch


def text_to_token_ids(text, tokenizer):
    """
    将文本转换为 token ID 张量（包含 batch 维度）
    
    Args:
        text: 输入文本字符串
        tokenizer: tiktoken 分词器
    
    Returns:
        包含 batch 维度的 token ID 张量，shape: [1, num_tokens]
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将 token ID 张量转换为文本
    
    Args:
        token_ids: token ID 张量，shape: [1, num_tokens] 或 [num_tokens]
        tokenizer: tiktoken 分词器
    
    Returns:
        解码后的文本字符串
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
