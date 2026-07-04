"""
文本生成模块
提供基于 GPT 模型的简单文本生成方法
"""

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    简单文本生成方法

    Args:
        model: GPT 模型
        idx: 初始 token ID 张量，shape: [batch_size, current_length]
        max_new_tokens: 生成的新 token 数量
        context_size: 模型支持的最大上下文长度

    Returns:
        生成的 token ID 张量，shape: [batch_size, current_length + max_new_tokens]
    """
    for _ in range(max_new_tokens):
        # 如果上下文超过模型支持的最大长度，截取最后 context_size 个 token
        idx_cond = idx[:, -context_size:]

        # 获取预测（不计算梯度）
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步的输出
        logits = logits[:, -1, :]

        # 取概率最大的 token ID
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 将生成的 token 追加到序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
