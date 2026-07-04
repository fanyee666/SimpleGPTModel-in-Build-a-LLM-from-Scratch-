import torch


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    基于已有 token 序列，使用自回归方式生成新 token。

    支持两种解码策略：
    - temperature + top-k 采样：当 temperature > 0 时启用，输出更随机、多样化
    - 贪婪解码：当 temperature = 0 时，总是选择概率最高的 token

    Args:
        model: GPTModel 实例
        idx: 起始 token ID 张量，shape: [batch_size, num_tokens]
        max_new_tokens: 最多生成的新 token 数量
        context_size: 模型支持的最大上下文长度，用于截断输入序列
        temperature: 温度参数，控制采样随机性
                     - 0.0: 贪婪解码（确定性输出）
                     - (0, 1): 降低随机性，输出更保守
                     - >1: 提高随机性，输出更多样化
        top_k: 仅保留概率最高的前 k 个 token 进行采样，None 表示不限制
        eos_id: 结束符（End-of-Sequence）的 token ID，遇到时停止生成。
                设为 None 则不提前停止。

    Returns:
        完整的 token ID 序列，shape: [batch_size, num_tokens + 生成数]
    """
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        # 截断输入序列，只保留最后 context_size 个 token
        # 防止序列长度超过模型最大上下文限制
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            # 获取当前 logits 中值最高的 top_k 个 token
            top_logits, _ = torch.topk(logits, top_k)
            # 取第 top_k 大的值作为阈值
            min_val = top_logits[:, -1]
            # 将所有低于阈值的 logits 设为负无穷，使其在 softmax 后概率为 0
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            # 将 logits 除以 temperature，改变概率分布的尖锐程度
            # temperature > 1: 分布变平坦，低概率 token 更容易被选中
            # temperature < 1: 分布变尖锐，高概率 token 更占主导
            logits = logits / temperature

            # Apply softmax to get probabilities
            # 将调整后的 logits 转换为概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            # 根据概率分布随机采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            # temperature = 0 时，退化为贪婪解码：选择概率最高的 token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果生成的 token 是结束符，则提前停止生成
        # 注意：当 batch_size > 1 时，idx_next 是一个二维张量，此处直接比较
        # 可能会在部分样本已遇到 eos 而部分未遇到时产生歧义错误。
        # 当前实现仅适用于 batch_size = 1 的场景。
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        # 将新生成的 token 追加到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
