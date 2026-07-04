import torch
import numpy as np


def assign(left, right):
    """
    将右侧（numpy 数组）的值赋给左侧（PyTorch Parameter），并进行形状校验。

    Args:
        left: 目标 PyTorch Parameter，用于获取期望的形状
        right: 源 numpy 数组，包含预训练权重值

    Returns:
        一个新的 torch.nn.Parameter，包含 right 的数据

    Raises:
        ValueError: 如果 left 和 right 的形状不匹配
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    将 OpenAI GPT-2 官方预训练权重（TensorFlow 检查点格式）加载到自定义 PyTorch GPTModel 中。

    映射关系说明：
    - GPT-2 原始实现使用单一的 c_attn 权重矩阵 [emb_dim, 3*emb_dim]，
      这里将其拆分为 Query、Key、Value 三个独立矩阵
    - 所有线性层的权重需要转置（TensorFlow 使用 [in, out]，PyTorch Linear 使用 [out, in]）
    - 输出头（out_head）与词嵌入（tok_emb）共享权重，这是 GPT-2 原始设计的核心特征

    Args:
        gpt: GPTModel 实例，待加载权重的目标模型
        params: 从 TensorFlow 检查点解析出的参数字典，由 download_and_load_gpt2 返回
    """
    # 加载位置嵌入（Positional Embedding）权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])

    # 加载词嵌入（Token Embedding）权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 逐层遍历 Transformer 块，将每个块的权重映射到对应位置
    for b in range(len(params["blocks"])):
        # 从组合注意力权重 c_attn [emb_dim, 3*emb_dim] 中拆分出 Q、K、V 的权重
        # np.split 按最后一个维度（axis=-1）三等分
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)

        # 为每个注意力头加载 Query/Key/Value 的权重
        # 注意：需要转置，因为 TF 和 PyTorch 的线性层权重布局不同
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 从组合注意力偏置 c_attn [3*emb_dim] 中拆分出 Q、K、V 的偏置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)

        # 加载 Query/Key/Value 的偏置（无需转置，偏置是一维的）
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载注意力输出投影层（c_proj）的权重和偏置
        # 该层将多头注意力的输出投影回 emb_dim 维度
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 加载前馈网络（FFN）第一层的权重和偏置
        # c_fc: 将 emb_dim 扩展到 4*emb_dim（GPT-2 默认扩展倍数）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])

        # 加载前馈网络（FFN）第二层的权重和偏置
        # c_proj: 将 4*emb_dim 投影回 emb_dim
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 加载第一个层归一化（注意力前的 LayerNorm）的参数
        # g -> scale (gamma), b -> shift (beta)
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])

        # 加载第二个层归一化（前馈前的 LayerNorm）的参数
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 加载最终的层归一化参数（位于所有 Transformer 块之后）
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # 将输出头的权重设置为与词嵌入权重相同（权重共享机制）
    # 注意：这里只是复制了权重值，并未实现真正的参数共享。
    # 如果后续进行微调训练，tok_emb 和 out_head 的权重会独立更新，
    # 与原始 GPT-2 的设计（共享同一个 Parameter 对象）不完全一致。
    # 对于纯推理场景，这种差异不影响结果。
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
