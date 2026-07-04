"""
GPT 模型模块
定义完整的 GPT 语言模型
"""

import torch
import torch.nn as nn
from simple_gpt.components import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    """
    GPT 语言模型

    架构包含：
    - 词嵌入层 (Token Embedding)
    - 位置嵌入层 (Positional Embedding)
    - Dropout 正则化
    - 多个 Transformer 块
    - 最终的层归一化
    - 输出头 (映射到词汇表大小)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: 配置字典，包含 vocab_size, context_length, emb_dim, n_layers 等
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        前向传播

        Args:
            in_idx: 输入 token ID 张量，shape: [batch_size, seq_len]

        Returns:
            logits: 输出 logits 张量，shape: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape: [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
