"""
数据加载模块
提供 GPT 训练所需的数据集和数据加载器
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    GPT 训练数据集
    使用滑动窗口将文本切分为重叠的序列
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt: 原始文本字符串
            tokenizer: tiktoken 分词器
            max_length: 每个序列的最大长度（上下文长度）
            stride: 滑动窗口步长
        """
        self.input_ids = []
        self.target_ids = []

        # 将整个文本分词并编码成词元 ID
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口切分文本为重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # 目标序列向右偏移一个位置（预测下一个 token）
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    """
    创建 GPT 训练数据加载器

    Args:
        txt: 原始文本字符串
        batch_size: 批次大小
        max_length: 每个序列的最大长度
        stride: 滑动窗口步长
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的批次
        num_workers: 数据加载的进程数

    Returns:
        DataLoader 对象
    """
    # 初始化 GPT-2 分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
