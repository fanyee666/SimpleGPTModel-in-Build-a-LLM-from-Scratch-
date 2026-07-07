"""
训练入口脚本
使用 the-verdict.txt 小说文本训练 GPT 模型

用法:
    python scripts/train.py

输出:
    - 训练过程中打印损失变化
    - 每轮结束后打印生成文本样本
"""

import os
import sys

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import tiktoken

from simple_gpt.config import GPT_CONFIG_124M
from simple_gpt.model import GPTModel
from simple_gpt.data_loader import create_dataloader_v1
from simple_gpt.training import (
    calc_loss_loader,
    train_model_simple
)


def load_data(data_dir, train_ratio=0.90, separator="<|endoftext|>"):
    """
    加载 data 目录下所有 .txt 文件，拼接后用 <|endoftext|> 分隔，
    然后按 train_ratio 分割训练集和验证集。

    Args:
        data_dir: 数据目录路径
        train_ratio: 训练集比例
        separator: 文本分隔符（如 <|endoftext|>）

    Returns:
        (train_data, val_data, file_info) 元组
        file_info 是字典，记录每个文件的信息
    """
    # 收集所有 .txt 文件
    txt_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ])

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    file_info = {}
    all_texts = []

    for fname in txt_files:
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # 每个文本末尾加上分隔符
        if text:
            text_with_sep = text + f"\n{separator}\n"
            all_texts.append(text_with_sep)
            file_info[fname] = len(text)

    # 拼接所有文本
    combined_text = "".join(all_texts)
    split_idx = int(train_ratio * len(combined_text))
    train_data = combined_text[:split_idx]
    val_data = combined_text[split_idx:]

    return train_data, val_data, file_info


def create_dataloaders(train_data, val_data, config, batch_size=2):
    """
    创建训练集和验证集的数据加载器

    Args:
        train_data: 训练文本
        val_data: 验证文本
        config: 模型配置字典
        batch_size: 批次大小

    Returns:
        (train_loader, val_loader) 元组
    """
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def main():
    # ============ 1. 加载数据（支持多文本 + <|endoftext|> 分隔）============
    # data_dir = os.path.join(project_root, "data")


    file_path = "./data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    #train_data, val_data, file_info = load_data(data_dir, train_ratio=0.90)
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]


    # print(f"数据加载完成：共 {len(file_info)} 个文件，总字符数 {sum(file_info.values())}")
    # for fname, flen in file_info.items():
    #     print(f"  {fname}: {flen} chars")
    # print(f"训练集 {len(train_data)} 字符，验证集 {len(val_data)} 字符")

    # ============ 2. 创建数据加载器 ============
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, GPT_CONFIG_124M, batch_size=2
    )
    print(f"数据加载器创建完成：训练集 {len(train_loader)} 批次，验证集 {len(val_loader)} 批次")

    # ============ 3. 初始化模型 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    # 设置随机种子保证可复现性
    torch.manual_seed(123)

    # ============ 4. 评估初始损失（训练前）============
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"\n训练前损失:")
    print(f"  训练集: {train_loss:.4f}")
    print(f"  验证集: {val_loss:.4f}")

    # ============ 5. 训练模型 ============
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 10

    print(f"\n开始训练 ({num_epochs} 轮)...\n")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    print("\n训练完成!")

    # ============ 6. 保存模型 ============
    # model_path = os.path.join(project_root, "checkpoints", "gpt_model_trained.pth")
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # torch.save(model.state_dict(), model_path)
    # print(f"\n模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
