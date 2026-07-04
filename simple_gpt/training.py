"""
训练模块
包含损失计算、模型训练、模型评估等功能
"""

import torch
from simple_gpt.tokenizer import text_to_token_ids, token_ids_to_text


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算单个批次的交叉熵损失

    Args:
        input_batch: 输入批次张量
        target_batch: 目标批次张量
        model: GPT 模型
        device: 计算设备

    Returns:
        标量损失值
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算数据加载器中所有（或指定数量）批次的平均损失

    Args:
        data_loader: 数据加载器
        model: GPT 模型
        device: 计算设备
        num_batches: 评估的批次数量，None 表示全部批次

    Returns:
        平均损失值
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的损失

    Args:
        model: GPT 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        eval_iter: 评估的批次数量

    Returns:
        (train_loss, val_loss) 元组
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer
):
    """
    简易模型训练函数

    Args:
        model: GPT 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        eval_freq: 评估频率（每多少步评估一次）
        eval_iter: 每次评估使用的批次数量
        start_context: 生成文本的初始上下文
        tokenizer: 分词器

    Returns:
        (train_losses, val_losses, track_tokens_seen) 元组
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # 每轮结束后打印生成样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    生成并打印文本样本

    Args:
        model: GPT 模型
        tokenizer: 分词器
        device: 计算设备
        start_context: 初始上下文文本
    """
    from simple_gpt.generation import generate_text_simple

    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))

    model.train()
