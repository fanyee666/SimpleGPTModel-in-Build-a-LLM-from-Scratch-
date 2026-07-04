"""
第五章测试：交叉熵损失计算
验证模型在固定输入下的概率分布和交叉熵损失
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from simple_gpt.config import GPT_CONFIG_124M
from simple_gpt.model import GPTModel


def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # 固定输入与目标
    inputs = torch.tensor([
        [16833, 3626, 6100],  # "every effort moves"
        [40, 1107, 588]       # "I really like"
    ])

    targets = torch.tensor([
        [3626, 6100, 345],    # " effort moves you"
        [1107, 588, 11311]    # " really like chocolate"
    ])

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print(f"Logits shape: {logits.shape}")
    print(f"Probability shape: {probas.shape}")

    # 查看目标位置的概率
    for i in range(2):
        target_probas = probas[i, [0, 1, 2], targets[i]]
        print(f"Text {i+1} target probabilities: {target_probas}")

    # 计算交叉熵损失
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f"\nCross-entropy loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
