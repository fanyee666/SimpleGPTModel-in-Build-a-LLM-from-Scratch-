"""
测试脚本
验证各模块组件的正确性

用法:
    python scripts/test_components.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #把上级目录加载进来

import torch
import tiktoken

from simple_gpt.config import GPT_CONFIG_124M
from simple_gpt.model import GPTModel
from simple_gpt.components import (
    LayerNorm, GELU, FeedForward, TransformerBlock, MultiHeadAttention
)
from simple_gpt.tokenizer import text_to_token_ids, token_ids_to_text
from simple_gpt.generation import generate_text_simple
from simple_gpt.training import calc_loss_batch


def test_layer_norm():
    print("\n=== 测试 LayerNorm ===")
    ln = LayerNorm(emb_dim=768)
    x = torch.randn(2, 10, 768)
    out = ln(x)
    print(f"  输入 shape: {x.shape} -> 输出 shape: {out.shape}")
    assert out.shape == x.shape
    print("  [OK] LayerNorm test passed")


def test_gelu():
    print("\n=== 测试 GELU ===")
    gelu = GELU()
    x = torch.tensor([1.0, 2.0, -1.0])
    out = gelu(x)
    print(f"  输入: {x.tolist()} -> 输出: {out.tolist()}")
    assert out.shape == x.shape
    print("  [OK] GELU test passed")


def test_feedforward():
    print("\n=== 测试 FeedForward ===")
    ff = FeedForward(GPT_CONFIG_124M)
    x = torch.randn(2, 10, 768)
    out = ff(x)
    print(f"  输入 shape: {x.shape} -> 输出 shape: {out.shape}")
    assert out.shape == x.shape
    print("  [OK] FeedForward test passed")


def test_multihead_attention():
    print("\n=== 测试 MultiHeadAttention ===")
    att = MultiHeadAttention(
        d_in=768,
        d_out=768,
        context_length=256,
        dropout=0.1,
        num_heads=12,
        qkv_bias=False
    )
    x = torch.randn(2, 10, 768)
    out = att(x)
    print(f"  输入 shape: {x.shape} -> 输出 shape: {out.shape}")
    assert out.shape == x.shape
    print("  [OK] MultiHeadAttention test passed")


def test_transformer_block():
    print("\n=== 测试 TransformerBlock ===")
    block = TransformerBlock(GPT_CONFIG_124M)
    x = torch.randn(2, 10, 768)
    out = block(x)
    print(f"  输入 shape: {x.shape} -> 输出 shape: {out.shape}")
    assert out.shape == x.shape
    print("  [OK] TransformerBlock test passed")


def test_gpt_model():
    print("\n=== 测试 GPTModel ===")
    model = GPTModel(GPT_CONFIG_124M)
    batch_size = 2
    seq_len = 10
    in_idx = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))
    out = model(in_idx)
    expected_shape = (batch_size, seq_len, GPT_CONFIG_124M["vocab_size"])
    print(f"  输入 shape: {in_idx.shape} -> 输出 shape: {out.shape}")
    assert out.shape == expected_shape, f"期望 {expected_shape}, 实际 {out.shape}"
    print("  [OK] GPTModel test passed")


def test_tokenizer():
    print("\n=== 测试 Tokenizer ===")
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, world!"
    token_ids = text_to_token_ids(text, tokenizer)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"  原文: '{text}' -> token IDs: {token_ids.shape} -> 解码: '{decoded_text}'")
    assert decoded_text == text
    print("  [OK] Tokenizer test passed")


def test_loss_calculation():
    print("\n=== 测试损失计算 ===")
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 10
    input_batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))
    target_batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    print(f"  损失值: {loss.item():.4f}")
    assert loss.item() > 0
    print("  [OK] Loss calculation test passed")


def test_text_generation():
    print("\n=== 测试文本生成 ===")
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Every effort"
    encoded = text_to_token_ids(start_text, tokenizer)
    generated = generate_text_simple(model, encoded, max_new_tokens=10, context_size=256)
    decoded = token_ids_to_text(generated, tokenizer)
    print(f"  初始文本: '{start_text}'")
    print(f"  生成文本: '{decoded}'")
    assert generated.shape[1] == encoded.shape[1] + 10
    print("  [OK] Text generation test passed")


def main():
    print("=" * 50)
    print("开始测试 simple_gpt 各组件")
    print("=" * 50)

    test_layer_norm()
    test_gelu()
    test_feedforward()
    test_multihead_attention()
    test_transformer_block()
    test_gpt_model()
    test_tokenizer()
    test_loss_calculation()
    test_text_generation()

    print("\n" + "=" * 50)
    print("All tests passed! [OK]")
    print("=" * 50)


if __name__ == "__main__":
    main()
