"""
加载已训练的指令微调模型（Instruction Fine-tuned Model），并进行交互式推理测试。

模型：gpt2-medium (355M) 指令微调版本
权重文件：gpt2/355M/gpt2-medium355M-sft.pth
"""

import os
import sys
import torch
import tiktoken
import re

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_gpt import config
from simple_gpt.model import GPTModel
from loadPretrainingWeights.gpt_download import download_and_load_gpt2
from loadPretrainingWeights.load_weights import load_weights_into_gpt
from simple_gpt.new_generation_temp_topk import generate
from simple_gpt.tokenizer import text_to_token_ids, token_ids_to_text


# ============================================================
# 1. 模型配置与实例化
# ============================================================
CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG = config.BASE_CONFIG.copy()
BASE_CONFIG.update(config.model_configs[CHOOSE_MODEL])

# 实例化模型（输出头仍为 50257 维，与训练时一致）
model = GPTModel(BASE_CONFIG)

# ============================================================
# 2. 加载预训练权重（355M）
# ============================================================
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
load_weights_into_gpt(model, params)

# ============================================================
# 3. 加载指令微调权重（SFT）
# ============================================================
sft_file_name = f"./gpt2/355M/{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"

if not os.path.exists(sft_file_name):
    raise FileNotFoundError(
        f"微调权重文件不存在: {sft_file_name}\n"
        "请先运行 ch7_train_for_instruction.py 完成训练。"
    )

model.load_state_dict(torch.load(sft_file_name, map_location="cpu"))
print(f"[INFO] 已加载指令微调权重: {sft_file_name}")

# 设置设备并切换到评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 初始化 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


# ============================================================
# 4. 辅助函数：格式化输入与生成回复
# ============================================================
def format_input(entry):
    """
    将指令数据格式化为模型输入 prompt。
    与训练脚本中完全一致。
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def generate_response(prompt, model, tokenizer, device,
                      max_new_tokens=256, temperature=0.0, top_k=None):
    """
    根据 prompt 生成模型回复。

    Args:
        prompt: 输入文本字符串（已格式化的指令 prompt）
        max_new_tokens: 最多生成的新 token 数
        temperature: 采样温度（0.0 为贪婪解码）
        top_k: top-k 采样参数（None 表示不限制）

    Returns:
        模型生成的回复文本（去除 prompt 部分）
    """
    model.eval()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=BASE_CONFIG["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 只保留 prompt 之后的部分
    response_text = generated_text[len(prompt):].replace("### Response:", "").strip()
    return response_text


# ============================================================
# 5. 测试用例
# ============================================================
if __name__ == "__main__":

    print("=" * 70)
    print("指令微调模型推理测试")
    print(f"模型: {CHOOSE_MODEL}")
    print(f"设备: {device}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 测试用例 1：翻译任务
    # ------------------------------------------------------------------
    test_1 = {
        "instruction": "Translate the following sentence from English to French.",
        "input": "The cat is sleeping on the sofa.",
    }
    prompt_1 = format_input(test_1)
    response_1 = generate_response(prompt_1, model, tokenizer, device)
    print(f"\n[测试 1] 翻译任务")
    print(f"指令: {test_1['instruction']}")
    print(f"输入: {test_1['input']}")
    print(f"模型回复:\n>> {response_1}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # 测试用例 2：开放式问答（无 input）
    # ------------------------------------------------------------------
    test_2 = {
        "instruction": "Explain the concept of overfitting in machine learning.",
        "input": "",
    }
    prompt_2 = format_input(test_2)
    response_2 = generate_response(prompt_2, model, tokenizer, device, max_new_tokens=200)
    print(f"\n[测试 2] 开放式问答")
    print(f"指令: {test_2['instruction']}")
    print(f"模型回复:\n>> {response_2}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # 测试用例 3：文本总结
    # ------------------------------------------------------------------
    test_3 = {
        "instruction": "Summarize the following text in one sentence.",
        "input": (
            "Large language models have revolutionized natural language processing. "
            "They can generate text, translate languages, summarize documents, and answer questions. "
            "However, they also suffer from hallucinations and bias issues."
        ),
    }
    prompt_3 = format_input(test_3)
    response_3 = generate_response(prompt_3, model, tokenizer, device, max_new_tokens=100)
    print(f"\n[测试 3] 文本总结")
    print(f"指令: {test_3['instruction']}")
    print(f"输入: {test_3['input'][:80]}...")
    print(f"模型回复:\n>> {response_3}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # 测试用例 4：代码生成
    # ------------------------------------------------------------------
    test_4 = {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "input": "",
    }
    prompt_4 = format_input(test_4)
    response_4 = generate_response(prompt_4, model, tokenizer, device, max_new_tokens=150)
    print(f"\n[测试 4] 代码生成")
    print(f"指令: {test_4['instruction']}")
    print(f"模型回复:\n>> {response_4}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # 测试用例 5：常识推理
    # ------------------------------------------------------------------
    test_5 = {
        "instruction": "Answer the following question based on common sense.",
        "input": "If you put a glass of water in the freezer, what will happen to it?",
    }
    prompt_5 = format_input(test_5)
    response_5 = generate_response(prompt_5, model, tokenizer, device, max_new_tokens=120)
    print(f"\n[测试 5] 常识推理")
    print(f"指令: {test_5['instruction']}")
    print(f"输入: {test_5['input']}")
    print(f"模型回复:\n>> {response_5}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
