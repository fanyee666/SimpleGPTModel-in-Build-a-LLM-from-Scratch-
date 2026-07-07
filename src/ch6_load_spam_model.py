"""
加载已微调的垃圾邮件分类模型，并对示例文本进行分类测试。

逻辑：
1. 实例化原始 GPTModel（vocab_size=50257，与预训练一致）
2. 加载 OpenAI GPT-2 预训练权重
3. 复现训练时的修改：替换 out_head 为 2 分类、解冻最后层
4. 加载微调保存的 state_dict
5. 用示例邮件文本测试分类效果
"""

import torch
import tiktoken
import sys
import os

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_gpt import config
from simple_gpt.model import GPTModel
from loadPretrainingWeights.gpt_download import download_and_load_gpt2
from loadPretrainingWeights.load_weights import load_weights_into_gpt


# ============================================================
# 1. 模型配置与实例化（与训练时完全一致）
# ============================================================
CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = config.BASE_CONFIG.copy()
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 实例化原始 GPT-2 模型（输出头还是 50257 维）
model = GPTModel(BASE_CONFIG)

# 加载预训练权重
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
load_weights_into_gpt(model, params)

# ============================================================
# 2. 复现训练时的结构修改（必须一致才能加载 state_dict）
# ============================================================
# 替换输出头为 2 分类（ham / spam）
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# 选择性解冻（保持与训练时一致，虽然加载权重后这些已经包含在 state_dict 中，
# 但显式设置确保与训练时结构相同）
for param in model.parameters():
    param.requires_grad = False
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
for param in model.out_head.parameters():
    param.requires_grad = True

# ============================================================
# 3. 加载微调后的权重
# ============================================================
model_path = "./gpt2/spam_classifier/review_classifier.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"微调权重文件不存在: {model_path}")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
print(f"[INFO] 已加载微调权重: {model_path}")

# 设置设备并切换到评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 初始化 tokenizer（与 GPT-2 预训练一致）
tokenizer = tiktoken.get_encoding("gpt2")

# ============================================================
# 4. 分类函数（与训练脚本中一致）
# ============================================================
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    对输入文本进行垃圾邮件分类。

    Args:
        text: 输入邮件文本字符串
        model: 已加载权重的分类模型
        tokenizer: tiktoken GPT-2 编码器
        device: 计算设备
        max_length: 最大序列长度（用于 padding/truncation）
        pad_token_id: 填充 token ID，默认 <|endoftext|> = 50256

    Returns:
        "spam" 或 "not spam"
    """
    model.eval()

    # 编码输入文本
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    # 截断到模型支持的最大长度
    if max_length is not None:
        input_ids = input_ids[:min(max_length, supported_context_length)]
    else:
        input_ids = input_ids[:supported_context_length]

    # 填充到固定长度（如果指定了 max_length）
    if max_length is not None and len(input_ids) < max_length:
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # 取最后一个 token 的输出 logits
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


# ============================================================
# 5. 测试示例文本
# ============================================================
if __name__ == "__main__":
    # 邮件最大长度，与训练时数据集中最长邮件一致
    # 注意：如果记不清确切值，可以传入 None（不 padding，只截断）
    # 但要与训练时保持一致效果最好。这里先用训练数据集中推断的值，
    # 或直接在训练脚本中打印并记录。这里假设为 120，也可以从 CSV 重新计算。
    MAX_LENGTH = 120  # 可以根据训练时的 train_dataset.max_length 调整

    test_texts = [
        # 1. 典型 spam
        (
            "You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award."
        ),
        # 2. 正常邮件
        (
            "Hey, just wanted to check if we're still on"
            " for dinner tonight? Let me know!"
        ),
        # 3. 促销语气（可能是 spam）
        (
            "Congratulations! You've been selected for a free iPhone."
            " Click here to claim your prize now!"
        ),
        # 4. 工作相关（正常）
        (
            "Can you send me the report by 3pm?"
            " I need to review it before the meeting."
        ),
        # 5. 银行诈骗（spam）
        (
            "URGENT: Your bank account has been suspended."
            " Please verify your details immediately at this link."
        ),
    ]

    print("=" * 60)
    print("垃圾邮件分类测试")
    print(f"设备: {device}")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        result = classify_review(text, model, tokenizer, device, max_length=MAX_LENGTH)
        print(f"\n[测试 {i}] -> {result}")
        print(f"  文本: {text[:80]}{'...' if len(text) > 80 else ''}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
