"""
本文件做测试，包含了从加载数据集到试着生成文本，到未训练时计算准确度，再到微调成功
代码有些长，包含简单的测试，你可以将这些测试的代码去掉
其实在Jupyter Notebook上是最好写的
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #把上级目录加载进来
from simple_gpt import config
from simple_gpt.model import GPTModel
from simple_gpt.generation import generate_text_simple
from loadPretrainingWeights.gpt_download import download_and_load_gpt2
from loadPretrainingWeights.load_weights import load_weights_into_gpt
from simple_gpt.tokenizer import token_ids_to_text, text_to_token_ids
from simple_gpt.new_generation_temp_topk import generate
import torch
import tiktoken

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        将所有邮件填充<|endoftext|>至相同的长度
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        """
        找长度最大的邮件
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
# 创建三个数据集对象
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(
    csv_file="./data/sms+spam+collection/train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file="./data/sms+spam+collection/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file="./data/sms+spam+collection/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
# print(train_dataset.max_length)


#from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8 # 8条邮件为一批

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# 查看加载到dataloader中的数据形状
print("Train loader:")
for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")


CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = config.BASE_CONFIG

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 这是做什么的？
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

# 建立模型
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")  #124M
settings,params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

#///////////////////////////////////////////////////////////////////////
# 试着生成一下文本
#///////////////////////////////////////////////////////////////////////

# text_1 = "Every effort moves you"

# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=15,
#     context_size=BASE_CONFIG["context_length"]
# )

# print(token_ids_to_text(token_ids, tokenizer))

# # 再做一个测试，说明模型不会判断，只会继续生成文本
# text_2 = (
#     "Is the following text 'spam'? Answer with 'yes' or 'no':"
#     " 'You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award.'"
# )

# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_2, tokenizer),
#     max_new_tokens=23,
#     context_size=BASE_CONFIG["context_length"]
# )

# print(token_ids_to_text(token_ids, tokenizer))


#///////////////////////////////////////////////////////////////////////
# 接下来开始微调模型
#///////////////////////////////////////////////////////////////////////

# 对所有参数禁用梯度更新
for param in model.parameters():
    param.requires_grad = False

# 将输出的维度改为2
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# 将输出层和最后一个Transformer块设为可训练的部分
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    计算判断的准确度
    """
    model.eval()
    correct_predictions, num_examples = 0, 0
    # 自定义在多少批里算，None就以所有批次算，不为None就以自定义的批次数算
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # For reproducibility due to the shuffling in the training data loader

# 计算初始的判断准确度，应该在50%左右，和瞎猜没什么区别
# train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
# val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
# test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

# print(f"Training accuracy: {train_accuracy*100:.2f}%")
# print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 计算交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    # 这里和第五章的计算损失是很相似的，见the_pre_trained_model.py
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# 查看未训练前，模型对训练集、验证集和测试集的损失
# with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
#     train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
#     val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
#     test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

# print(f"Training loss: {train_loss:.3f}")
# print(f"Validation loss: {val_loss:.3f}")
# print(f"Test loss: {test_loss:.3f}")

# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # # Optional evaluation step
            # if global_step % eval_freq == 0:
            #     train_loss, val_loss = evaluate_model(
            #         model, train_loader, val_loader, device, eval_iter)
            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)
            #     print(f"Ep {epoch+1} (Step {global_step:06d}): "
            #           f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


#///////////////////////////////////////////////////////////////////////////////////
# 开始训练
#///////////////////////////////////////////////////////////////////////////////////
import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
print("\n开始训练...")
print(f"使用设备: {device}")
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


# 训练之后计算准确度
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")



def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    进行分类，返回ham和spam字符串
    """
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

# 第一次测试，这是一个spam
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print("The first test is " + classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# 第二次测试，这不是spam
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print("The second test is " +classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

"""
这么长的文件目的就是这一行，将模型保存下来
"""
torch.save(model.state_dict(), "./gpt2/spam_classifier/review_classifier.pth")