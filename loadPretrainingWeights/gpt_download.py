# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# 本文件已修改：优先检查本地已下载的权重文件，如果齐全则直接加载，
# 跳过网络下载步骤，避免 Azure Blob Storage 连接超时问题。
# 原始下载逻辑已注释保留，如需重新下载可取消注释。

import os

# import urllib.request  # 保留但不再使用，本地加载不需要网络请求
# import requests        # 保留备用
import json
import numpy as np
import tensorflow as tf
# from tqdm import tqdm  # 保留但不再使用，本地加载不需要进度条


def download_and_load_gpt2(model_size, models_dir):
    """
    加载 GPT-2 预训练权重。

    修改后的逻辑：
    1. 先检查本地 models_dir/model_size/ 目录下是否已有完整的权重文件
    2. 如果齐全，直接调用 TensorFlow 加载本地 checkpoint，跳过所有网络请求
    3. 如果不齐全，抛出 FileNotFoundError，提示用户手动补全文件

    原始下载逻辑（从 Azure Blob Storage 下载）已注释保留在下方，
    如需重新下载可取消注释恢复。

    Args:
        model_size: 模型尺寸，可选 "124M" / "355M" / "774M" / "1558M"
        models_dir: 权重文件存放的根目录，预期结构为 models_dir/model_size/

    Returns:
        settings: 从 hparams.json 读取的模型超参数字典
        params: 解析后的权重参数嵌套字典
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # ------------------------------------------------------------------
    # 新增：本地文件完整性检查 —— 如果齐全则直接加载，跳过下载
    # ------------------------------------------------------------------
    missing_files = []
    for filename in filenames:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)

    if missing_files:
        # 本地文件不完整，抛出清晰错误（不再尝试联网下载，避免超时）
        raise FileNotFoundError(
            f"本地 GPT-2 {model_size} 权重文件不完整，缺少以下文件：\n"
            + "\n".join(f"  - {f}" for f in missing_files)
            + f"\n\n请确保以下所有文件存在于目录：{model_dir}\n"
            + "或手动从 OpenAI 公开存储桶下载后放置到该目录。\n"
            + "（原始自动下载逻辑已注释保留在本文件中，可取消注释恢复。）"
        )

    # 文件齐全，直接加载本地权重
    print(f"[INFO] 本地权重文件已齐全，直接加载: {model_dir}")
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params

    # ------------------------------------------------------------------
    # 原始下载逻辑（已注释保留）
    # ------------------------------------------------------------------
    # base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # os.makedirs(model_dir, exist_ok=True)
    # for filename in filenames:
    #     file_url = os.path.join(base_url, model_size, filename)
    #     file_path = os.path.join(model_dir, filename)
    #     download_file(file_url, file_path)
    #
    # # Load settings and params
    # tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    # params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    #
    # return settings, params


# ------------------------------------------------------------------
# 原始下载辅助函数（已注释保留）
# ------------------------------------------------------------------

# def download_file(url, destination):
#     """
#     从指定 URL 下载文件到本地路径，支持断点续传检查。
#
#     注意：此函数需要联网访问 Azure Blob Storage，在国内网络环境下
#     可能连接超时或速度极慢。本地文件已齐全时不会调用此函数。
#     """
#     # Send a GET request to download the file
#     with urllib.request.urlopen(url) as response:
#         # Get the total file size from headers, defaulting to 0 if not present
#         file_size = int(response.headers.get("Content-Length", 0))
#
#         # Check if file exists and has the same size
#         if os.path.exists(destination):
#             file_size_local = os.path.getsize(destination)
#             if file_size == file_size_local:
#                 print(f"File already exists and is up-to-date: {destination}")
#                 return
#
#         # Define the block size for reading the file
#         block_size = 1024  # 1 Kilobyte
#
#         # Initialize the progress bar with total file size
#         progress_bar_description = os.path.basename(url)  # Extract filename from URL
#         with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
#             # Open the destination file in binary write mode
#             with open(destination, "wb") as file:
#                 # Read the file in chunks and write to destination
#                 while True:
#                     chunk = response.read(block_size)
#                     if not chunk:
#                         break
#                     file.write(chunk)
#                     progress_bar.update(len(chunk))  # Update progress bar


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    从 TensorFlow checkpoint 文件中解析权重参数。

    遍历 checkpoint 中的所有变量，按层次结构组织成嵌套字典：
    - 顶层: wpe, wte, g, b
    - blocks: 按层号组织，每层包含 attn, mlp, ln_1, ln_2 等

    Args:
        ckpt_path: TensorFlow checkpoint 路径前缀（不含后缀）
        settings: 模型超参数字典，用于确定层数

    Returns:
        params: 嵌套字典，键名结构与 load_weights_into_gpt 期望的格式一致
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
