"""
本文件用于加载垃圾邮件数据集，做一个数据预处理
"""
import pandas as pd
import os

# 
data_file_path = "./data/sms+spam+collection/SMSSpamCollection.tsv"

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
# print(df)
# print(df["Label"].value_counts())

def create_balanced_dataset(df):
    """
    让ham和spam邮件的数量保持相等，让模型接收均匀信息
    """
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

balanced_df = create_balanced_dataset(df)
# 将ham和spam转成0和1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
# print(balanced_df["Label"].value_counts())

def random_split(df, train_frac, validation_frac):
    """
    将数据集分为训练、验证和测试集
    """
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# 将三个数据集存为csv文件
if(not(os.path.exists('./data/sms+spam+collection/train.csv'))):
    train_df.to_csv("./data/sms+spam+collection/train.csv", index=None)
if(not(os.path.exists('./data/sms+spam+collection/validation.csv'))):
    validation_df.to_csv("./data/sms+spam+collection/validation.csv", index=None)
if(not(os.path.exists('./data/sms+spam+collection/test.csv'))):
    test_df.to_csv("./data/sms+spam+collection/test.csv", index=None)

# train_df.to_csv("./data/sms+spam+collection/train.csv", index=None)
# validation_df.to_csv("./data/sms+spam+collection/validation.csv", index=None)
# test_df.to_csv("./data/sms+spam+collection/test.csv", index=None)

