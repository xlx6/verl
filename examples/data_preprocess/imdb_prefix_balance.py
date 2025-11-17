# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (License header as in original file)
"""
Preprocess the IMDB dataset to parquet format,
sampling based on the sentiment of the generated prompt prefix.
"""

import argparse
import os
import datasets
import numpy as np
import random
import torch
from transformers import pipeline
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs

def create_prefix_balanced_dataset(dataset, split_name, sentiment_pipe, label_map, n_per_label):
    """
    高效地迭代数据集，实时生成和分类前缀，
    直到三个情感类别的桶都满了。
    """
    
    # 三个桶，用于存放不同情感前缀的数据
    samples_buckets = {0: [], 1: [], 2: []} # 0: Neg, 1: Neu, 2: Pos
    
    total_needed = n_per_label * 3
    
    print(f"Starting prefix sentiment sampling for '{split_name}' set...")
    print(f"Target: {n_per_label} samples per label (Neg, Neu, Pos).")

    # 随机打乱数据集以便每次运行都能获得不同的前缀
    shuffled_dataset = dataset.shuffle(seed=42)
    
    idx = 0
    # 使用 tqdm 进度条
    pbar = tqdm(total=total_needed)

    for example in shuffled_dataset:
        # 检查所有桶是否都满了
        if all(len(samples_buckets[label]) >= n_per_label for label in label_map.values()):
            print("All buckets are full. Stopping sampling.")
            break
        
        idx += 1
        raw_text = example["text"]
        original_label = example["label"]

        # 1. 生成前缀 (与原脚本逻辑一致)
        clean_text = raw_text.replace("<br />", " ")
        words = clean_text.split()
        prefix_len = random.randint(16, 20)
        if len(words) > prefix_len:
            prompt_text = " ".join(words[:prefix_len])
        else:
            continue # 文本太短，跳过

        # 2. 分类前缀
        try:
            result = sentiment_pipe(prompt_text)[0][0]
            prefix_label_str = result['label']
            prefix_sentiment_int = label_map.get(prefix_label_str, 1) # 默认为 1 (Neutral)
        except Exception as e:
            print(f'classification error: {e}')
            continue # 分类失败，跳过

        # 3. 检查桶是否已满
        if len(samples_buckets[prefix_sentiment_int]) < n_per_label:
            # 桶未满，添加此样本
            
            # (与原脚本逻辑一致)
            instruction_following = "Continue the following movie review text."
            content = instruction_following + "\n\nReview start: " + prompt_text

            data = {
                "data_source": "stanfordnlp/imdb",
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "ability": "creative_writing",
                "reward_model": {
                    "style": "model", 
                    "ground_truth": original_label # 原始IMDB标签
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "original_text": prompt_text, # 这是前缀
                    "original_label": original_label,
                    "prefix_sentiment_label": prefix_sentiment_int # 0, 1, or 2
                },
            }
            
            samples_buckets[prefix_sentiment_int].append(data)
            pbar.update(1)

    pbar.close()
    
    # 合并所有桶
    final_samples = []
    for bucket in samples_buckets.values():
        final_samples.extend(bucket)
        
    return datasets.Dataset.from_list(final_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/imdb_ppo_prefix_balanced_long", help="The save directory for the preprocessed dataset."
    )
    # 自定义采样数量
    parser.add_argument("--n_train_per_label", type=int, default=1000, help="Number of train samples per prefix label (Neg, Neu, Pos)")
    parser.add_argument("--n_test_per_label", type=int, default=100, help="Number of test samples per prefix label (Neg, Neu, Pos)")


    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # --- 1. 加载情感模型 ---
    print("Loading sentiment analysis model...")
    # 检查是否有 GPU
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="/home/xinglixian/models/twitter-roberta-base-sentiment-latest", 
        device=device,
        top_k=1 # 只需要最高分的标签
    )
    # (模型输出 'Negative', 'Neutral', 'Positive')
    label_map = {"negative": 0, "neutral": 1, "positive": 2}


    # --- 2. 加载 IMDB 数据集 ---
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset("imdb")

    train_dataset_full = dataset["train"]
    test_dataset_full = dataset["test"]

    # --- 3. 执行采样和处理 ---
    train_dataset_new = create_prefix_balanced_dataset(
        train_dataset_full, 
        "train", 
        sentiment_pipe, 
        label_map, 
        args.n_train_per_label
    )
    
    test_dataset_new = create_prefix_balanced_dataset(
        test_dataset_full, 
        "test", 
        sentiment_pipe, 
        label_map, 
        args.n_test_per_label
    )

    print("Sampling complete!")
    print(f"New Train Dataset: {train_dataset_new}")
    print(f"New Test Dataset: {test_dataset_new}")

    # --- 4. 保存 (与原脚本逻辑一致) ---
    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    # 确保目录存在
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset_new.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset_new.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

    print(f"Data processing complete. Saved to {local_save_dir}")