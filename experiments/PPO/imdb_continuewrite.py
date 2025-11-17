# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the IMDB dataset to parquet format
"""

import argparse
import os
import datasets
import numpy as np
import random

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/imdb_ppo", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "Lucylulu/imdb"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset("imdb")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    def balanced_sample(ds, n_per_label):

        label0 = [x for x in ds if x["label"] == 0]
        label1 = [x for x in ds if x["label"] == 1]
        return random.sample(label0, n_per_label) + random.sample(label1, n_per_label)

    train_samples = balanced_sample(train_dataset, 1000)
    test_samples = balanced_sample(test_dataset, 250)
    train_dataset = datasets.Dataset.from_list(train_samples)
    test_dataset = datasets.Dataset.from_list(test_samples)
    print(train_dataset, test_dataset)
    instruction_following = "Continue the following movie review text."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            raw_text = example.pop("text")
            label = example.pop("label") # 0 for neg, 1 for pos

            clean_text = raw_text.replace("<br />", " ")
            words = clean_text.split()
            
            prefix_len = random.randint(4, 6)
            if len(words) > prefix_len:
                prompt_text = " ".join(words[:prefix_len])
            else:
                prompt_text = clean_text

            content = instruction_following + "\n\nReview start: " + prompt_text

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "ability": "creative_writing",
                "reward_model": {
                    "style": "model", 
                    "ground_truth": label
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_text": prompt_text,
                    "original_label": label,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    # 确保目录存在
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "instruction_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "instruction_test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

    print(f"Data processing complete. Saved to {local_save_dir}")