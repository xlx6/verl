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

from collections import defaultdict
from typing import Any
from transformers import pipeline
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source


    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """(已修改) 使用批量推理计算奖励，并保留日志记录。"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        all_full_text = [] # 存储发送给 API 的文本
        metadata_list = [] # 存储日志和 Bug 修复所需的信息

        # --- 阶段 1: 收集所有数据 ---
        for i in range(len(data)):
            data_item = data[i]
            
            # 解码 Prompt (日志需要)
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            # 解码 Response (日志 & Bug修复 需要)
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # 获取日志所需信息
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # 准备待打分的文本 (这里假设您想评估 prompt + response)
            # (如果您想使用 'original_text'，请确保它在所有数据中都存在)
            full_text = prompt_str + " " + response_str
            all_full_text.append(full_text)
            
            # 存储元数据
            metadata_list.append({
                "valid_response_length": valid_response_length, # <-- 修复 Bug
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source
            })

        # --- 阶段 2: 批量推理 ---
        # 假设 self.compute_score 现在返回一个浮点数列表
        scores = self.compute_score(all_full_text)
        assert len(scores) == len(data), "API 返回的分数数量与批次大小不匹配"

        # --- 阶段 3: 分配分数和打印日志 ---
        for i in range(len(scores)):
            reward = scores[i]
            meta = metadata_list[i] # 获取第 i 项的元数据
            
            # BUG 修复: 使用与分数[i]对应的元数据中的长度
            reward_tensor[i, meta["valid_response_length"] - 1] = reward

            # 填充日志信息
            reward_extra_info["score"].append(reward)

            # 恢复日志打印逻辑
            data_source = meta["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", meta["prompt_str"])
                print("[response]", meta["response_str"])
                print("[ground_truth]", meta["ground_truth"])
                print("[score]", reward) # 简化版日志，因为 score 现在只是一个 float

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor