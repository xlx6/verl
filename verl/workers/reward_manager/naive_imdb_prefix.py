# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (License header as in original file)

from collections import defaultdict
from typing import Any
from transformers import pipeline
import torch
import numpy as np # <-- 1. 导入 numpy

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """(已修改) The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.
        (Init logic as in original file)
        """
        self.tokenizer = tokenizer  
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key


    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """(已修改) 使用批量推理计算奖励，并按前缀情感分类记录日志。"""

        # If there is rm score, we directly return rm score. (Logic as in original file)
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
        
        all_full_text = [] 
        metadata_list = [] 

        # --- 阶段 1: 收集所有数据 (修复 Bug) ---
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # (新功能) 获取在数据处理时存入的 'prefix_sentiment_label'
            # (来自 imdb_continuewrite.py)
            prefix_label = extra_info.get("prefix_sentiment_label", -1) # 0, 1, 2, or -1 (Unknown)

            # 拼接 full_text
            full_text = prompt_str + " " + response_str
            all_full_text.append(full_text)
            
            # 存储所有需要在第二阶段使用的信息
            metadata_list.append({
                "valid_response_length": valid_response_length, 
                "prefix_label": prefix_label,                   
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source
            })

        # --- 阶段 2: 批量推理 ---
        scores = self.compute_score(all_full_text) # 假设返回 List[float]
        assert len(scores) == len(data), "API 返回的分数数量与批次大小不匹配"

        # --- 阶段 3: 分配分数和分类日志 ---
        label_to_name_map = {0: "Negative", 1: "Neutral", 2: "Positive", -1: "Unknown"}

        for i in range(len(scores)):
            score = scores[i]
            meta = metadata_list[i] 
            
            # BUG 修复: 使用与 scores[i] 对应的元数据中的长度
            reward_tensor[i, meta["valid_response_length"] - 1] = score

            # --- (新功能) Assert 修复：填充所有列表以保持长度一致 ---
            
            # 1. 填充 "score" (verl 会自动记录为 .../score/mean)
            reward_extra_info["score"].append(score) 
            
            # 2. 获取当前项的标签
            current_prefix_label_int = meta["prefix_label"]
            current_prefix_label_name = label_to_name_map.get(current_prefix_label_int, "Unknown")
            
            # 3. 遍历 *所有* 可能的标签，要么填 score, 要么填 nan
            for label_name in label_to_name_map.values():
                key = f"score_prefix_{label_name}"
                
                if label_name == current_prefix_label_name:
                    # 这是正确的类别，添加分数
                    reward_extra_info[key].append(score)
                else:
                    # 这不是正确的类别，添加 nan 来占位
                    reward_extra_info[key].append(np.nan)
            
            # --- 恢复日志打印逻辑 ---
            data_source = meta["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", meta["prompt_str"])
                print("[response]", meta["response_str"])
                print("[ground_truth]", meta["ground_truth"])
                print(f"[score] (prefix: {current_prefix_label_name})", score) 
        print('='*100)
        for label_name in label_to_name_map.values():
            key = f"score_prefix_{label_name}"
            print(reward_extra_info[key])
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor