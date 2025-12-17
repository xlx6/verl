from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl import DataProto
from verl.utils.reward_score.gsm8k import extract_solution
from typing import Any
from collections import defaultdict, Counter
import torch
import re

@register("group_voting")
class GroupVotingRewardManager(AbstractRewardManager):
    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source", 
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        self.n_rollouts = kwargs.get('n_rollouts', 8)
        
        print(f">>> [GroupVotingRM] Initialized. Group Size (n_rollouts): {self.n_rollouts}")
        
        self.extract_answer_fn = extract_solution



    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # 如果数据中已经包含 rm_scores (比如来自外部 RM)，直接返回
        if "rm_scores" in data.batch.keys():
             if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
             else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_cnt = 0

        # 检查 batch 大小是否符合 n_rollouts 的设定
        batch_size = len(data)
        if batch_size % self.n_rollouts != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by n_rollouts ({self.n_rollouts}) for group voting.")

        # 按组遍历数据
        for i in range(0, batch_size, self.n_rollouts):
            # 获取当前组的索引范围
            indices = range(i, i + self.n_rollouts)
            
            group_responses = []
            group_extracted_answers = []
            
            # --- 第一步：解码并提取当前组的所有答案 ---
            for idx in indices:
                data_item = data[idx]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                # Decode
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                group_responses.append((idx, response_str, valid_response_length))
                
                # Extract Answer
                ans = self.extract_answer_fn(response_str)
                group_extracted_answers.append(ans)

            # --- 第二步：计算投票统计 (Soft Labels) ---
            # 过滤掉 None (格式错误的回复)，只统计有效答案
            valid_answers = [ans for ans in group_extracted_answers if ans is not None]
            
            if len(valid_answers) > 0:
                answer_counts = Counter(valid_answers)
                # 计算软标签：该答案出现的次数 / 组内有效答案总数 (或者总数 n_rollouts)
                # 策略A (Soft): counts / n_rollouts
                # 策略B (Hard Majority): 如果是众数得1，否则得0 (这里我们用 Soft)
            else:
                answer_counts = {}

            # --- 第三步：分配 Reward ---
            for j, (global_idx, resp_str, v_len) in enumerate(group_responses):
                extracted_ans = group_extracted_answers[j]
                
                # 计算分数
                if extracted_ans is None:
                    # 格式错误，给予惩罚 (比如 0.0 或 -1.0)
                    score = 0.0 
                else:
                    # Soft Label: 频率越高，分数越高
                    # 例如：5个sample，3个回答"A"，2个回答"B"。
                    # "A"的得分为 3/5=0.6, "B"的得分为 2/5=0.4
                    score = answer_counts[extracted_ans] / self.n_rollouts
                
                # 将分数填入 Tensor 的最后一个 token 位置
                reward_tensor[global_idx, v_len - 1] = score
                
                # 记录额外信息用于 Logging
                reward_extra_info["extracted_answer"].append(str(extracted_ans))
                reward_extra_info["group_score"].append(score)

                # --- 打印调试信息 ---
                if already_print_cnt < self.num_examine:
                    print(f"\n[Group Voting Debug] Group Index: {i // self.n_rollouts}, Item: {j}")
                    print(f"[Response]: {resp_str}")
                    print(f"[Extracted]: {extracted_ans}")
                    print(f"[Score (Soft Label)]: {score}")
                    already_print_cnt += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor



