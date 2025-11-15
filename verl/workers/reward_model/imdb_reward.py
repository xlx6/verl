from verl.workers.reward_model.base import BasePPORewardModel
from transformers import AutoModel, AutoTokenizer
import torch

class MyLLMRewardModel(BasePPORewardModel):
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
        self.model = AutoModel.from_pretrained(model_config.model_name_or_path)

        self.reward_head = torch.nn.Linear(self.model.config.hidden_size, 1)

    def compute_reward(self, data):
        # 假设 data.input_ids [batch, seq]
        inputs = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
        }
        outputs = self.model(**inputs)
        # 假设last_hidden_state: [batch, seq, hidden]
        last_hidden = outputs.last_hidden_state
        reward_logits = self.reward_head(last_hidden)  # [batch, seq, 1]
        reward_logits = reward_logits.squeeze(-1)      # [batch, seq]
        # 只取EOS（实际可能要根据数据中eri mask获取EOS位置）
        if "eos_mask" in data:
            eos_mask = data["eos_mask"]  # [batch, seq]
            rewards = reward_logits * eos_mask
        else:
            # 示例：假设input_ids等于tokenizer.eos_token_id是EOS
            eos_id = self.tokenizer.eos_token_id
            eos_mask = (data["input_ids"] == eos_id).float()
            rewards = reward_logits * eos_mask
        out = data.copy()
        out["reward"] = rewards
        return out