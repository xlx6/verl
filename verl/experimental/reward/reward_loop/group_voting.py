from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto

@register("group_voting")
class GroupVotingRewardLoopManager(RewardLoopManagerBase):
    """
    异步 Loop Manager 的包装器。
    它实际上并不执行复杂的异步逻辑，只是为了适配接口，
    内部直接调用同步的 GroupVotingRewardManager 进行计算。
    """
    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        super().__init__(config, tokenizer)
        
        # 1. 实例化核心逻辑：我们之前写的 GroupVotingRewardManager
        # 我们从 config 中提取参数，或者设置默认值
        from verl.workers.reward_manager import GroupVotingRewardManager  # 假设你把之前的类放在了这里
        
        # 尝试从 config.reward_model 或 config 中获取参数
        # 注意：这里的 config 结构取决于 verl 传入的是整个 config 还是只是 rm config
        # 通常建议通过 safe get 获取
        num_examine = config.get("num_examine", 1)
        
        # 这里的 n_rollouts 需要与 actor 采样数一致
        # 我们尝试从 config 中深层查找，找不到就用默认值 8
        n_rollouts = config.get("n_rollouts", 8)
        
        self.voting_manager = GroupVotingRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            n_rollouts=n_rollouts
            # 如果之前的 manager 还有其他参数，在这里透传
        )
        
        print(f"[GroupVotingLoop] Initialized with n_rollouts={n_rollouts}")

    async def run_single(self, data: DataProto):
        """
        执行单次（或一个批次）的奖励计算。
        注意：为了进行组内投票，这里的 'data' 必须包含一组完整的 response。
        """
        # 2. 直接委托给同步的 Manager 处理
        # 因为我们的计算主要在 CPU (正则匹配+统计)，不需要 await I/O
        # async def 会自动将结果包装成 Coroutine
        
        # 这一步会执行：解码 -> 提取答案 -> 计算投票 -> 填充 Tensor
        reward_tensor = self.voting_manager(data, return_dict=False)
        
        return reward_tensor