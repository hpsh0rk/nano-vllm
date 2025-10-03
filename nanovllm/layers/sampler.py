"""
采样器模块
实现基于温度的随机采样
支持批量采样和温度控制
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器类
    实现基于温度的随机采样算法
    """

    def __init__(self):
        """
        初始化采样器
        """
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        前向传播：执行采样
        
        Args:
            logits: 模型输出的logits张量，形状为[batch_size, vocab_size]
            temperatures: 温度参数张量，形状为[batch_size]
            
        Returns:
            采样得到的token ID张量，形状为[batch_size]
            
        采样过程：
        1. 根据温度调整logits
        2. 计算softmax概率
        3. 使用指数分布进行随机采样
        """
        # 根据温度调整logits
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # 计算softmax概率
        probs = torch.softmax(logits, dim=-1)
        
        # 使用指数分布进行采样（Gumbel-max trick）
        # 生成指数分布随机数，避免除零错误
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        
        return sample_tokens
