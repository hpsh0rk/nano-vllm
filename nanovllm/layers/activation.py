"""
激活函数模块
实现各种激活函数和组合操作
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SiLU激活函数与乘法组合
    用于SwiGLU激活函数的实现
    """

    def __init__(self):
        """初始化"""
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，最后一维会被分成两部分
            
        Returns:
            SiLU激活后的结果与另一半相乘的结果
        """
        # 将最后一维分成两部分
        x, y = x.chunk(2, -1)
        
        # SiLU激活后相乘
        return F.silu(x) * y
