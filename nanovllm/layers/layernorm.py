"""
层归一化模块
实现RMSNorm（均方根归一化）
支持带残差连接的归一化
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMSNorm（均方根归一化）
    一种高效的归一化方法，相比LayerNorm计算更简单
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        初始化RMSNorm
        
        Args:
            hidden_size: 隐藏层维度
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放参数

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        RMS归一化前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量
        """
        orig_dtype = x.dtype
        
        # 转换为float32计算，避免精度问题
        x = x.float()
        
        # 计算均方根
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))  # 归一化
        
        # 应用缩放参数并恢复原始数据类型
        x = x.to(orig_dtype).mul_(self.weight)
        
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        带残差连接的RMS归一化
        
        Args:
            x: 输入张量
            residual: 残差张量
            
        Returns:
            tuple: (归一化后的张量, 更新后的残差)
        """
        orig_dtype = x.dtype
        
        # 添加残差连接
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)  # 保存残差用于后续层
        
        # 计算均方根并归一化
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        
        # 应用缩放参数并恢复原始数据类型
        x = x.to(orig_dtype).mul_(self.weight)
        
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            residual: 可选的残差张量
            
        Returns:
            归一化结果，带残差时返回tuple
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
