"""
线性层模块
实现各种线性层，支持张量并行
包括复制线性层、列并行线性层、行并行线性层等
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """
    整数除法，确保能整除
    
    Args:
        numerator: 被除数
        denominator: 除数
        
    Returns:
        除法结果
    """
    assert numerator % denominator == 0, f"{numerator}不能被{denominator}整除"
    return numerator // denominator


class LinearBase(nn.Module):
    """
    线性层基类
    提供张量并行的基础功能
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """
        初始化线性层基类
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            bias: 是否使用偏置
            tp_dim: 张量并行维度（0表示按列切分，1表示按行切分）
        """
        super().__init__()
        self.tp_dim = tp_dim  # 张量并行维度
        self.tp_rank = dist.get_rank()  # 当前进程的rank
        self.tp_size = dist.get_world_size()  # 总进程数
        
        # 权重参数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        
        # 偏置参数
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，子类需要实现"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层
    所有进程使用相同的权重，不进行张量并行
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        初始化复制线性层
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            bias: 是否使用偏置
        """
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载：直接复制权重"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：标准线性变换"""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层
    按列切分权重矩阵，每个进程处理一部分输出维度
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        初始化列并行线性层
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            bias: 是否使用偏置
        """
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载：按列切分权重"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：标准线性变换"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层
    支持多个输出维度的合并，用于gate_proj和up_proj的合并
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """
        初始化合并列并行线性层
        
        Args:
            input_size: 输入特征维度
            output_sizes: 输出维度列表
            bias: 是否使用偏置
        """
        self.output_sizes = output_sizes  # 存储各个输出维度
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """权重加载：按输出维度切分权重"""
        param_data = param.data
        
        # 计算当前shard的偏移和大小
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        
        # 切分参数和权重
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV并行线性层
    专门用于处理Q、K、V的并行计算
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """
        初始化QKV并行线性层
        
        Args:
            hidden_size: 隐藏层维度
            head_size: 每个注意力头的维度
            total_num_heads: 总注意力头数
            total_num_kv_heads: 总key/value头数，默认为total_num_heads
            bias: 是否使用偏置
        """
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        # 计算当前进程的head数
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)  # 当前进程的查询头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 当前进程的KV头数
        
        # 计算输出维度：Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """权重加载：按Q/K/V切分权重"""
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"], f"无效的shard_id: {loaded_shard_id}"
        
        if loaded_shard_id == "q":
            # 查询权重
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # key权重
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            # value权重
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # 切分参数和权重
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层
    按行切分权重矩阵，每个进程处理一部分输入维度
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        初始化行并行线性层
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            bias: 是否使用偏置（只在rank=0上有效）
        """
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载：按行切分权重"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        执行线性变换，并在多GPU情况下进行all-reduce
        """
        # 只在rank=0上使用偏置，避免重复计算
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        
        # 多GPU情况下进行all-reduce
        if self.tp_size > 1:
            dist.all_reduce(y)  # 聚合所有进程的输出
        
        return y
