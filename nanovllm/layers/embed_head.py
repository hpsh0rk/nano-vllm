"""
嵌入层和输出头模块
实现词嵌入层和语言模型输出头
支持词汇表并行
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词汇表并行嵌入层
    将词汇表按维度切分，支持大词汇表的并行处理
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """
        初始化词汇表并行嵌入层
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入维度
        """
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 验证词汇表大小能被进程数整除
        assert num_embeddings % self.tp_size == 0, "词汇表大小必须能被进程数整除"
        
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        
        # 计算当前进程负责的词汇范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 嵌入权重
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载：按词汇范围切分嵌入权重"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 输入token ID张量
            
        Returns:
            嵌入向量张量
        """
        if self.tp_size > 1:
            # 创建掩码，标记当前进程负责的token
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 调整token ID为本地索引
            x = mask * (x - self.vocab_start_idx)
        
        # 执行嵌入查找
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 应用掩码并聚合结果
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型输出头
    将隐藏状态映射到词汇表logits
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """
        初始化并行语言模型输出头
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 隐藏层维度
            bias: 是否使用偏置（目前不支持）
        """
        assert not bias, "ParallelLMHead暂不支持偏置"
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 隐藏状态张量
            
        Returns:
            logits张量，仅在rank=0上返回有效值
        """
        context = get_context()
        
        # 在prefill阶段，只取每个序列的最后一个token
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # 线性变换生成logits
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # 收集所有进程的logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        
        return logits
