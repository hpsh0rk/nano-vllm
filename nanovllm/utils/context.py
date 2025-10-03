"""
上下文管理模块
管理全局推理上下文，包括序列长度、KV缓存映射等信息
"""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    推理上下文类
    存储当前推理过程中的各种参数和状态
    """
    
    # 阶段标识
    is_prefill: bool = False  # 是否为prefill阶段
    
    # 序列长度信息（prefill阶段使用）
    cu_seqlens_q: torch.Tensor | None = None  # query序列的累积长度
    cu_seqlens_k: torch.Tensor | None = None  # key序列的累积长度
    max_seqlen_q: int = 0  # 最大query序列长度
    max_seqlen_k: int = 0  # 最大key序列长度
    
    # KV缓存映射
    slot_mapping: torch.Tensor | None = None  # KV缓存槽位映射
    context_lens: torch.Tensor | None = None  # 上下文长度（decode阶段使用）
    block_tables: torch.Tensor | None = None  # 块表


# 全局上下文实例
_CONTEXT = Context()


def get_context():
    """
    获取当前全局上下文
    
    Returns:
        Context: 当前上下文对象
    """
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    设置全局上下文
    
    Args:
        is_prefill: 是否为prefill阶段
        cu_seqlens_q: query序列的累积长度
        cu_seqlens_k: key序列的累积长度
        max_seqlen_q: 最大query序列长度
        max_seqlen_k: 最大key序列长度
        slot_mapping: KV缓存槽位映射
        context_lens: 上下文长度
        block_tables: 块表
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    """
    重置上下文到默认状态
    """
    global _CONTEXT
    _CONTEXT = Context()
