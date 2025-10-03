"""
注意力层模块
实现高效的注意力计算，支持KV缓存和Flash Attention优化
包括KV缓存存储和注意力计算的核心逻辑
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton内核：将key和value存储到KV缓存
    
    Args:
        key_ptr: key张量的指针
        key_stride: key张量的stride
        value_ptr: value张量的指针
        value_stride: value张量的stride
        k_cache_ptr: key缓存张量的指针
        v_cache_ptr: value缓存张量的指针
        slot_mapping_ptr: 槽位映射张量的指针
        D: 总维度大小(num_heads * head_dim)
    """
    idx = tl.program_id(0)  # 获取当前程序ID
    slot = tl.load(slot_mapping_ptr + idx)  # 加载对应的槽位ID
    
    if slot == -1:  # 无效槽位，直接返回
        return
    
    # 计算内存偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 加载key和value数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算缓存偏移并存储
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    将key和value存储到KV缓存的封装函数
    
    Args:
        key: key张量，形状为[N, num_heads, head_dim]
        value: value张量，形状为[N, num_heads, head_dim]
        k_cache: key缓存张量
        v_cache: value缓存张量
        slot_mapping: 槽位映射张量，形状为[N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 总维度大小
    
    # 验证张量格式
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 最后一维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 正确的stride
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # 缓存stride匹配
    assert slot_mapping.numel() == N  # 槽位映射数量匹配
    
    # 调用Triton内核执行存储
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    注意力层
    实现高效的注意力计算，支持KV缓存和Flash Attention
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        """
        初始化注意力层
        
        Args:
            num_heads: 注意力头数
            head_dim: 每个注意力头的维度
            scale: 缩放因子
            num_kv_heads: key/value头数
        """
        super().__init__()
        self.num_heads = num_heads  # 查询头数
        self.head_dim = head_dim    # 每个头的维度
        self.scale = scale          # 注意力缩放因子
        self.num_kv_heads = num_kv_heads  # key/value头数
        
        # KV缓存张量，初始化为空，后续由模型运行器设置
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        前向传播
        
        Args:
            q: 查询张量，形状为[总token数, num_heads, head_dim]
            k: key张量，形状为[总token数, num_kv_heads, head_dim]
            v: value张量，形状为[总token数, num_kv_heads, head_dim]
            
        Returns:
            注意力输出张量
        """
        context = get_context()  # 获取当前推理上下文
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 存储KV到缓存（如果缓存已初始化）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill阶段：处理完整的序列
            if context.block_tables is not None:  # prefix cache启用
                # 使用缓存中的KV
                k, v = k_cache, v_cache
            
            # 使用Flash Attention处理变长序列
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            # Decode阶段：处理单个token
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),  # 添加序列维度
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        
        return o
