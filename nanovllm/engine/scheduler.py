"""
调度器模块
负责管理序列的执行顺序，实现prefill和decode阶段的调度
支持序列的优先级管理、资源分配和抢占机制
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    序列调度器
    管理序列的执行顺序，协调prefill和decode阶段的切换
    处理资源分配、抢占和完成状态的转换
    """

    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 模型配置对象
        """
        # 批处理限制
        self.max_num_seqs = config.max_num_seqs              # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 最大批处理token数
        self.eos = config.eos                                # 结束符token ID
        
        # KV缓存管理
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 序列队列
        self.waiting: deque[Sequence] = deque()   # 等待处理的序列队列
        self.running: deque[Sequence] = deque()   # 正在运行的序列队列

    def is_finished(self):
        """
        检查所有序列是否已完成处理
        
        Returns:
            True表示所有序列已完成，False表示还有序列在处理
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新序列到等待队列
        
        Args:
            seq: 要添加的序列
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度序列执行
        
        Returns:
            tuple: (调度的序列列表, 是否为prefill阶段)
            
        调度策略：
        1. 优先处理prefill阶段（新序列的首次处理）
        2. 然后处理decode阶段（序列的后续生成）
        3. 处理资源不足时的抢占
        """
        # prefill阶段：处理新序列的首次计算
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 检查资源限制：token数量限制和KV缓存空间
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            
            num_seqs += 1
            self.block_manager.allocate(seq)  # 分配KV缓存
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 实际需要计算的token数
            seq.status = SequenceStatus.RUNNING  # 状态转为运行中
            
            # 从等待队列移到运行队列
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # 如果有prefill序列，直接返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode阶段：处理运行中序列的后续生成
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 处理KV缓存空间不足的情况
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢占其他序列释放空间
                    self.preempt(self.running.pop())
                else:
                    # 无法抢占，只能抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 空间充足，可以处理
                num_seqs += 1
                self.block_manager.may_append(seq)  # 可能需要追加新块
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs  # 确保至少有一个序列被调度
        
        # 将调度的序列重新放回运行队列前端
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占序列，将其从运行状态转为等待状态
        释放其占用的KV缓存资源
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING  # 状态转为等待
        self.block_manager.deallocate(seq)   # 释放KV缓存
        self.waiting.appendleft(seq)         # 放回等待队列前端

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理生成的token
        更新序列状态，检查是否完成
        
        Args:
            seqs: 处理的序列列表
            token_ids: 生成的token ID列表
            
        Returns:
            完成状态列表，True表示序列已完成
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 追加新生成的token
            
            # 检查序列是否完成：遇到结束符或达到最大长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED  # 状态转为完成
                self.block_manager.deallocate(seq)   # 释放KV缓存
                self.running.remove(seq)             # 从运行队列移除
