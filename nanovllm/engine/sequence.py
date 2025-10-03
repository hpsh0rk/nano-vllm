"""
序列管理模块
用于管理和跟踪单个文本序列的状态和属性
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举类
    表示序列在推理过程中的不同状态
    """
    WAITING = auto()   # 等待状态：序列已创建但尚未开始处理
    RUNNING = auto()   # 运行状态：序列正在处理中
    FINISHED = auto()  # 完成状态：序列处理已完成


class Sequence:
    """
    序列类
    表示一个文本序列，包含prompt和生成的completion
    管理序列的token、状态、缓存等属性
    """
    
    # 类变量
    block_size = 256  # 块大小，用于KV缓存分块管理
    counter = count()  # 计数器，为每个序列生成唯一ID

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列
        
        Args:
            token_ids: 初始token ID列表，通常为prompt的token序列
            sampling_params: 采样参数配置，控制生成行为
        """
        # 序列标识
        self.seq_id = next(Sequence.counter)  # 序列唯一标识符
        self.status = SequenceStatus.WAITING  # 初始状态为等待
        
        # Token相关属性
        self.token_ids = copy(token_ids)  # 完整的token序列（prompt + completion）
        self.last_token = token_ids[-1]   # 最后一个token，用于快速访问
        self.num_tokens = len(self.token_ids)  # 总token数量
        self.num_prompt_tokens = len(token_ids)  # prompt的token数量
        self.num_cached_tokens = 0  # 已缓存的token数量，用于KV缓存管理
        
        # KV缓存管理
        self.block_table = []  # 块表，记录序列使用的KV缓存块
        
        # 采样参数
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens  # 最大生成token数
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略结束符

    def __len__(self):
        """返回序列的总token数量"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引访问，返回指定位置的token ID"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的completion token数量"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """prompt部分的token序列"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """completion部分的token序列"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """已缓存的KV缓存块数量"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """序列所需的KV缓存块总数"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中的token数量"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第i个块的token序列
        
        Args:
            i: 块索引
            
        Returns:
            第i个块的token序列
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        向序列追加一个token
        
        Args:
            token_id: 要追加的token ID
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化方法，用于pickle
        优化内存使用，对于已开始生成的序列只保存最后一个token
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        反序列化方法，用于pickle
        恢复序列状态
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
