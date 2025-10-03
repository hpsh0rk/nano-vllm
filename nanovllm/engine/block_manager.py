"""
KV缓存块管理模块
实现高效的KV缓存分配、回收和复用机制
支持基于哈希的KV缓存共享，减少内存占用
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV缓存块类
    表示一个KV缓存块，包含块ID、引用计数、哈希值和token序列
    """

    def __init__(self, block_id):
        """
        初始化KV缓存块
        
        Args:
            block_id: 块唯一标识符
        """
        self.block_id = block_id     # 块ID，唯一标识
        self.ref_count = 0           # 引用计数，跟踪有多少序列使用此块
        self.hash = -1               # 哈希值，用于KV缓存共享
        self.token_ids = []          # 该块包含的token序列

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的哈希值和token序列
        
        Args:
            hash: 新的哈希值
            token_ids: 新的token序列
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态
        将块恢复到初始状态，准备重新使用
        """
        self.ref_count = 1  # 新分配的块引用计数设为1
        self.hash = -1      # 重置哈希值
        self.token_ids = []  # 清空token序列


class BlockManager:
    """
    KV缓存块管理器
    负责KV缓存块的分配、回收、复用和共享
    使用基于哈希的缓存共享机制减少内存使用
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器
        
        Args:
            num_blocks: 总KV缓存块数量
            block_size: 每个块的token容量
        """
        self.block_size = block_size  # 块大小，每个块可存储的token数量
        
        # 块管理
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有KV缓存块
        self.hash_to_block_id: dict[int, int] = dict()  # 哈希到块ID的映射，用于缓存共享
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids: set[int] = set()  # 已使用块集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token序列的哈希值
        使用xxhash64算法，支持前缀哈希链
        
        Args:
            token_ids: token序列
            prefix: 前缀哈希值，用于构建哈希链
            
        Returns:
            64位哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配指定的KV缓存块
        
        Args:
            block_id: 要分配的块ID
            
        Returns:
            分配的块对象
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保块未被引用
        block.reset()  # 重置块状态
        self.free_block_ids.remove(block_id)  # 从空闲队列移除
        self.used_block_ids.add(block_id)     # 添加到已使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        回收指定的KV缓存块
        
        Args:
            block_id: 要回收的块ID
        """
        assert self.blocks[block_id].ref_count == 0  # 确保块未被引用
        self.used_block_ids.remove(block_id)  # 从已使用集合移除
        self.free_block_ids.append(block_id)  # 添加到空闲队列

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块分配给序列
        
        Args:
            seq: 要分配的序列
            
        Returns:
            True表示可以分配，False表示空间不足
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配KV缓存块
        支持基于哈希的缓存共享，复用相同内容的块
        
        Args:
            seq: 要分配缓存的序列
        """
        assert not seq.block_table  # 确保序列尚未分配缓存
        
        h = -1  # 初始哈希值
        cache_miss = False  # 缓存未命中标志
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取第i个块的token序列
            
            # 计算当前块的哈希值（只有完整块才计算哈希）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 查找匹配的缓存块
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中
            
            if cache_miss:
                # 分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 复用缓存块
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 块已在使用中，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在空闲队列中，重新分配
                    block = self._allocate_block(block_id)
            
            # 更新块的哈希和token序列
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)  # 记录到序列的块表

    def deallocate(self, seq: Sequence):
        """
        回收序列使用的KV缓存块
        
        Args:
            seq: 要回收缓存的序列
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数
            if block.ref_count == 0:
                # 引用计数为0，回收块
                self._deallocate_block(block_id)
        
        # 重置序列的缓存状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以向序列追加新的KV缓存块
        
        Args:
            seq: 要检查的序列
            
        Returns:
            True表示可以追加，False表示空间不足
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        根据序列长度可能需要追加新的KV缓存块
        
        Args:
            seq: 要处理的序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # 需要新块：序列长度刚好跨越块边界
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 完成当前块：序列长度刚好填满一个块
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 继续填充当前块
            assert last_block.hash == -1
