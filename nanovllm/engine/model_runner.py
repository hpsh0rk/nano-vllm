"""
模型运行器模块
负责模型的初始化、推理执行和资源管理
支持多GPU并行、CUDA图优化、KV缓存管理等功能
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.sharedmemory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器类
    管理模型推理的整个生命周期，包括初始化、执行、优化和资源清理
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器
        
        Args:
            config: 模型配置对象
            rank: 当前进程的GPU rank
            event: 用于多进程通信的事件对象
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # KV缓存块大小
        self.enforce_eager = config.enforce_eager    # 是否强制使用eager模式
        self.world_size = config.tensor_parallel_size  # 张量并行大小
        self.rank = rank  # 当前进程rank
        self.event = event  # 多进程通信事件

        # 初始化分布式训练
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 设置默认数据类型和设备
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 初始化模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)  # 加载模型权重
        self.sampler = Sampler()  # 初始化采样器
        
        # 模型预热和KV缓存分配
        self.warmup_model()
        self.allocate_kv_cache()
        
        # CUDA图优化
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程通信设置
        if self.world_size > 1:
            if rank == 0:
                # 主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 其他进程等待并连接共享内存
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 进入监听循环

    def exit(self):
        """
        清理资源并退出
        关闭共享内存、销毁进程组、清理CUDA图等
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 主进程清理共享内存
        
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 清理CUDA图
        
        torch.cuda.synchronize()
        dist.destroy_process_group()  # 销毁进程组

    def loop(self):
        """
        多进程工作循环
        监听共享内存中的命令并执行
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取命令
        
        Returns:
            tuple: (方法名, 参数列表)
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待事件信号
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 清除事件信号
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入命令
        
        Args:
            method_name: 要调用的方法名
            *args: 方法参数
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()  # 通知所有工作进程

    def call(self, method_name, *args):
        """
        调用方法，支持多进程通信
        
        Args:
            method_name: 方法名
            *args: 方法参数
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热
        使用最大配置的序列进行预热，确保后续推理稳定
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        
        # 计算最大可能的序列数量
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        
        self.run(seqs, True)  # 执行预热
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配KV缓存
        根据GPU内存使用情况计算KV缓存块数量并分配
        """
        config = self.config
        hf_config = config.hf_config
        
        # 获取GPU内存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算KV缓存参数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 每个GPU的KV头数
        
        # 计算每个KV缓存块的字节大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        
        # 计算可分配的KV缓存块数量
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "KV缓存块数量必须大于0"
        
        # 分配KV缓存张量
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        
        # 将KV缓存绑定到模型层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # Key缓存
                module.v_cache = self.kv_cache[1, layer_id]  # Value缓存
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表张量
        将序列的块表转换为PyTorch张量，用于模型推理
        
        Args:
            seqs: 序列列表
            
        Returns:
            块表张量
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备prefill阶段的输入数据
        
        Args:
            seqs: 要处理的序列列表
            
        Returns:
            tuple: (input_ids, positions)
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # query序列长度累积
        cu_seqlens_k = [0]  # key序列长度累积
        max_seqlen_q = 0    # 最大query长度
        max_seqlen_k = 0    # 最大key长度
        slot_mapping = []   # KV缓存槽位映射
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            
            # 收集需要计算的token（跳过已缓存的部分）
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            # 计算序列长度信息
            seqlen_q = seqlen - seq.num_cached_tokens  # 需要计算的token数
            seqlen_k = seqlen  # 总的key长度
            
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # 跳过warmup阶段
            if not seq.block_table:
                continue
                
            # 构建KV缓存槽位映射
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                
                if i != seq.num_blocks - 1:
                    end = start + self.block_size  # 完整块
                else:
                    end = start + seq.last_block_num_tokens  # 最后一个部分块
                    
                slot_mapping.extend(list(range(start, end)))
        
        # 如果有KV缓存，准备块表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为PyTorch张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置全局上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备decode阶段的输入数据
        
        Args:
            seqs: 要处理的序列列表
            
        Returns:
            tuple: (input_ids, positions)
        """
        input_ids = []
        positions = []
        slot_mapping = []    # KV缓存槽位映射
        context_lens = []    # 上下文长度
        
        for seq in seqs:
            # 收集最后一个token作为输入
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)  # 位置为序列最后一个位置
            context_lens.append(len(seq))   # 上下文长度为完整序列长度
            
            # 计算KV缓存槽位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # 转换为PyTorch张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置全局上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数
        
        Args:
            seqs: 序列列表
            
        Returns:
            温度参数张量
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        运行模型推理
        
        Args:
            input_ids: 输入token ID张量
            positions: 位置编码张量
            is_prefill: 是否为prefill阶段
            
        Returns:
            模型输出的logits
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 使用eager模式：prefill阶段或大batch
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用CUDA图优化：decode阶段且小batch
            bs = input_ids.size(0)
            context = get_context()
            
            # 选择合适大小的CUDA图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 更新CUDA图变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放CUDA图
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行模型推理
        
        Args:
            seqs: 要处理的序列列表
            is_prefill: 是否为prefill阶段
            
        Returns:
            生成的token ID列表
        """
        # 准备输入数据
        # positions: 位置编码张量
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # 运行模型
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # 采样生成token
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # 重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获CUDA图进行优化
        为不同batch大小预编译CUDA图，提高decode阶段性能
        """
        config = self.config
        hf_config = config.hf_config
        
        # 配置参数
        max_bs = min(self.config.max_num_seqs, 512)  # 最大batch大小
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 创建CUDA图输入张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义batch大小序列
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 为每个batch大小捕获CUDA图
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 预热
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 捕获CUDA图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 管理CUDA图池
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 存储CUDA图变量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
