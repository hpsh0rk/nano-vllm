import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    模型配置类
    用于存储和管理大语言模型的各种配置参数
    """
    
    # 模型路径配置
    model: str  # 模型路径或标识符，指向预训练模型的本地目录
    
    # 批处理相关配置
    max_num_batched_tokens: int = 16384  # 最大批处理token数量，限制单次推理的总token数
    max_num_seqs: int = 512  # 最大并发序列数，限制同时处理的请求数量
    max_model_len: int = 4096  # 模型最大输入长度，限制单个序列的最大token数
    
    # 系统资源配置
    gpu_memory_utilization: float = 0.9  # GPU内存利用率，控制使用的GPU内存比例(0-1)
    tensor_parallel_size: int = 1  # 张量并行大小，用于多GPU并行计算
    
    # 性能优化配置
    enforce_eager: bool = False  # 是否强制使用eager模式，True时禁用图优化
    
    # 模型配置对象
    hf_config: AutoConfig | None = None  # Hugging Face模型配置对象，存储模型架构信息
    
    # 特殊token配置
    eos: int = -1  # 结束符token ID，用于标识序列结束
    
    # KV缓存配置
    kvcache_block_size: int = 256  # KV缓存块大小，每个缓存块包含的token数
    num_kvcache_blocks: int = -1  # KV缓存块数量，-1表示自动计算

    def __post_init__(self):
        """
        初始化后的验证和设置
        确保配置参数的有效性并加载模型配置
        """
        # 验证模型路径是否存在
        assert os.path.isdir(self.model)
        
        # 验证KV缓存块大小必须是256的倍数
        assert self.kvcache_block_size % 256 == 0
        
        # 验证张量并行大小在合理范围内
        assert 1 <= self.tensor_parallel_size <= 8
        
        # 从预训练模型加载Hugging Face配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 确保最大模型长度不超过模型支持的最大位置编码
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 确保批处理token数量不小于单个序列的最大长度
        assert self.max_num_batched_tokens >= self.max_model_len
