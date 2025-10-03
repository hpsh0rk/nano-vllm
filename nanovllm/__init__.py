"""
nano-vllm: 轻量级大语言模型推理引擎

提供高效的文本生成服务，支持：
- 多GPU张量并行
- KV缓存优化
- Flash Attention
- CUDA图优化
- 动态批处理

使用示例:
    >>> from nanovllm import LLM, SamplingParams
    >>> llm = LLM("path/to/model")
    >>> result = llm.generate(["你好，世界！"])
"""

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

# 导出主要API
__all__ = ["LLM", "SamplingParams"]
