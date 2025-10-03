"""
大语言模型主类
提供简洁的API接口，继承自LLMEngine，用于文本生成任务
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    大语言模型封装类
    
    这是一个简单的包装类，继承自LLMEngine，提供了更简洁的API接口。
    用户可以通过此类来初始化和使用大语言模型进行文本生成。
    
    使用示例:
        >>> from nanovllm import LLM, Config
        >>> config = Config(model="path/to/model")
        >>> llm = LLM(config)
        >>> result = llm.generate("你好，世界！")
    """
    pass
