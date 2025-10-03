from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    采样参数配置类
    用于控制文本生成的采样策略和限制条件
    """
    
    # 采样策略参数
    temperature: float = 1.0  # 温度参数，控制生成的随机性(>0)
                            # - 接近0: 更确定性，倾向于选择最高概率的词
                            # - 等于1: 标准随机采样
                            # - 大于1: 增加随机性，生成更多样化的文本
    
    # 生成限制参数
    max_tokens: int = 64  # 最大生成token数量，限制生成文本的长度
    
    # 结束控制参数
    ignore_eos: bool = False  # 是否忽略结束符，True时即使遇到EOS token也继续生成

    def __post_init__(self):
        """
        初始化后的验证
        确保采样参数的有效性
        """
        # 温度必须大于最小值，避免除零错误
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
