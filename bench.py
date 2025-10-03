"""
基准测试脚本
测试nano-vllm的吞吐量和性能
"""

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    """
    主函数：执行基准测试
    测试大批量文本生成的吞吐量
    """
    # 设置随机种子确保可重复性
    seed(0)
    
    # 测试参数
    num_seqs = 256  # 序列数量
    max_input_len = 1024  # 最大输入长度
    max_ouput_len = 1024  # 最大输出长度

    # 模型路径
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 初始化LLM，启用CUDA图优化
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 生成随机token序列作为输入
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    
    # 生成随机采样参数
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    
    # vllm兼容性注释：
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 预热：执行一次小规模生成
    llm.generate(["Benchmark: "], SamplingParams())
    
    # 正式测试
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    
    # 计算吞吐量
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    
    # 输出结果
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
