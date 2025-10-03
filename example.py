"""
使用示例
演示如何使用nano-vllm进行文本生成
"""

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """
    主函数：演示nano-vllm的基本用法
    """
    # 模型路径
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # 初始化LLM
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 配置采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 准备提示词
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    
    # 应用聊天模板
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # 生成文本
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
