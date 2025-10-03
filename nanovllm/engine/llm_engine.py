"""
LLM引擎模块
提供高级API接口，管理整个文本生成流程
包括请求处理、调度、执行和结果返回
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM引擎主类
    提供完整的文本生成服务，包括请求管理、调度、执行和结果返回
    支持多GPU并行、进度显示、吞吐量统计等功能
    """

    def __init__(self, model, **kwargs):
        """
        初始化LLM引擎
        
        Args:
            model: 模型路径或标识符
            **kwargs: 其他配置参数，会被传递给Config类
        """
        # 解析配置参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # 初始化多进程
        self.ps = []      # 子进程列表
        self.events = []  # 事件列表用于进程通信
        ctx = mp.get_context("spawn")
        
        # 启动工作进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 初始化主进程模型运行器
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 设置结束符ID
        
        # 初始化调度器
        self.scheduler = Scheduler(config)
        
        # 注册退出清理函数
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源并退出
        通知所有子进程退出，等待进程结束
        """
        # 通知模型运行器退出
        self.model_runner.call("exit")
        del self.model_runner
        
        # 等待所有子进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加文本生成请求
        
        Args:
            prompt: 输入文本或token ID列表
            sampling_params: 采样参数配置
        """
        # 文本转token
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # 创建序列对象
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理
        包括调度、模型推理和后处理
        
        Returns:
            tuple: (完成的输出列表, token数量统计)
                - 完成的输出列表: [(序列ID, token列表), ...]
                - token数量统计: 正数表示prefill的token数，负数表示decode的序列数
        """
        # 调度序列
        seqs, is_prefill = self.scheduler.schedule()
        
        # 执行模型推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 后处理
        self.scheduler.postprocess(seqs, token_ids)
        
        # 收集完成的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 统计token数量：prefill为正数，decode为负数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已处理完成
        
        Returns:
            True表示所有请求已完成，False表示还有请求在处理
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量文本生成主接口
        
        Args:
            prompts: 输入文本列表或token ID列表列表
            sampling_params: 采样参数，可以是单个配置或配置列表
            use_tqdm: 是否显示进度条
            
        Returns:
            生成结果列表，每个元素包含生成的文本和token ID
        """
        # 初始化进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 标准化采样参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # 收集结果
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 主循环：处理所有请求
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # 更新进度条和吞吐量统计
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # 排序并解码结果
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        # 关闭进度条
        if use_tqdm:
            pbar.close()
        
        return outputs
