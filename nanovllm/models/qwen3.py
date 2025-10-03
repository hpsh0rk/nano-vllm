"""
Qwen3模型实现
基于transformers的Qwen3模型架构
支持张量并行和高效的推理
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3注意力层
    实现多头注意力机制，支持张量并行
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        """
        初始化Qwen3注意力层
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            num_kv_heads: key/value头数
            max_position: 最大位置编码长度
            head_dim: 每个注意力头的维度，默认为hidden_size//num_heads
            rms_norm_eps: RMSNorm的epsilon值
            qkv_bias: QKV线性层是否使用偏置
            rope_theta: RoPE的theta参数
            rope_scaling: RoPE缩放配置
        """
        super().__init__()
        tp_size = dist.get_world_size()
        
        # 计算张量并行后的头数
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0, "注意力头数必须能被进程数整除"
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0, "KV头数必须能被进程数整除"
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        # 计算维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5  # 注意力缩放因子

        # QKV投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        # 输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # 注意力计算
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
        # Q和K的归一化层
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            positions: 位置编码张量
            hidden_states: 隐藏状态张量
            
        Returns:
            注意力输出张量
        """
        # QKV投影
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 重塑为多头格式并归一化
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        
        # 注意力计算
        o = self.attn(q, k, v)
        
        # 输出投影
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP层
    实现前馈神经网络，使用SwiGLU激活函数
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        初始化Qwen3 MLP层
        
        Args:
            hidden_size: 隐藏层维度
            intermediate_size: 中间层维度
            hidden_act: 激活函数类型，必须为"silu"
        """
        super().__init__()
        
        # gate和up投影的合并层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate和up两个投影
            bias=False,
        )
        
        # down投影层
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        
        # 验证激活函数
        assert hidden_act == "silu", f"不支持的激活函数: {hidden_act}"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            MLP输出张量
        """
        # gate和up投影
        gate_up = self.gate_up_proj(x)
        
        # SwiGLU激活
        x = self.act_fn(gate_up)
        
        # down投影
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3解码器层
    实现一个完整的Transformer解码器层
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化Qwen3解码器层
        
        Args:
            config: Qwen3配置对象
        """
        super().__init__()
        
        # 自注意力层
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # MLP层
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # 归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            positions: 位置编码张量
            hidden_states: 隐藏状态张量
            residual: 残差张量（可选）
            
        Returns:
            tuple: (隐藏状态, 残差)
        """
        # 输入归一化
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 自注意力
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 后注意力归一化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3模型
    完整的Transformer模型，包含嵌入层和多个解码器层
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化Qwen3模型
        
        Args:
            config: Qwen3配置对象
        """
        super().__init__()
        
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        # 解码器层列表
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID张量
            positions: 位置编码张量
            
        Returns:
            最终隐藏状态张量
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        # 逐层处理
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 最终归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3因果语言模型
    完整的语言模型，包含Qwen3模型和输出头
    """
    
    # 模块映射，用于权重加载
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        """
        初始化Qwen3因果语言模型
        
        Args:
            config: Qwen3配置对象
        """
        super().__init__()
        
        # 核心模型
        self.model = Qwen3Model(config)
        
        # 语言模型输出头
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 权重绑定：如果启用，共享嵌入层和输出层的权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
