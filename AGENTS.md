# Nano-vLLM Architecture Overview

## Project Overview

Nano-vLLM is a lightweight, high-performance inference engine for large language models, built from scratch as a simplified alternative to vLLM. The project implements key optimizations including prefix caching, tensor parallelism, Torch compilation, and CUDA graph acceleration while maintaining a clean, readable codebase of approximately 1,200 lines of Python code.

### Core Architecture

The system follows a modular architecture with clear separation of concerns:

- **Engine Layer** (`nanovllm/engine/`): Core inference orchestration including scheduling, sequence management, and model execution
- **Model Layer** (`nanovllm/models/`): Model-specific implementations (currently Qwen3 support)
- **Layer Components** (`nanovllm/layers/`): Reusable neural network components (attention, activation, normalization, etc.)
- **Configuration** (`nanovllm/config.py`): Centralized configuration management
- **Utilities** (`nanovllm/utils/`): Helper functions for model loading and context management

### Key Components

**LLMEngine** (`nanovllm/engine/llm_engine.py:15`): Main orchestrator that coordinates the entire inference pipeline, manages multiprocessing for tensor parallelism, and handles request lifecycle.

**Scheduler** (`nanovllm/engine/scheduler.py:8`): Implements intelligent batching strategies for both prefill and decode phases, manages memory allocation through block manager, and handles sequence preemption.

**ModelRunner** (`nanovllm/engine/model_runner.py:15`): Executes model computation with CUDA graph optimization, handles distributed inference across multiple GPUs, and manages KV cache allocation.

**BlockManager** (`nanovllm/engine/block_manager.py`): Manages KV cache memory allocation using a block-based approach for efficient memory utilization.

## Build & Commands

### Installation
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm

# Install in development mode
pip install -e .
```

### Running Examples
```bash
# Basic inference example
python example.py

# Performance benchmarking
python bench.py
```

### Model Download
```bash
# Manual model download (recommended)
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Code Style

### Python Version
- Requires Python 3.10-3.12
- Uses modern Python features including type hints and dataclasses

### Dependencies
- **Core**: torch>=2.4.0, transformers>=4.51.0
- **Performance**: triton>=3.0.0, flash-attn
- **Utilities**: xxhash for efficient hashing

### Code Organization
- Modular design with clear separation between engine, model, and layer components
- Uses dataclasses for configuration management
- Implements multiprocessing for tensor parallelism support
- Follows PyTorch conventions for tensor operations and device management

### Key Patterns
- **Configuration**: Centralized in `Config` dataclass with validation in `__post_init__`
- **Multiprocessing**: Uses spawn context for tensor parallelism with shared memory communication
- **Memory Management**: Block-based KV cache allocation with preemption support
- **Error Handling**: Assertions for configuration validation, graceful cleanup with atexit handlers

## Testing

### Current Testing Approach
- No formal test framework currently implemented
- Testing is done through example scripts (`example.py`) and benchmarking (`bench.py`)
- Performance validation through throughput comparison with vLLM

### Testing Recommendations
- Test with various model sizes and tensor parallelism configurations
- Validate memory usage stays within configured limits
- Verify correctness of generated outputs against reference implementations
- Performance regression testing using benchmark suite

### Test Configuration
- Default test model: Qwen3-0.6B
- Hardware reference: RTX 4070 Laptop (8GB)
- Test parameters: 256 sequences, input/output lengths 100-1024 tokens

## Security

### Model Loading
- Supports local model directories only (no remote model loading)
- Uses HuggingFace transformers for secure model loading
- Validates model configuration against supported architectures

### Memory Safety
- Implements memory utilization limits (default 90% GPU memory)
- Uses shared memory for inter-process communication with proper cleanup
- Includes atexit handlers for graceful shutdown and resource cleanup

### Data Protection
- No persistent storage of user inputs or model outputs
- All processing happens in-memory
- No network communication except for initial model download

## Configuration

### Core Parameters (`nanovllm/config.py:7`)
- `max_num_batched_tokens`: Maximum tokens per batch (default: 16384)
- `max_num_seqs`: Maximum concurrent sequences (default: 512)
- `max_model_len`: Maximum sequence length (default: 4096)
- `gpu_memory_utilization`: GPU memory usage limit (default: 0.9)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `enforce_eager`: Disable CUDA graph optimization (default: False)

### Sampling Parameters (`nanovllm/sampling_params.py:5`)
- `temperature`: Sampling temperature (default: 1.0, minimum: 1e-10)
- `max_tokens`: Maximum output tokens (default: 64)
- `ignore_eos`: Continue generation after EOS token (default: False)

### Environment Requirements
- CUDA-capable GPU with sufficient memory
- NCCL for multi-GPU communication
- Compatible PyTorch installation with CUDA support

### Performance Tuning
- Enable CUDA graphs by setting `enforce_eager=False`
- Adjust `max_num_batched_tokens` based on GPU memory
- Use tensor parallelism for large models across multiple GPUs
- Configure `kvcache_block_size` in multiples of 256 for optimal memory alignment