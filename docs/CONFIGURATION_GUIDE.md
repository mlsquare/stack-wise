# Configuration Guide

This guide explains how to configure the Stack-Wise training system for different use cases.

## Quick Start

### Basic Configuration
```yaml
# config.yaml
model:
  vocab_size: 50257
  d_model: 768
  n_layers: 12
  n_heads: 12
  attention_mode: "causal"
  tie_embeddings: true

training:
  mode: "fused"              # layerwise | blockwise | fused
  fusion_mode: "frozen"      # frozen | trainable
  total_blocks: 3
  block_size: 4
  learning_rate: 5.0e-4
  batch_size: 8
  max_steps: 1000
```

## Dual-LoRA Configuration

### Progressive QLoRA Setup
```yaml
training:
  progressive:
    enabled: true
    trunk_strategy: "qlora"        # "frozen" or "qlora"
    new_stack_precision: "full"    # "full", "half", "bfloat16", "nvfp4"
    
    # Stack LoRA parameters (added to each stack)
    qlora_enabled: true
    qlora_strategy: "progressive"  # "simplified", "progressive", "variable"
    qlora_rank: 16
    qlora_alpha: 32
    
    # Progressive QLoRA parameters (added to trunk when new stacks are added)
    progressive_qlora: true
    progressive_qlora_rank: 8
    progressive_qlora_alpha: 16
    
    # Progressive patterns (for qlora_strategy: "progressive")
    qlora_rank_pattern: "increasing"  # "constant", "increasing", "decreasing", "linear"
    qlora_alpha_pattern: "constant"   # "constant", "increasing", "decreasing", "linear"
    
    # Variable QLoRA configurations (for qlora_strategy: "variable")
    qlora_configs:
      0: {rank: 8, alpha: 16}       # Stack 0: small QLoRA
      1: {rank: 16, alpha: 32}      # Stack 1: medium QLoRA  
      2: {rank: 32, alpha: 64}      # Stack 2: large QLoRA
      3: {rank: 64, alpha: 128}     # Stack 3: very large QLoRA
```

### Precision Options
```yaml
# Supported precision modes
precision_options:
  - "full"      # torch.float32
  - "half"      # torch.float16
  - "bfloat16"  # torch.bfloat16
  - "nvfp4"     # NVIDIA FP4 precision
  - "qlora"     # QLoRA training (not a precision)
```

## Training Modes

### 1. Layer-wise Training
```yaml
training:
  mode: "layerwise"
  block_size: 1              # Each layer is a separate block
  epochs_per_layer: 5
```

### 2. Block-wise Training
```yaml
training:
  mode: "blockwise"
  block_size: 4              # 4 layers per block
  total_blocks: 3
```

### 3. Fusion Training
```yaml
training:
  mode: "fused"
  fusion_mode: "frozen"       # frozen | trainable
  total_blocks: 3
  block_size: 4
```

## Advanced Features

### Quantization & QLoRA
```yaml
training:
  qlora_enabled: true
  qlora_lr: 1.0e-5
  current_block_lr: 5.0e-4
  quantization_enabled: true
  quantization_type: "fp16"   # fp4 | fp8 | fp16 | fp32
```

### Time-step Masking
```yaml
training:
  time_step_masking: true
  num_time_steps: 12
  time_step_mask_fractions: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
```

### Progressive Masking
```yaml
training:
  min_mask_fraction: 0.1
  max_mask_fraction: 0.99
  mask_schedule_type: "linear"  # linear | exponential | cosine
```

## Model Architectures

### GPT-2 Small
```yaml
model:
  vocab_size: 50257
  d_model: 768
  n_layers: 12
  n_heads: 12
  d_ff: 3072
  attention_mode: "causal"
  tie_embeddings: true
```

### GPT-2 Medium
```yaml
model:
  vocab_size: 50257
  d_model: 1024
  n_layers: 24
  n_heads: 16
  d_ff: 4096
  attention_mode: "causal"
  tie_embeddings: true
```

### Custom Architecture
```yaml
model:
  vocab_size: 32000
  d_model: 1024
  n_layers: 16
  n_heads: 16
  n_kv_heads: 8              # For GQA
  d_ff: 4096
  attention_type: "gqa"      # standard | gqa | mla | kernel
  attention_mode: "bidirectional"  # bidirectional | causal
  use_rope: true
  tie_embeddings: true
```

## Data Configuration

### Dataset Settings
```yaml
data:
  dataset_path: "./data/corpus.json"
  use_dummy_data: false
  num_samples: 10000
  tokenizer_path: "gpt2"
  max_length: 512
  num_workers: 4
  shuffle: true
```

### Tokenizer Integration
```yaml
model:
  tokenizer_embedding:
    family: "gpt2"            # gpt2 | llama-3-8b | mistral-7b
    embedding_option: "embed_tokens"
    freeze_embeddings: false
    adapter_hidden_dim: null
```

## Training Parameters

### Learning Rates
```yaml
training:
  learning_rate: 5.0e-4      # Base learning rate
  qlora_lr: 1.0e-5           # QLoRA adapter learning rate
  current_block_lr: 5.0e-4   # Current block learning rate
  weight_decay: 0.01
```

### Batch and Sequence
```yaml
training:
  batch_size: 8
  seq_len: 512
  max_steps: 1000
  log_interval: 10
  save_interval: 100
```

### Memory Management
```yaml
training:
  run_id: "my_experiment"    # Unique identifier for this run
  checkpoint_dir: "./checkpoints"
  save_fused_checkpoints: true
  mixed_precision: true
```

## Examples

### Minimal Configuration
```yaml
model:
  d_model: 256
  n_layers: 4
  n_heads: 4

training:
  mode: "fused"
  batch_size: 4
  max_steps: 100
```

### Production Configuration
```yaml
model:
  vocab_size: 50257
  d_model: 768
  n_layers: 12
  n_heads: 12
  attention_mode: "causal"
  tie_embeddings: true

training:
  mode: "fused"
  fusion_mode: "frozen"
  total_blocks: 3
  block_size: 4
  qlora_enabled: true
  quantization_type: "fp16"
  time_step_masking: true
  batch_size: 8
  max_steps: 1000
  run_id: "gpt2_fusion_v1"
```

## Validation

The configuration system includes validation to catch common errors:

- **Required fields**: All essential parameters must be specified
- **Value ranges**: Learning rates, dimensions, etc. are validated
- **Dependencies**: Related parameters are checked for consistency
- **Type checking**: Parameter types are validated

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Memory Issues**: Reduce batch size or enable quantization
3. **Training Instability**: Check learning rates and mask fractions
4. **Poor Convergence**: Verify data quality and model architecture

### Debug Mode
```yaml
training:
  debug: true
  log_level: "DEBUG"
```

### Performance Monitoring
```yaml
training:
  log_interval: 10
  save_interval: 100
  validation_interval: 50
```
