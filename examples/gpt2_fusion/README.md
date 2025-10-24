# GPT-2 Fusion Training Example

This example demonstrates how to train a GPT-2 style model using the FusionTrainer with mask-diffusion objectives on a small English corpus.

## Overview

- **Model**: GPT-2 Small (12 layers, 768 hidden, 12 heads)
- **Dataset**: 10k English examples
- **Training**: Fusion training with progressive masking
- **Features**: QLoRA adapters, quantization, time-step masking

## Files

- `gpt2.yaml` - GPT-2 specific configuration
- `train_gpt2_fusion.py` - Main training script
- `data_loader.py` - Data loading utilities
- `evaluate_gpt2.py` - Evaluation script
- `data/` - Dataset directory

## Quick Start

1. **Setup Environment**:
   ```bash
   source .venv/bin/activate
   pip install torch transformers datasets
   ```

2. **Prepare Data**:
   ```bash
   python data_loader.py --prepare
   ```

3. **Train Model**:
   ```bash
   python train_gpt2_fusion.py
   ```

4. **Evaluate Model**:
   ```bash
   python evaluate_gpt2.py
   ```

## Training Strategy

### Block Structure
- **Block 1**: Layers 0-3 (10-40% masking)
- **Block 2**: Layers 4-7 (40-70% masking)  
- **Block 3**: Layers 8-11 (70-99% masking)

### Progressive Masking
- **Time Steps**: 12 discrete time steps
- **Mask Fractions**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
- **Progression**: Encoder-like → Decoder-like behavior

### Memory Efficiency
- **Quantization**: FP16 precision for backbone
- **QLoRA**: Low-rank adapters for efficient fine-tuning
- **Disk Backup**: Full-precision weights saved before conversion

## Expected Results

- **Memory Usage**: 2-4x reduction with quantization
- **Training Speed**: Faster with QLoRA adapters
- **Model Quality**: Comparable to standard GPT-2 training
- **Convergence**: Stable training with progressive masking

## Configuration

Key parameters in `gpt2.yaml`:

```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  attention_mode: "causal"

training:
  mode: "fused"
  fusion_mode: "frozen"
  total_blocks: 3
  block_size: 4
  qlora_enabled: true
  quantization_type: "fp16"
  time_step_masking: true
```

## Monitoring

- **Logs**: Training progress and metrics
- **Checkpoints**: Model weights and optimizer state
- **Validation**: Perplexity and accuracy metrics
- **Memory**: GPU memory usage tracking

## Troubleshooting

- **Memory Issues**: Reduce batch size or use gradient checkpointing
- **Training Instability**: Adjust learning rates
- **Poor Convergence**: Check mask fractions and data quality
- **Import Errors**: Ensure virtual environment is activated
