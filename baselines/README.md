# StackWise Baselines Module

A comprehensive benchmarking framework for encoder-decoder model families with Hydra configuration management and experimental tracking.

## 🎯 Overview

This module provides:
- **Reproducible baselines** for BERT, GPT, and LLaMA families
- **Depth-as-time training** comparison with classical approaches
- **Comprehensive evaluation** on GLUE, language modeling, and reasoning tasks
- **Hydra-powered configuration** management for complex experiments
- **Automated experimental tracking** and result analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- CUDA-capable GPU (optional, for training)

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Navigate to the project root
cd /path/to/stack-wise

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install the main package with advanced dependencies
uv pip install -e .[advanced]

# Install the baselines module
cd baselines
uv pip install -e .
```

#### Option 2: Using pip

```bash
# Navigate to the project root
cd /path/to/stack-wise

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the main package with advanced dependencies
pip install -e .[advanced]

# Install the baselines module
cd baselines
pip install -e .
```

### Basic Usage

#### Using uv (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate

# Train BERT-base on GLUE (reproduction)
uv run python scripts/train.py --config-name=bert_base_glue

# Train GPT-2-small with depth-as-time
uv run python scripts/train.py model=decoder/gpt2_family/small training=depth_time

# Run ablation study
uv run python scripts/benchmark.py --config-name=bert_depth_time_vs_classical

# Evaluate trained model
uv run python scripts/evaluate.py --config-name=bert_base_glue model_path=./checkpoints/model.pt
```

#### Using pip

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Train BERT-base on GLUE (reproduction)
python scripts/train.py --config-name=bert_base_glue

# Train GPT-2-small with depth-as-time
python scripts/train.py model=decoder/gpt2_family/small training=depth_time

# Run ablation study
python scripts/benchmark.py --config-name=bert_depth_time_vs_classical

# Evaluate trained model
python scripts/evaluate.py --config-name=bert_base_glue model_path=./checkpoints/model.pt
```

## 📁 Directory Structure

```
baselines/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── model/                 # Model configurations
│   │   ├── encoder/           # BERT, ModernBERT families
│   │   └── decoder/           # GPT-2, LLaMA families
│   ├── training/              # Training regime configs
│   ├── benchmarks/            # Benchmark task configs
│   ├── datasets/              # Dataset-specific configs
│   └── experiments/           # Complete experiment configs
├── src/                       # Source code
│   ├── evaluation/            # Evaluation harness
│   ├── benchmarks/            # Benchmark implementations
│   └── utils/                 # Utility functions
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── benchmark.py          # Benchmark runner
└── experiments/              # Experiment outputs
```

## 🔧 Configuration

### Model Configurations

**BERT Family:**
- `tiny.yaml` - TinyBERT (14M params)
- `base.yaml` - BERT-base (110M params)
- `large.yaml` - BERT-large (340M params)

**GPT-2 Family:**
- `small.yaml` - GPT-2-small (124M params)
- `medium.yaml` - GPT-2-medium (355M params)
- `large.yaml` - GPT-2-large (774M params)

### Training Regimes

**Classical Training:**
```yaml
strategy: "end_to_end"
end_to_end_scope: "rackwise"
progressive:
  enabled: false
```

**Depth-as-Time Training:**
```yaml
strategy: "progressive"
end_to_end_scope: "stackwise"
progressive:
  enabled: true
  time_interpretation: "depth"
  building_mode: "prepend"
```

### Benchmark Tasks

**NLU Tasks (GLUE):**
- CoLA, SST-2, MRPC, STS-B, QQP
- MNLI, QNLI, RTE, WNLI

**NLG Tasks:**
- WikiText-103, PTB (perplexity)
- LAMBADA, HellaSwag, PIQA (reasoning)
- CNN/DailyMail (summarization)

## 🧪 Experiments

### Reproduction Experiments

Reproduce existing model performance:

```bash
# Activate virtual environment
source .venv/bin/activate  # or source venv/bin/activate

# BERT-base GLUE reproduction
uv run python scripts/train.py --config-name=bert_base_glue_reproduction

# GPT-2-small language modeling
uv run python scripts/train.py --config-name=gpt2_small_wikitext
```

### Ablation Studies

Compare training regimes:

```bash
# Activate virtual environment
source .venv/bin/activate  # or source venv/bin/activate

# Depth-as-time vs classical
uv run python scripts/benchmark.py --config-name=bert_depth_time_vs_classical

# Multi-run parameter sweep
uv run python scripts/train.py --config-name=bert_base_glue --multirun training.lr=1e-5,2e-5,5e-5
```

### Scaling Studies

Analyze scaling laws:

```bash
# Activate virtual environment
source .venv/bin/activate  # or source venv/bin/activate

# Model size scaling
uv run python scripts/benchmark.py --config-name=scaling_study --multirun model_variant=tiny,small,base,large

# Compute equalization
uv run python scripts/train.py --config-name=compute_equalized_scaling
```

## 📊 Evaluation

### Metrics

**NLU Metrics:**
- Accuracy, F1-score
- Matthews correlation
- Pearson/Spearman correlation

**NLG Metrics:**
- Perplexity
- BLEU, ROUGE scores
- Task-specific accuracy

### Results Organization

```
experiments/
├── {experiment_name}/
│   ├── config.yaml              # Final configuration
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # Training logs
│   ├── outputs/                 # Evaluation results
│   │   ├── metrics.json
│   │   ├── evaluation_report.md
│   │   └── comparison_plots.png
│   └── metadata/                # Run metadata
```

## 🔬 Advanced Usage

### Custom Configurations

Create custom experiment configs:

```yaml
# configs/experiments/custom/my_experiment.yaml
defaults:
  - model: encoder/bert_family/base
  - training: depth_time
  - benchmark: nlu/glue

# Custom overrides
model:
  d_model: 512
  n_heads: 8

training:
  batch_size: 32
  learning_rate: 1e-4
```

### Multi-Run Experiments

Run parameter sweeps:

```bash
# Activate virtual environment
source .venv/bin/activate  # or source venv/bin/activate

# Learning rate sweep
uv run python scripts/train.py --config-name=bert_base_glue --multirun training.lr=1e-5,2e-5,5e-5

# Model size sweep
uv run python scripts/train.py --config-name=scaling_study --multirun model_variant=tiny,small,base,large

# Training regime comparison
uv run python scripts/train.py --config-name=ablation_study --multirun training_regime=classical,depth_time,hybrid
```

### Custom Benchmarks

Add new benchmark tasks:

```yaml
# configs/benchmarks/custom/my_benchmark.yaml
name: "my_benchmark"
tasks:
  - name: "my_task"
    metric: "accuracy"
    target_score: 85.0
```

## 📈 Analysis & Reporting

### Automated Reports

The framework generates comprehensive reports:

- **Reproduction Reports** - Compare with baseline models
- **Ablation Analysis** - Statistical significance testing
- **Scaling Analysis** - Power law fitting and visualization
- **Performance Plots** - Automated visualization generation

### Statistical Analysis

- Confidence intervals
- Multiple comparison correction
- Effect size analysis
- Power analysis

## 🛠️ Development

### Adding New Models

1. Create model config in `configs/model/`
2. Implement model architecture
3. Add to training scripts
4. Create reproduction configs

### Adding New Benchmarks

1. Create benchmark config in `configs/benchmarks/`
2. Implement task loader in `src/evaluation/`
3. Add metrics computation
4. Create evaluation configs

### Adding New Training Regimes

1. Create training config in `configs/training/`
2. Implement trainer class
3. Add to training script
4. Create ablation configs

## 📚 Examples

See the `examples/` directory for:
- Basic training examples
- Configuration examples
- Evaluation examples
- Analysis examples

## 🔧 Troubleshooting

### Virtual Environment Issues

**Problem**: `uv: command not found`
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

**Problem**: Virtual environment not activated
```bash
# Check if virtual environment is active
echo $VIRTUAL_ENV

# Activate virtual environment
source .venv/bin/activate  # for uv
# or
source venv/bin/activate   # for pip
```

**Problem**: Package not found after installation
```bash
# Reinstall in development mode
uv pip install -e .[advanced]
# or
pip install -e .[advanced]
```

**Problem**: CUDA/GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration Issues

**Problem**: Hydra configuration not found
```bash
# Run from the baselines directory
cd baselines
uv run python scripts/train.py --config-name=bert_base_glue
```

**Problem**: Model checkpoint not found
```bash
# Check checkpoint path
ls -la checkpoints/
# Update model_path in evaluation script
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers
- Hydra configuration framework
- GLUE and SuperGLUE benchmarks
- EleutherAI evaluation harness
