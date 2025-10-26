# Changelog

All notable changes to this project will be documented in this file.

## [0.7.0] - 2024-12-01

### Added
- **Baselines Module**: Comprehensive benchmarking framework for encoder-decoder model families
- **Hydra Integration**: Full Hydra configuration management system
- **Model Configurations**: BERT and GPT-2 family model configs (tiny, base, large)
- **Training Regimes**: Classical and depth-as-time training configurations
- **Benchmark Tasks**: GLUE (NLU) and language modeling (NLG) evaluation tasks
- **Experimental Tracking**: Automated logging, checkpointing, and result analysis
- **Multi-run Support**: Parameter sweeps and comparative experiments
- **Evaluation Harness**: Unified evaluation system for all model families
- **Documentation**: Comprehensive guides and tutorials for Hydra usage

### Features
- **Reproducible Baselines**: Pre-configured experiments for BERT and GPT-2 reproduction
- **Ablation Studies**: Depth-as-time vs classical training comparisons
- **Scaling Studies**: Model size and compute scaling analysis
- **Configuration Management**: Hierarchical YAML-based configuration system
- **Command-line Interface**: Easy-to-use CLI with override support
- **Educational Resources**: Step-by-step tutorials and examples

### Configuration Structure
```
baselines/
├── configs/                    # Hydra configuration files
│   ├── model/                 # Model family configs
│   ├── training/              # Training regime configs
│   ├── benchmarks/            # Evaluation task configs
│   └── experiments/           # Complete experiment configs
├── src/                       # Source code
│   ├── evaluation/            # Evaluation harness
│   ├── benchmarks/            # Benchmark implementations
│   └── utils/                 # Utility functions
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── benchmark.py          # Benchmark runner
└── examples/                  # Tutorials and examples
```

### Dependencies Added
- `hydra-core>=1.3.0` - Configuration management
- `omegaconf>=2.3.0` - Configuration objects
- `scikit-learn>=1.3.0` - Evaluation metrics
- `scipy>=1.11.0` - Statistical analysis
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `pandas>=2.0.0` - Data analysis

### Usage Examples
```bash
# Basic training
uv run python scripts/train.py model=encoder/bert_family/tiny

# Complete experiment
uv run python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue

# Mix and match
uv run python scripts/train.py model=encoder/bert_family/base training=depth_time

# Multi-run experiments
uv run python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base
```

### Documentation
- **README.md**: Updated with baselines module information
- **baselines/README.md**: Comprehensive baselines documentation
- **HYDRA_VISUAL_GUIDE.md**: Visual guide for Hydra usage
- **HYDRA_QUICK_REFERENCE.md**: Quick reference for commands
- **hydra_simple_explanation.py**: Interactive tutorial
- **IMPLEMENTATION_SUMMARY.md**: Detailed implementation overview

### Breaking Changes
- None (backward compatible)

### Migration Guide
- No migration required
- New baselines module is additive
- Existing functionality remains unchanged

## [0.6.0] - Previous Release

### Features
- Layer-wise Transformer architecture
- Mask-diffusion objectives
- Progressive training capabilities
- QLoRA and quantization support
- Fusion training system

### Architecture
- Block/Stack/Rack hierarchy
- Bidirectional and causal attention
- Modern attention mechanisms (GQA, MLA, kernel attention)
- Advanced quantization and memory management

---

For more details, see the [README.md](README.md) and [baselines/README.md](baselines/README.md).
