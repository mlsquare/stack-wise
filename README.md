# 🧠 StackWise — Layer-Wise Transformer with Mask-Diffusion Objective

A **comprehensive PyTorch framework** for training GPT-2/LLaMA-style decoder models **layer-by-layer** with bidirectional attention, modern attention mechanisms, and mask-diffusion objectives. Features advanced quantization, QLoRA adapters, and fusion training capabilities.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets numpy tqdm pyyaml
```

### 2. Run GPT-2 Fusion Training Example
```bash
# Navigate to example directory
cd examples/gpt2_fusion

# Prepare data (creates 10k English corpus)
python3 data_loader.py --prepare

# Run the complete training demo
python3 simple_train.py
```

### 3. Test Core Functionality
```bash
# Test FusionTrainer functionality
python3 test_fusion_direct.py

# Run example training
python3 example_fusion_training.py
```

**Status**: ✅ **Complete** - All training modules implemented and tested!

## 🏗️ Architecture

### Key Features
- **Bidirectional Training**: Use bidirectional attention (BERT-style) for more efficient learning
- **Task-Specific Fine-tuning**: Switch to causal attention (GPT-style) for autoregressive tasks
- **Modern Attention**: Support for standard, GQA, MLA, and kernel-based attention
- **Layer-wise Training**: Train each layer independently with cached activations
- **Mask-Diffusion**: Variable masking rates (15% to 90%) for better representation learning
- **Fusion Training**: Train multiple layers/blocks simultaneously with frozen or trainable backbones
- **Advanced Quantization**: FP4, FP8, FP16, FP32 precision support with QLoRA adapters
- **Memory Efficiency**: Persistent quantization and gradient clearing for large models
- **Disk Backup System**: Run ID-based organization with full-precision weight storage

### Training Modes
- **Layer-wise**: Train each layer independently (BlockTrainer with block_size=1)
- **Block-wise**: Train groups of layers together (BlockTrainer with block_size>1)
- **Fusion**: Train multiple blocks with frozen or trainable backbone (FusionTrainer)

### Configuration
Edit `config.yaml` or `examples/gpt2_fusion/gpt2.yaml` to customize:
- Model architecture (dimensions, layers, attention type)
- Training parameters (learning rate, batch size, steps)
- Attention modes (bidirectional vs causal)
- Fine-tuning modes (CLM, MLM, diffusion)
- Quantization settings (precision, QLoRA adapters)
- Time-step masking (progressive masking across layers)

## 📁 Project Structure

```
src/
├── config/                    # Configuration management
│   └── base.py               # Base configuration classes
├── model/                    # Model components
│   ├── layers.py             # MLGKALayer, LexicalKernelManager, SwiGLUFFN
│   └── __init__.py           # Model exports
├── training/                 # Training pipelines
│   ├── core/                 # Core training modules
│   │   ├── unified_trainer.py    # Main entry point
│   │   ├── block_trainer.py      # Block-based training
│   │   └── fusion_trainer.py     # Fusion training with quantization
│   └── __init__.py           # Training exports
├── data/                     # Data handling and preprocessing
└── utils/                    # Utilities and helpers

docs/                        # Documentation
├── README.md                # Documentation index
├── TRAINER_MODULE.md        # Comprehensive trainer guide
├── CONFIGURATION_GUIDE.md   # Configuration reference
└── API_REFERENCE.md         # API documentation

tests/                       # Test suite
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── examples/                # Example tests
├── run_tests.py            # Test runner
└── README.md               # Test documentation

examples/
└── gpt2_fusion/              # GPT-2 fusion training example
    ├── train_gpt2_fusion.py  # Main training script
    ├── simple_train.py       # Simplified demo
    ├── data_loader.py        # Data preparation
    ├── evaluate_gpt2.py      # Model evaluation
    ├── gpt2.yaml            # GPT-2 specific config
    └── README.md            # Example documentation
```

## 🔧 Usage

### GPT-2 Fusion Training (Recommended)
```bash
cd examples/gpt2_fusion
python3 simple_train.py
```

### Direct FusionTrainer Usage
```python
from src.training.core.fusion_trainer import FusionTrainer
from src.config.base import StackWiseConfig

# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Initialize trainer
trainer = FusionTrainer(config)

# Train with frozen backbone
trainer.train_with_frozen_backbone()
```

### Block-wise Training
```python
from src.training.core.block_trainer import BlockTrainer

# Initialize block trainer
trainer = BlockTrainer(config, masking_strategy, quantization_manager, cache_manager)

# Train blocks
trainer.train_blocks(all_blocks)
```

## 🧪 Development

### Testing
```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --unit          # Unit tests
python tests/run_tests.py --integration # Integration tests
python tests/run_tests.py --examples    # Example tests

# Run with verbose output
python tests/run_tests.py --verbose

# Test GPT-2 fusion example
cd examples/gpt2_fusion
python3 simple_train.py
```

### Code Quality
```bash
# Format code
black src/ examples/

# Lint code
flake8 src/ examples/

# Type checking
mypy src/
```

## 📚 Documentation

- **[Documentation Index](docs/README.md)** - Start here for comprehensive documentation
- [Trainer Module Documentation](docs/TRAINER_MODULE.md) - Comprehensive guide to training modes
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md) - Complete configuration reference
- [API Reference](docs/API_REFERENCE.md) - Detailed API documentation
- [GPT-2 Fusion Example](examples/gpt2_fusion/README.md) - Complete training example
- [Model Architecture](src/model/README.md) - Model components and attention mechanisms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Roberta-Diffusion](https://github.com/nathan-barry/RoBERTaDiffusion)
- [MMDiffBERT](https://github.com/mlsquare/mmDiffBERT)
- [DeepSeek-V2/V3](https://github.com/deepseek-ai/DeepSeek-V2)
- [BERT](https://github.com/google-research/bert)
- [GPT](https://github.com/openai/gpt-2)
