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

## 🎯 MLGKA Examples

### Text Classification with MLGKA Layers
The model module now includes `MLGKALayer` - a complete transformer block that combines:
- **Multi-Latent Attention (MLA)**: Low-rank factorization for efficiency
- **Grouped Query Attention (GQA)**: Shared K/V heads for memory efficiency  
- **Laplacian Kernel Attention**: Non-linear attention patterns via Random Kitchen Sinks
- **SwiGLU Feed-Forward**: Efficient activation with optional frozen projections

```bash
# Run comprehensive MLGKA text classification example
python examples/mlgka_text_classification.py

# Run simple MLGKA example
python examples/simple_mlgka_example.py
```

**Features**:
- Complete transformer blocks ready for any architecture
- Configurable attention presets (bert_style, gpt_style, efficient_gqa, mla_attention, kernel_attention, mlgka, custom)
- Flexible configuration system with YAML support
- Memory-efficient GQA and MLA implementations

## 📚 Working Examples

All examples have been verified and are ready to run:

### 🏗️ Architecture Examples
```bash
# Basic architecture creation
python examples/architecture_example.py

# Simple architecture with different attention types
python examples/simple_architecture_example.py
```

### ⚙️ Configuration Examples
```bash
# Configuration system usage
python examples/config_example.py

# Attention configuration examples
python examples/attention_config_example.py
```

### 🎯 MLGKA Examples
```bash
# Comprehensive text classification with MLGKA
python examples/mlgka_text_classification.py

# Simple MLGKA layer usage
python examples/simple_mlgka_example.py
```

### 🚀 Training Examples
```bash
# Progressive training with TinyBERT
cd examples/tiny_bert
python train_tiny_bert.py

# Progressive training system
python examples/progressive_training_system_example.py

# Progressive QLoRA training
python examples/progressive_qlora_example.py
```

### 🔧 Utility Examples
```bash
# Checkpointing examples
python examples/checkpointing_example.py

# Tokenizer integration
python examples/tokenizer_integration_example.py

# Dual LoRA example
python examples/dual_lora_example.py
```

**All examples are tested and working!** 🎉

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

### Progressive Training Usage
```python
from src.training import ProgressiveTrainer, ProgressiveRackBuilder
from src.config.base import StackWiseConfig

# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Create progressive rack builder
rack_builder = ProgressiveRackBuilder(config=config)

# Add stacks progressively
stack1 = rack_builder.append_stack(n_blocks=4)
stack2 = rack_builder.append_stack(n_blocks=4)

# Train with progressive trainer
trainer = ProgressiveTrainer(config=config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=2)
```

### Block-wise Training
```python
from src.training import Trainer

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
