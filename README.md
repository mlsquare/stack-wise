# 🧠 StackWise — Layer-Wise Transformer with Mask-Diffusion Objective

A **from-scratch PyTorch scaffold** for training GPT-2/LLaMA-style decoder models **layer-by-layer** with bidirectional attention, modern attention mechanisms, and mask-diffusion objectives.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Or use the convenience script
./activate.sh
```

### 2. Install Dependencies
```bash
# Core dependencies (already installed)
pip install torch>=2.1 pyyaml>=6.0 transformers>=4.30.0 numpy>=1.24.0

# Optional development dependencies
pip install pytest black flake8 mypy

# Optional advanced features
pip install accelerate datasets wandb tensorboard
```

### 3. Run Examples
```bash
# Set Python path and activate environment
source setup_env.sh

# Test configuration system
python src/config/example.py

# Test tokenizer integration
python src/config/tokenizer_integration.py

# Note: Training modules are being implemented
# python -m src.train_layerwise --config config.yaml --dummy True --max_steps 50
```

**Note**: Always use `source setup_env.sh` first to set the Python path correctly.

**Status**: Configuration system is complete. Training modules are being implemented next.

## 🏗️ Architecture

### Key Features
- **Bidirectional Training**: Use bidirectional attention (BERT-style) for more efficient learning
- **Task-Specific Fine-tuning**: Switch to causal attention (GPT-style) for autoregressive tasks
- **Modern Attention**: Support for standard, GQA, MLA, and kernel-based attention
- **Layer-wise Training**: Train each layer independently with cached activations
- **Mask-Diffusion**: Variable masking rates (15% to 90%) for better representation learning

### Configuration
Edit `config.yaml` to customize:
- Model architecture (dimensions, layers, attention type)
- Training parameters (learning rate, batch size, steps)
- Attention modes (bidirectional vs causal)
- Fine-tuning modes (CLM, MLM, diffusion)

## 📁 Project Structure

```
src/
├── config/           # Configuration management
├── model/           # Model components (attention, normalization, etc.)
├── training/        # Training pipelines (layerwise, fusion, fine-tuning)
├── data/            # Data handling and preprocessing
├── utils/           # Utilities and helpers
└── scripts/         # Entry point scripts
```

## 🔧 Usage

### Layer-wise Training
```bash
python -m src.train_layerwise --config config.yaml --attention_mode bidirectional
```

### Fine-tuning
```bash
# Causal Language Modeling (autoregressive)
python -m src.fine_tune --config config.yaml --mode clm --checkpoint fused_model.pt

# Masked Language Modeling (bidirectional)
python -m src.fine_tune --config config.yaml --mode mlm --checkpoint fused_model.pt
```

## 🧪 Development

### Testing
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## 📚 Documentation

- [Configuration Guide](src/config/README.md)
- [Model Architecture](src/model/README.md)
- [Training Pipeline](src/training/README.md)

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
