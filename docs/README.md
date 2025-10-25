# Stack-Wise Documentation

Welcome to the Stack-Wise documentation! This directory contains comprehensive guides and references for using the Stack-Wise training system.

## 🆕 Recent Updates

### Dual-LoRA Implementation
- **Progressive QLoRA Training** - Advanced dual-LoRA approach with stack LoRA + progressive QLoRA
- **NVFP4 Precision Support** - Added NVIDIA FP4 precision for efficient training
- **Enhanced Configuration** - New progressive QLoRA parameters and strategies
- **Improved Examples** - Updated examples demonstrating dual-LoRA approach

### Key Features
- ✅ **Stack LoRA**: Always added to each stack as it's created
- ✅ **Progressive QLoRA**: Added to entire trunk when new stacks are added
- ✅ **Multiple Precision Options**: full, half, bfloat16, nvfp4, QLoRA
- ✅ **Flexible Training Strategies**: Frozen trunk vs QLoRA trunk training

## 📚 Documentation Index

### Core Documentation
- **[Progressive Training Guide](PROGRESSIVE_TRAINING.md)** - Complete guide to progressive training with dual-LoRA
- **[Trainer Module Documentation](TRAINER_MODULE.md)** - Comprehensive guide to all training modes and features
- **[Configuration Guide](CONFIGURATION_GUIDE.md)** - Complete configuration reference with examples
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation for all classes and methods

### Examples
- **[GPT-2 Fusion Example](../examples/gpt2_fusion/README.md)** - Complete working example with data preparation and training

## 🚀 Quick Start

1. **Read the [Configuration Guide](CONFIGURATION_GUIDE.md)** to understand how to configure your training
2. **Check the [GPT-2 Fusion Example](../examples/gpt2_fusion/README.md)** for a complete working example
3. **Refer to the [API Reference](API_REFERENCE.md)** for detailed class and method documentation
4. **Use the [Trainer Module Documentation](TRAINER_MODULE.md)** for advanced training strategies

## 📖 Documentation Structure

### For Beginners
Start with:
1. [Configuration Guide](CONFIGURATION_GUIDE.md) - Learn how to configure the system
2. [GPT-2 Fusion Example](../examples/gpt2_fusion/README.md) - Run a complete example
3. [API Reference](API_REFERENCE.md) - Understand the core classes

### For Advanced Users
- [Trainer Module Documentation](TRAINER_MODULE.md) - Deep dive into training modes
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Advanced configuration options

### For Developers
- [API Reference](API_REFERENCE.md) - Complete class and method documentation
- [Trainer Module Documentation](TRAINER_MODULE.md) - Implementation details
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Configuration system architecture

## 🔧 Key Features Covered

### Training Modes
- **Layer-wise Training**: Individual layer training with cached activations
- **Block-wise Training**: Group training for better gradient flow
- **Fusion Training**: Multi-block training with frozen/trainable backbones

### Advanced Features
- **Quantization**: FP4, FP8, FP16, FP32 precision support
- **QLoRA Adapters**: Low-rank adaptation for efficient fine-tuning
- **Time-step Masking**: Progressive masking across layers
- **Memory Management**: Persistent quantization and gradient clearing
- **Disk Backup**: Run ID-based organization with full-precision storage

### Model Architectures
- **GPT-2**: Small, Medium, and custom configurations
- **Attention Mechanisms**: Standard, GQA, MLA, and kernel-based attention
- **Embeddings**: Lexical kernel integration with pre-trained models

## 🎯 Common Use Cases

### Research
- Progressive training experiments
- Attention mechanism comparisons
- Mask-diffusion objectives
- Memory-efficient training

### Production
- Large model training with quantization
- Efficient fine-tuning with QLoRA
- Multi-block training strategies
- Disk-based checkpoint management

### Education
- Understanding transformer architectures
- Learning about attention mechanisms
- Exploring training strategies
- Hands-on examples

## 📞 Getting Help

1. **Check the [Configuration Guide](CONFIGURATION_GUIDE.md)** for configuration issues
2. **Run the [GPT-2 Fusion Example](../examples/gpt2_fusion/README.md)** to verify your setup
3. **Refer to the [API Reference](API_REFERENCE.md)** for specific class/method questions
4. **Read the [Trainer Module Documentation](TRAINER_MODULE.md)** for advanced training strategies

## 🔄 Documentation Updates

This documentation is actively maintained and updated with each release. Key areas of focus:

- **Configuration**: New parameters and validation rules
- **Training Modes**: Enhanced training strategies and optimizations
- **Examples**: Working examples with different model architectures
- **API**: Complete coverage of all classes and methods

## 📝 Contributing to Documentation

If you find issues or want to improve the documentation:

1. **Report Issues**: Use the issue tracker for documentation bugs
2. **Suggest Improvements**: Propose enhancements to existing documentation
3. **Add Examples**: Contribute new examples and use cases
4. **Update References**: Keep API references current with code changes

---

**Happy Training! 🚀**
