# Contributing to StackWise

Thank you for your interest in contributing to StackWise! This document provides guidelines and information for contributors.

## 🎯 **Project Vision**

StackWise aims to **democratize large model training** by enabling 70B parameter models to be trained on a single H200 GPU through revolutionary layer-wise progressive training with bidirectional attention and mask-diffusion objectives.

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8+
- PyTorch 2.1+
- CUDA 12+ (for GPU training)
- Git

### **Development Setup**
```bash
# Clone the repository
git clone https://github.com/mlsquare/stack-wise.git
cd stack-wise

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install development dependencies
uv pip install -e .[advanced,dev]

# Run tests to verify setup
python tests/run_tests.py
```

## 📋 **Contribution Areas**

### **High Priority Areas**
1. **Core Architecture**
   - Block-Stack-Rack paradigm improvements
   - Layer-wise training optimizations
   - Memory efficiency enhancements

2. **Training Modes**
   - Stack-wise training implementation
   - Progressive curriculum learning
   - Left-to-right vs right-to-left strategies

3. **Unified Training Objectives**
   - MLM/CLM unification improvements
   - Mask-diffusion objective enhancements
   - Attention mechanism optimizations

4. **Memory Optimization**
   - Activation caching improvements
   - Quantization support (FP4, FP8, FP16)
   - QLoRA integration enhancements

### **Research Areas**
1. **Curriculum Learning**
   - New curriculum strategies
   - Semantic preservation techniques
   - Progressive model building

2. **Attention Mechanisms**
   - GQA, MLA, and kernel-based attention
   - Bidirectional vs causal attention
   - Modern attention variants

3. **Diffusion Training**
   - Time-as-depth training
   - Masked-diffusion objectives
   - Progressive masking schedules

## 🔧 **Development Workflow**

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/stack-wise.git
cd stack-wise

# Add upstream remote
git remote add upstream https://github.com/mlsquare/stack-wise.git
```

### **2. Create Feature Branch**
```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### **3. Make Changes**
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### **4. Test Your Changes**
```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --examples

# Run with verbose output
python tests/run_tests.py --verbose
```

### **5. Commit and Push**
```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add new curriculum learning strategy"

# Push to your fork
git push origin feature/your-feature-name
```

### **6. Create Pull Request**
- Go to your fork on GitHub
- Click "New Pull Request"
- Fill out the PR template
- Request review from maintainers

## 📝 **Coding Standards**

### **Python Code Style**
```bash
# Format code with black
black src/ examples/ tests/

# Lint with flake8
flake8 src/ examples/ tests/

# Type checking with mypy
mypy src/
```

### **Code Organization**
```
src/
├── config/           # Configuration management
├── model/            # Model components
├── training/         # Training pipelines
├── data/            # Data handling
└── utils/           # Utilities

tests/
├── unit/            # Unit tests
├── integration/     # Integration tests
└── examples/        # Example tests

docs/               # Documentation
examples/           # Usage examples
```

### **Documentation Standards**
- Use docstrings for all functions and classes
- Follow Google docstring format
- Include type hints
- Update README.md for significant changes
- Add examples for new features

### **Commit Message Format**
```
type: brief description

Detailed description of changes

- Bullet point 1
- Bullet point 2

Closes: #issue-number
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

## 🧪 **Testing Guidelines**

### **Test Structure**
```python
# Unit tests for individual components
def test_block_creation():
    """Test block creation with different configurations."""
    pass

# Integration tests for component interactions
def test_stack_training_integration():
    """Test stack training with progressive curriculum."""
    pass

# Example tests for end-to-end workflows
def test_gpt2_fusion_example():
    """Test GPT-2 fusion training example."""
    pass
```

### **Test Requirements**
- All new features must have tests
- Maintain >80% code coverage
- Include edge cases and error conditions
- Test both CPU and GPU paths when applicable

### **Running Tests**
```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python -m pytest tests/unit/test_model.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📚 **Documentation Guidelines**

### **Code Documentation**
```python
def train_stack_wise(
    model: StackWiseModel,
    dataloader: DataLoader,
    curriculum: str = "left_to_right"
) -> TrainingResults:
    """Train a stack using the specified curriculum.
    
    Args:
        model: The StackWise model to train
        dataloader: Data loader for training
        curriculum: Training curriculum ("left_to_right" or "right_to_left")
        
    Returns:
        TrainingResults containing loss, metrics, and checkpoints
        
    Raises:
        ValueError: If curriculum is not supported
        RuntimeError: If training fails
    """
    pass
```

### **README Updates**
- Update feature lists for new capabilities
- Add usage examples for new features
- Update performance characteristics
- Include configuration examples

### **Documentation Files**
- `docs/architecture.md`: Architecture concepts
- `docs/progressive_training.md`: Training strategies
- `docs/configuration_guide.md`: Configuration reference
- `docs/api_reference.md`: API documentation

## 🐛 **Bug Reports**

### **Before Reporting**
1. Check existing issues
2. Verify bug with latest version
3. Test with minimal reproduction case

### **Bug Report Template**
```markdown
**Bug Description**
Brief description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 2.1.0]
- CUDA: [e.g., 12.1]

**Additional Context**
Any other relevant information
```

## 💡 **Feature Requests**

### **Feature Request Template**
```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 🔍 **Code Review Process**

### **For Contributors**
- Address all review comments
- Update tests if needed
- Ensure CI passes
- Respond to feedback promptly

### **For Reviewers**
- Check code quality and style
- Verify tests are adequate
- Ensure documentation is updated
- Test the changes locally

### **Review Checklist**
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Memory usage is optimized

## 🚀 **Release Process**

### **Version Numbering**
- `MAJOR.MINOR.PATCH` (Semantic Versioning)
- `MAJOR`: Breaking changes
- `MINOR`: New features, backward compatible
- `PATCH`: Bug fixes, backward compatible

### **Release Checklist**
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version number is bumped
- [ ] Release notes are written

## 🤝 **Community Guidelines**

### **Code of Conduct**
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the golden rule

### **Communication**
- Use clear, descriptive commit messages
- Provide context in PR descriptions
- Ask questions when unsure
- Share knowledge and best practices

## 📞 **Getting Help**

### **Resources**
- **Documentation**: [docs/README.md](docs/README.md)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/mlsquare/stack-wise/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlsquare/stack-wise/discussions)

### **Contact**
- **Maintainers**: @mlsquare
- **Email**: [Add contact email if available]
- **Discord/Slack**: [Add community link if available]

## 🎯 **Project Roadmap**

### **Short Term (Next 3 months)**
- [ ] Complete stack-wise training implementation
- [ ] Add more curriculum learning strategies
- [ ] Improve memory optimization
- [ ] Add comprehensive benchmarks

### **Medium Term (3-6 months)**
- [ ] Support for 100B+ models
- [ ] Multi-GPU training support
- [ ] Advanced attention mechanisms
- [ ] Production deployment tools

### **Long Term (6+ months)**
- [ ] Distributed training across nodes
- [ ] Model compression techniques
- [ ] Hardware-specific optimizations
- [ ] Commercial deployment support

## 🙏 **Recognition**

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation
- Community acknowledgments

---

**Thank you for contributing to StackWise! Together, we can democratize large model training and make AI more accessible to everyone.** 🚀
