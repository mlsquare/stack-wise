# Stack-Wise Test Suite

This directory contains all tests for the Stack-Wise training system, organized by test type and functionality.

## 📁 Test Organization

### Unit Tests (`tests/unit/`)
Tests individual components in isolation:
- `test_attention_validation.py` - Attention mechanism validation
- `test_config_validation.py` - Configuration system validation
- `test_mask_handling.py` - Mask generation and handling

### Integration Tests (`tests/integration/`)
Tests interaction between multiple components:
- `test_direct_fusion.py` - Direct fusion trainer testing
- `test_fusion_direct.py` - Fusion trainer direct testing
- `test_fusion_trainer.py` - Complete fusion trainer integration

### Example Tests (`tests/examples/`)
Demonstration and example tests:
- `test_simple_fusion.py` - Simple fusion training example
- `example_fusion_training.py` - Complete training example

## 🚀 Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Types
```bash
# Unit tests only
python tests/run_tests.py --unit

# Integration tests only
python tests/run_tests.py --integration

# Example tests only
python tests/run_tests.py --examples
```

### Verbose Output
```bash
python tests/run_tests.py --verbose
```

### Run Individual Test Files
```bash
# Run a specific test file
python tests/unit/test_config_validation.py
python tests/integration/test_fusion_trainer.py
python tests/examples/test_simple_fusion.py
```

## 🧪 Test Categories

### Unit Tests
- **Component Isolation**: Test individual classes and functions
- **Fast Execution**: Quick feedback on component functionality
- **Mock Dependencies**: Use mocks to isolate components
- **Edge Cases**: Test boundary conditions and error handling

### Integration Tests
- **Component Interaction**: Test how components work together
- **Real Dependencies**: Use actual implementations
- **End-to-End**: Test complete workflows
- **Performance**: Monitor resource usage and timing

### Example Tests
- **Usage Patterns**: Demonstrate common usage scenarios
- **Documentation**: Serve as living documentation
- **Regression**: Prevent breaking changes to examples
- **User Experience**: Validate user-facing functionality

## 📋 Test Requirements

### Prerequisites
```bash
# Install test dependencies
pip install pytest torch transformers datasets numpy tqdm pyyaml

# Activate virtual environment
source .venv/bin/activate
```

### Environment Setup
```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use the setup script
source setup_env.sh
```

## 🔧 Test Configuration

### Test Data
- Tests use dummy data by default
- No external datasets required
- Configurable data size for performance testing

### Mock Objects
- Use mocks for external dependencies
- Isolate components for unit testing
- Simulate different scenarios

### Test Isolation
- Each test is independent
- No shared state between tests
- Clean setup and teardown

## 📊 Test Coverage

### Core Components
- ✅ Configuration system
- ✅ Model components (MLGKALayer, LexicalKernelManager, SwiGLUFFN)
- ✅ Training modules (FusionTrainer, BlockTrainer)
- ✅ Attention mechanisms
- ✅ Masking strategies

### Training Modes
- ✅ Layer-wise training
- ✅ Block-wise training
- ✅ Fusion training
- ✅ Quantization and QLoRA

### Advanced Features
- ✅ Time-step masking
- ✅ Memory management
- ✅ Disk backup system
- ✅ Multi-precision support

## 🐛 Debugging Tests

### Verbose Output
```bash
python tests/run_tests.py --verbose
```

### Individual Test Debugging
```bash
# Run with Python debugger
python -m pdb tests/unit/test_config_validation.py

# Run with detailed output
python -u tests/integration/test_fusion_trainer.py
```

### Test Logging
```bash
# Enable debug logging
PYTHONPATH=src python tests/run_tests.py --verbose
```

## 📈 Performance Testing

### Memory Usage
- Monitor GPU/CPU memory during tests
- Test with different batch sizes
- Validate memory cleanup

### Training Speed
- Measure training step time
- Compare different configurations
- Profile bottlenecks

### Quantization Testing
- Test different precision levels
- Validate conversion accuracy
- Check memory savings

## 🔄 Continuous Integration

### Automated Testing
```bash
# Run in CI environment
python tests/run_tests.py --unit --integration
```

### Test Reports
- Generate test reports
- Track test coverage
- Monitor test performance

## 📝 Writing New Tests

### Test Structure
```python
import unittest
from src.model.layers import MLGKALayer

class TestMLGKALayer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.layer = MLGKALayer(d_model=768, n_heads=12)
    
    def test_forward_pass(self):
        """Test forward pass functionality."""
        # Test implementation
        pass
```

### Best Practices
- Use descriptive test names
- Test one thing per test method
- Use appropriate assertions
- Clean up resources
- Mock external dependencies

## 🎯 Test Goals

### Reliability
- Tests should be deterministic
- No flaky tests
- Consistent results across runs

### Maintainability
- Easy to understand and modify
- Clear test structure
- Good documentation

### Performance
- Fast execution
- Efficient resource usage
- Scalable test suite

---

**Happy Testing! 🧪**
