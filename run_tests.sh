#!/bin/bash
# Script to run StackWise tests

echo "🧪 Running StackWise Tests"
echo "========================="

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "📦 Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
    echo "💡 Consider creating a virtual environment:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
fi

# Check if pytest is available
echo "🔍 Checking pytest availability..."
python -c "import pytest; print(f'✅ pytest version: {pytest.__version__}')" 2>/dev/null || {
    echo "❌ pytest not found in virtual environment"
    echo "💡 Installing pytest..."
    pip install pytest --no-cache-dir
}

echo ""
echo "🚀 Running all tests..."
echo "======================"

# Run all tests with verbose output
echo "🔍 Running unit tests..."
python -m pytest tests/unit/ -v

echo ""
echo "🔍 Running integration tests..."
python -m pytest tests/integration/ -v

echo ""
echo "🔍 Running attention tests..."
python -m pytest src/model/attention/test/ -v

echo ""
echo "🔍 Running example tests..."
python -m pytest tests/examples/ -v

echo ""
echo "🔍 Running checkpointing tests..."
python examples/simple_checkpointing_test.py

echo ""
echo "📊 Test Summary"
echo "==============="
echo "✅ All tests completed!"
echo ""
echo "💡 Additional test commands:"
echo "   # Run specific test file:"
echo "   python -m pytest tests/unit/test_config_validation.py -v"
echo ""
echo "   # Run tests with coverage:"
echo "   python -m pytest tests/ --cov=src --cov-report=html"
echo ""
echo "   # Run tests in parallel:"
echo "   python -m pytest tests/ -n auto"
echo ""
echo "   # Run only failed tests:"
echo "   python -m pytest tests/ --lf"
echo ""
echo "   # Run tests with detailed output:"
echo "   python -m pytest tests/ -v -s"
echo ""
echo "   # Run checkpointing tests only:"
echo "   python examples/simple_checkpointing_test.py"
echo ""
echo "   # Run progressive training example:"
echo "   python examples/progressive_training_system_example.py"
