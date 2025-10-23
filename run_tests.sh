#!/bin/bash
# Script to run StackWise tests

echo "🧪 Running StackWise Tests"
echo "========================="

# Navigate to project directory
cd /workspace/stack-wise

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

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
python -m pytest tests/ src/model/attention/test/ -v

echo ""
echo "📊 Test Summary"
echo "==============="
echo "✅ All tests completed!"
echo ""
echo "💡 Additional test commands:"
echo "   # Run specific test file:"
echo "   python -m pytest tests/test_config_validation.py -v"
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
