#!/bin/bash
# StackWise Unified Setup Script
# Handles virtual environment creation, activation, and dependency installation

echo "🚀 Setting up StackWise development environment..."
echo "==================================================="

# Navigate to project directory (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set Python path to repo root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Activate or create virtual environment
echo ""
echo "📦 Setting up virtual environment..."

if [ -f ".venv/bin/activate" ]; then
	echo "✅ Found existing virtual environment at .venv"
	source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
	echo "✅ Found existing virtual environment at venv"
	source venv/bin/activate
else
	echo "⚠️  No virtual environment found. Creating one..."
	python -m venv .venv
	source .venv/bin/activate
fi

# Display Python version
echo ""
echo "🐍 Python version: $(python --version)"

# Check and install dependencies
echo ""
echo "🔍 Checking dependencies..."

check_dependency() {
	local name=$1
	local check_cmd=$2
	if python -c "$check_cmd" &>/dev/null; then
		python -c "$check_cmd"
		return 0
	else
		echo "❌ $name not found"
		return 1
	fi
}

NEEDS_INSTALL=false

check_dependency "PyTorch" "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || NEEDS_INSTALL=true
check_dependency "Transformers" "import transformers; print(f'✅ Transformers {transformers.__version__}')" || NEEDS_INSTALL=true
check_dependency "NumPy" "import numpy; print(f'✅ NumPy {numpy.__version__}')" || NEEDS_INSTALL=true
check_dependency "PyYAML" "import yaml; print('✅ PyYAML available')" || NEEDS_INSTALL=true
check_dependency "Pytest" "import pytest; print(f'✅ pytest {pytest.__version__}')" || NEEDS_INSTALL=true

# Install dependencies if needed
if [ "$NEEDS_INSTALL" = true ]; then
	echo ""
	echo "💡 Installing packages from requirements.txt (this may take a while)..."
	if [ -f "requirements.txt" ]; then
		pip install -r requirements.txt
		echo "✅ Dependencies installed!"
	else
		echo "❌ requirements.txt not found in repo root."
		echo "💡 Install packages manually:"
		echo "   pip install torch torchvision transformers numpy pyyaml pytest"
	fi
fi

# Final status
echo ""
echo "==================================================="
echo "✅ Environment ready!"
echo "📦 Python path: $PYTHONPATH"

# Show installed packages (first 10)
echo ""
echo "📋 Installed packages (showing first 10):"
pip list | head -10

echo ""
echo "🔧 Available commands:"
echo "   python src/config/example.py                      # Test configuration"
echo "   python src/config/tokenizer_integration.py         # Test tokenizer integration"
echo "   ./run_tests.sh                                     # Run all tests"
echo "   python -m pytest tests/ -v                        # Run tests with pytest"
echo ""
echo "Ready to develop! 🎉"
