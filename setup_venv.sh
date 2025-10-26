#!/bin/bash
# Setup script for StackWise virtual environment

echo "🔧 Setting up StackWise virtual environment..."

# Navigate to project directory (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment (expect .venv in repo root)
echo "📦 Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
else
	echo "⚠️  Virtualenv not found at .venv — creating one now..."
	python -m venv .venv
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

# Check Python version
echo "🐍 Python version: $(python --version)"

# Check if torch is available
echo "🔥 Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ PyTorch not found in virtual environment"

# Check if transformers is available
echo "🤗 Checking Transformers installation..."
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" 2>/dev/null || echo "❌ Transformers not found in virtual environment"

# Check if numpy is available
echo "🔢 Checking NumPy installation..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy not found in virtual environment"

# Check if PyYAML is available
echo "📄 Checking PyYAML installation..."
python -c "import yaml; print('PyYAML available')" 2>/dev/null || echo "❌ PyYAML not found in virtual environment"

echo ""
echo "📋 Current pip list (first 10 packages):"
pip list | head -10

echo ""
echo "💡 Installing packages from requirements.txt (this may take a while)..."
if [ -f "requirements.txt" ]; then
	pip install -r requirements.txt
else
	echo "❌ requirements.txt not found in repo root. Install packages manually."
fi

echo ""
echo "💡 If disk space is an issue, try:"
echo "   pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "   pip install transformers numpy pyyaml"
