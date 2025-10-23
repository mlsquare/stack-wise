#!/bin/bash
# Setup script for StackWise virtual environment

echo "🔧 Setting up StackWise virtual environment..."

# Navigate to project directory
cd /workspace/stack-wise

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

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
echo "💡 If packages are missing, try:"
echo "   pip install torch pyyaml transformers numpy --no-cache-dir"
echo ""
echo "💡 If disk space is an issue, try:"
echo "   pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "   pip install transformers numpy pyyaml"
