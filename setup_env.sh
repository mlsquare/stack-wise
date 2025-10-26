#!/bin/bash
# StackWise Environment Setup Script

echo "🚀 Setting up StackWise development environment..."

# Set Python path to repo root (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
else
	echo "⚠️  Virtualenv .venv not found. Run ./setup_venv.sh to create and install dependencies."
fi

echo "✅ Environment ready!"
echo "📦 Python path set to: $PYTHONPATH"
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🤗 Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"

echo ""
echo "🔧 Available commands:"
echo "   python src/config/example.py                    # Test configuration"
echo "   python src/config/tokenizer_integration.py       # Test tokenizer integration"
echo "   python -m src.train_layerwise --help            # Training help"
echo ""
echo "Ready to develop! 🎉"
