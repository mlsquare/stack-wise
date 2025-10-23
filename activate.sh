#!/bin/bash
# StackWise Virtual Environment Activation Script

echo "🚀 Activating StackWise virtual environment..."
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Installed packages:"
echo "   - PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "   - Transformers $(python -c 'import transformers; print(transformers.__version__)')"
echo "   - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "🔧 Available commands:"
echo "   - stackwise-train    # Start layer-wise training"
echo "   - stackwise-finetune # Fine-tune model"
echo "   - stackwise-fuse     # Fuse trained layers"
echo ""
echo "📚 Configuration:"
echo "   - Edit config.yaml to modify settings"
echo "   - Run 'python src/config/example.py' for examples"
echo ""
echo "Ready to go! 🎉"
