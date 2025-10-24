#!/bin/bash
# GPT-2 Fusion Training Example Runner

set -e  # Exit on any error

echo "🚀 Starting GPT-2 Fusion Training Example"

# Check if virtual environment exists
if [ ! -d "../../.venv" ]; then
    echo "❌ Virtual environment not found. Please run 'uv venv' first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../../.venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking requirements..."
python -c "import torch, transformers, datasets" || {
    echo "❌ Required packages not found. Installing..."
    pip install torch transformers datasets
}

# Setup the example
echo "📁 Setting up example..."
python setup.py

# Prepare data
echo "📊 Preparing data..."
python data_loader.py --prepare

# Test data loader
echo "🧪 Testing data loader..."
python data_loader.py --test_loader

# Run training (test mode)
echo "🎯 Running training (test mode)..."
python train_gpt2_fusion.py --test_only

# Run evaluation
echo "📊 Running evaluation..."
python evaluate_gpt2.py

echo "🎉 GPT-2 Fusion Training Example completed successfully!"
echo "📋 Check the logs and checkpoints directories for results."
