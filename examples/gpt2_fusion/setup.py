#!/usr/bin/env python3
"""
Setup script for GPT-2 Fusion Training Example

This script sets up the environment and prepares data for training.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed."""
    logger.info("🔍 Checking requirements...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'numpy',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is not installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All requirements satisfied")
    return True

def setup_directories():
    """Setup required directories."""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "data",
        "checkpoints",
        "logs",
        "checkpoints/gpt2_fusion"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def prepare_data():
    """Prepare training data."""
    logger.info("📊 Preparing training data...")
    
    try:
        # Run data preparation
        result = subprocess.run([
            sys.executable, "data_loader.py", "--prepare"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Data preparation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data preparation failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def test_imports():
    """Test if all imports work correctly."""
    logger.info("🧪 Testing imports...")
    
    try:
        # Test data loader
        from data_loader import GPT2Dataset, get_tokenizer
        logger.info("✅ data_loader imports successful")
        
        # Test configuration
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from config.base import StackWiseConfig
        logger.info("✅ config imports successful")
        
        # Test FusionTrainer (direct import)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fusion_trainer", 
            str(Path(__file__).parent.parent.parent / "src" / "training" / "core" / "fusion_trainer.py")
        )
        fusion_trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fusion_trainer_module)
        logger.info("✅ fusion_trainer imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def run_quick_test():
    """Run a quick test of the training pipeline."""
    logger.info("🚀 Running quick test...")
    
    try:
        # Test data loader
        result = subprocess.run([
            sys.executable, "data_loader.py", "--test_loader"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Data loader test passed")
        
        # Test training script
        result = subprocess.run([
            sys.executable, "train_gpt2_fusion.py", "--test_only"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Training script test passed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Quick test failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Starting GPT-2 Fusion Training Setup")
    
    try:
        # Check requirements
        if not check_requirements():
            logger.error("❌ Requirements check failed")
            return False
        
        # Setup directories
        setup_directories()
        
        # Test imports
        if not test_imports():
            logger.error("❌ Import test failed")
            return False
        
        # Prepare data
        if not prepare_data():
            logger.error("❌ Data preparation failed")
            return False
        
        # Run quick test
        if not run_quick_test():
            logger.error("❌ Quick test failed")
            return False
        
        logger.info("🎉 Setup completed successfully!")
        logger.info("📋 Next steps:")
        logger.info("  1. Run training: python train_gpt2_fusion.py")
        logger.info("  2. Run evaluation: python evaluate_gpt2.py")
        logger.info("  3. Check logs in the logs/ directory")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
