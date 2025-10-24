#!/usr/bin/env python3
"""
GPT-2 Model Evaluation Script

This script evaluates the trained GPT-2 model and generates sample text.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "gpt2.yaml"):
    """Load training configuration."""
    logger.info(f"📋 Loading configuration from {config_path}")
    
    from config.base import StackWiseConfig
    config = StackWiseConfig.from_yaml(config_path)
    config.validate()
    
    logger.info(f"✅ Configuration loaded: run_id={config.training.run_id}")
    return config

def load_model_from_checkpoint(checkpoint_path: str, config):
    """Load model from checkpoint."""
    logger.info(f"📦 Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    logger.info(f"✅ Checkpoint loaded with keys: {list(checkpoint.keys())}")
    
    return checkpoint

def evaluate_model_perplexity(model, val_loader, device):
    """Evaluate model perplexity on validation set."""
    logger.info("📊 Evaluating model perplexity...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Calculate perplexity
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"✅ Perplexity: {perplexity:.2f}")
    return perplexity

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, device: str = "cpu"):
    """Generate text from model."""
    logger.info(f"🎯 Generating text from prompt: '{prompt}'")
    
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    logger.info(f"✅ Generated text: {generated_text}")
    return generated_text

def evaluate_disk_backup_system(config):
    """Evaluate disk backup system."""
    logger.info("💾 Evaluating disk backup system...")
    
    # Import FusionTrainer directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fusion_trainer", 
        str(Path(__file__).parent.parent.parent / "src" / "training" / "core" / "fusion_trainer.py")
    )
    fusion_trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion_trainer_module)
    FusionTrainer = fusion_trainer_module.FusionTrainer
    
    # Initialize FusionTrainer
    fusion_trainer = FusionTrainer(
        config=config,
        masking_strategy=None,
        quantization_manager=None,
        cache_manager=None,
        lexical_kernel_manager=None
    )
    
    # Test restoration
    run_id = config.training.run_id
    restored_blocks = fusion_trainer._restore_full_precision_weights_from_disk(
        run_id, "fp16", list(range(config.training.total_blocks))
    )
    
    if restored_blocks:
        logger.info(f"✅ Restored {len(restored_blocks)} blocks from disk")
        
        # Test model reconstruction
        reconstructed = fusion_trainer._reconstruct_model_from_disk(run_id, "fp16")
        logger.info(f"✅ Reconstructed {len(reconstructed)} blocks from disk")
        
        return True
    else:
        logger.warning("❌ No blocks restored from disk")
        return False

def evaluate_training_metrics(config):
    """Evaluate training metrics."""
    logger.info("📈 Evaluating training metrics...")
    
    # Check if checkpoints exist
    checkpoint_dir = config.training.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # List checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Evaluate each checkpoint
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        logger.info(f"Evaluating checkpoint: {checkpoint}")
        
        # Load checkpoint
        checkpoint_data = load_model_from_checkpoint(checkpoint_path, config)
        if checkpoint_data:
            logger.info(f"  Checkpoint size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")

def run_comprehensive_evaluation(config):
    """Run comprehensive evaluation."""
    logger.info("🔍 Running comprehensive evaluation...")
    
    # 1. Disk backup system evaluation
    logger.info("1. Evaluating disk backup system...")
    backup_success = evaluate_disk_backup_system(config)
    
    # 2. Training metrics evaluation
    logger.info("2. Evaluating training metrics...")
    evaluate_training_metrics(config)
    
    # 3. Configuration validation
    logger.info("3. Validating configuration...")
    config.validate()
    logger.info("✅ Configuration validation passed")
    
    # 4. Summary
    logger.info("📊 Evaluation Summary:")
    logger.info(f"  Run ID: {config.training.run_id}")
    logger.info(f"  Total Blocks: {config.training.total_blocks}")
    logger.info(f"  Disk Backup: {'✅ PASSED' if backup_success else '❌ FAILED'}")
    logger.info(f"  Configuration: ✅ PASSED")
    
    return backup_success

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="GPT-2 Model Evaluation")
    parser.add_argument("--config", default="gpt2.yaml", help="Configuration file")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--generate", action="store_true", help="Generate sample text")
    parser.add_argument("--prompt", default="The future of artificial intelligence", help="Text generation prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting GPT-2 Model Evaluation")
        
        # Load configuration
        config = load_config(args.config)
        
        # Run comprehensive evaluation
        evaluation_success = run_comprehensive_evaluation(config)
        
        if args.generate:
            logger.info("🎯 Text generation not implemented in this example")
            logger.info("This would require a fully trained model with proper forward pass")
        
        if evaluation_success:
            logger.info("🎉 Evaluation completed successfully!")
        else:
            logger.warning("⚠️ Some evaluation tests failed")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
