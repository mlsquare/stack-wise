#!/usr/bin/env python3
"""
Progressive Training Example

This example demonstrates progressive training with the new Block/Stack/Rack architecture:
- Time interpretation (time-as-input vs time-as-depth)
- Progressive building (append/prepend modes)
- Training strategies (frozen trunk vs QLoRA trunk)
- Activation caching and streaming

Usage:
    python examples/progressive_training_example.py
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.architecture import Block, Stack, Rack, create_block_spec, create_stack_from_spec
from training.progressive_trainer import ProgressiveTrainer
from training.progressive_rack_builder import ProgressiveRackBuilder, PrecisionManager
from training.progressive_dataloader import ProgressiveDataLoader
from training.strategies.masking.time_step_masking import TimeStepMasking
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataloader(batch_size: int = 4, seq_len: int = 16, vocab_size: int = 1000):
    """Create a dummy dataloader for testing"""
    # Create dummy data
    input_ids = torch.randint(0, vocab_size, (100, seq_len))  # 100 samples
    labels = input_ids.clone()  # Same as input for language modeling
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def example_1_time_as_depth_training():
    """Example 1: Time-as-depth progressive training"""
    logger.info("🧠 Example 1: Time-as-depth Progressive Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = StackWiseConfig.from_yaml("config.yaml")
        config.training.progressive.time_interpretation = "depth"
        config.training.progressive.trunk_strategy = "frozen"
        config.validate()
        
        # Create progressive rack builder
        rack_builder = ProgressiveRackBuilder(
            config=config,
            building_mode="append",
            default_precision="full"
        )
        
        # Create progressive trainer
        trainer = ProgressiveTrainer(config)
        
        # Create dataloader
        dataloader = create_dummy_dataloader()
        
        # Train progressively
        results = trainer.train_rack(
            rack_builder=rack_builder,
            dataloader=dataloader,
            target_stacks=3
        )
        
        # Display results
        logger.info("✅ Time-as-depth training completed")
        logger.info(f"Final rack info: {results['final_rack_info']}")
        
        for i, history in enumerate(results['training_history']):
            logger.info(f"Stack {i}: Loss = {history['results']['avg_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in time-as-depth training: {e}")
        raise


def example_2_time_as_input_training():
    """Example 2: Time-as-input progressive training"""
    logger.info("🧠 Example 2: Time-as-input Progressive Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = StackWiseConfig.from_yaml("config.yaml")
        config.training.progressive.time_interpretation = "input"
        config.training.progressive.trunk_strategy = "qlora"
        config.validate()
        
        # Create progressive rack builder
        rack_builder = ProgressiveRackBuilder(
            config=config,
            building_mode="append",
            default_precision="full"
        )
        
        # Create progressive trainer
        trainer = ProgressiveTrainer(config)
        
        # Create dataloader
        dataloader = create_dummy_dataloader()
        
        # Train progressively
        results = trainer.train_rack(
            rack_builder=rack_builder,
            dataloader=dataloader,
            target_stacks=3
        )
        
        # Display results
        logger.info("✅ Time-as-input training completed")
        logger.info(f"Final rack info: {results['final_rack_info']}")
        
        for i, history in enumerate(results['training_history']):
            logger.info(f"Stack {i}: Loss = {history['results']['avg_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in time-as-input training: {e}")
        raise


def example_3_prepend_mode_training():
    """Example 3: Prepend mode progressive training"""
    logger.info("🧠 Example 3: Prepend Mode Progressive Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = StackWiseConfig.from_yaml("config.yaml")
        config.training.progressive.time_interpretation = "depth"
        config.training.progressive.trunk_strategy = "frozen"
        config.validate()
        
        # Create progressive rack builder with prepend mode
        rack_builder = ProgressiveRackBuilder(
            config=config,
            building_mode="prepend",  # Use prepend mode
            default_precision="full"
        )
        
        # Create progressive trainer
        trainer = ProgressiveTrainer(config)
        
        # Create dataloader
        dataloader = create_dummy_dataloader()
        
        # Train progressively
        results = trainer.train_rack(
            rack_builder=rack_builder,
            dataloader=dataloader,
            target_stacks=3
        )
        
        # Display results
        logger.info("✅ Prepend mode training completed")
        logger.info(f"Final rack info: {results['final_rack_info']}")
        
        for i, history in enumerate(results['training_history']):
            logger.info(f"Stack {i}: Loss = {history['results']['avg_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in prepend mode training: {e}")
        raise


def example_4_qlora_trunk_training():
    """Example 4: QLoRA trunk training"""
    logger.info("🧠 Example 4: QLoRA Trunk Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = StackWiseConfig.from_yaml("config.yaml")
        config.training.progressive.time_interpretation = "depth"
        config.training.progressive.trunk_strategy = "qlora"  # Use QLoRA trunk
        config.training.progressive.cache_activations = True
        config.validate()
        
        # Create progressive rack builder
        rack_builder = ProgressiveRackBuilder(
            config=config,
            building_mode="append",
            default_precision="full"
        )
        
        # Create progressive trainer
        trainer = ProgressiveTrainer(config)
        
        # Create dataloader
        dataloader = create_dummy_dataloader()
        
        # Train progressively
        results = trainer.train_rack(
            rack_builder=rack_builder,
            dataloader=dataloader,
            target_stacks=3
        )
        
        # Display results
        logger.info("✅ QLoRA trunk training completed")
        logger.info(f"Final rack info: {results['final_rack_info']}")
        
        for i, history in enumerate(results['training_history']):
            logger.info(f"Stack {i}: Loss = {history['results']['avg_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in QLoRA trunk training: {e}")
        raise


def example_5_mixed_precision_training():
    """Example 5: Mixed precision training"""
    logger.info("🧠 Example 5: Mixed Precision Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = StackWiseConfig.from_yaml("config.yaml")
        config.training.progressive.time_interpretation = "depth"
        config.training.progressive.trunk_strategy = "frozen"
        config.training.progressive.new_stack_precision = "half"  # Use half precision
        config.validate()
        
        # Create progressive rack builder
        rack_builder = ProgressiveRackBuilder(
            config=config,
            building_mode="append",
            default_precision="half"  # Use half precision
        )
        
        # Create progressive trainer
        trainer = ProgressiveTrainer(config)
        
        # Create dataloader
        dataloader = create_dummy_dataloader()
        
        # Train progressively
        results = trainer.train_rack(
            rack_builder=rack_builder,
            dataloader=dataloader,
            target_stacks=3
        )
        
        # Display results
        logger.info("✅ Mixed precision training completed")
        logger.info(f"Final rack info: {results['final_rack_info']}")
        
        for i, history in enumerate(results['training_history']):
            logger.info(f"Stack {i}: Loss = {history['results']['avg_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in mixed precision training: {e}")
        raise


def main():
    """Run all progressive training examples"""
    logger.info("🚀 StackWise Progressive Training Examples")
    logger.info("=" * 80)
    
    try:
        # Example 1: Time-as-depth training
        example_1_time_as_depth_training()
        logger.info("")
        
        # Example 2: Time-as-input training
        example_2_time_as_input_training()
        logger.info("")
        
        # Example 3: Prepend mode training
        example_3_prepend_mode_training()
        logger.info("")
        
        # Example 4: QLoRA trunk training
        example_4_qlora_trunk_training()
        logger.info("")
        
        # Example 5: Mixed precision training
        example_5_mixed_precision_training()
        logger.info("")
        
        logger.info("🎉 All progressive training examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running progressive training examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
