#!/usr/bin/env python3
"""
TrainingMachine Example: Unified Time-Based Training

This example demonstrates the new TrainingMachine which provides unified
time-based orchestration across data, model, and optimizer planes.

Key concepts:
- Normalized time t ∈ [0,1] governing training evolution
- Phase schedule defining training phases
- Three planes: Data, Model, Optimizer
- Automatic assembly of (Data_t, Model_t, Optimizer_t) at each step
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from training import TrainingMachine, Phase, PhaseSchedule
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data(batch_size=4, seq_len=128, vocab_size=1000, num_samples=100):
    """Create dummy dataset for training."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def example_1_single_phase():
    """Example 1: Single-phase training (no curriculum)."""
    logger.info("\n" + "="*60)
    logger.info("Example 1: Single-Phase Training")
    logger.info("="*60)
    
    # Create config
    config = {
        'model': {
            'vocab_size': 1000,
            'd_model': 256,
            'n_heads': 8,
            'n_kv_heads': 4,
            'd_ff': 1024,
            'architecture': {
                'n_stacks': 2,
                'blocks_per_stack': 2
            }
        },
        'training': {
            'batch_size': 4,
            'seq_len': 128,
            'max_steps': 100,
            'optimizer': {
                'optimizer_type': 'AdamW',
                'lr': 1e-4
            }
        },
        'data': {
            'use_dummy_data': True,
            'num_samples': 100
        },
        'task': 'mlm'
    }
    
    # Create training machine with single phase
    machine = TrainingMachine(config, total_steps=100)
    
    logger.info(f"Number of phases: {len(machine.schedule.phases)}")
    logger.info(f"Phase name: {machine.schedule.phases[0].name}")
    
    # Run training for a few steps
    for step in range(5):
        state = machine.get_current_state()
        logger.info(
            f"Step {step}: t={state['t']:.4f}, "
            f"phase={state['phase'].name}"
        )
        
        # Perform training step
        metrics = machine.train_step()
        logger.info(f"  Loss: {metrics['loss']:.4f}")
    
    logger.info("\n✅ Single-phase training example completed\n")


def example_2_two_phase_curriculum():
    """Example 2: Two-phase curriculum with different configs."""
    logger.info("\n" + "="*60)
    logger.info("Example 2: Two-Phase Curriculum")
    logger.info("="*60)
    
    # Base config
    config = {
        'model': {
            'vocab_size': 1000,
            'd_model': 256,
            'n_heads': 8,
            'n_kv_heads': 4,
            'd_ff': 1024,
            'architecture': {
                'n_stacks': 2,
                'blocks_per_stack': 2
            }
        },
        'training': {
            'batch_size': 4,
            'seq_len': 128,
            'max_steps': 100,
            'optimizer': {
                'optimizer_type': 'AdamW',
                'lr': 1e-4
            }
        },
        'data': {
            'use_dummy_data': True,
            'num_samples': 100
        },
        'schedule': [
            {
                'start_t': 0.0,
                'end_t': 0.5,
                'name': 'phase_1_warmup',
                'task': 'mlm',
                'optimizer_config': {
                    'lr': 1e-5  # Lower LR for warmup
                }
            },
            {
                'start_t': 0.5,
                'end_t': 1.0,
                'name': 'phase_2_main',
                'task': 'mlm',
                'optimizer_config': {
                    'lr': 1e-4  # Higher LR for main training
                }
            }
        ]
    }
    
    # Create training machine with curriculum
    machine = TrainingMachine(config, total_steps=100)
    
    logger.info(f"Number of phases: {len(machine.schedule.phases)}")
    for i, phase in enumerate(machine.schedule.phases):
        logger.info(
            f"Phase {i}: {phase.name} "
            f"[{phase.start_t:.2f}, {phase.end_t:.2f}), "
            f"LR={phase.optimizer_config.get('lr', 'default')}"
        )
    
    # Run training and track phase transitions
    phase_transitions = []
    last_phase = None
    
    for step in range(50):
        state = machine.get_current_state()
        current_phase = state['phase'].name
        
        if last_phase != current_phase:
            phase_transitions.append((step, current_phase))
            last_phase = current_phase
            logger.info(
                f"\n🔄 Phase transition at step {step}: {current_phase}"
            )
        
        metrics = machine.train_step()
        
        if step % 20 == 0:
            logger.info(
                f"Step {step}: t={state['t']:.4f}, "
                f"phase={state['phase'].name}, "
                f"loss={metrics['loss']:.4f}"
            )
    
    logger.info(f"\nPhase transitions observed: {phase_transitions}")
    logger.info("\n✅ Two-phase curriculum example completed\n")


def example_3_phase_transitions():
    """Example 3: Track phase transitions and verify behavior."""
    logger.info("\n" + "="*60)
    logger.info("Example 3: Phase Transition Tracking")
    logger.info("="*60)
    
    config = {
        'model': {
            'vocab_size': 1000,
            'd_model': 256,
            'n_heads': 8,
            'n_kv_heads': 4,
            'd_ff': 1024,
            'architecture': {
                'n_stacks': 2,
                'blocks_per_stack': 2
            }
        },
        'training': {
            'batch_size': 4,
            'seq_len': 128,
            'max_steps': 200,
            'optimizer': {
                'optimizer_type': 'AdamW',
                'lr': 1e-4
            }
        },
        'data': {
            'use_dummy_data': True,
            'num_samples': 100
        },
        'schedule': [
            {
                'start_t': 0.0,
                'end_t': 0.25,
                'name': 'early',
                'task': 'mlm'
            },
            {
                'start_t': 0.25,
                'end_t': 0.75,
                'name': 'middle',
                'task': 'mlm'
            },
            {
                'start_t': 0.75,
                'end_t': 1.0,
                'name': 'late',
                'task': 'mlm'
            }
        ]
    }
    
    machine = TrainingMachine(config, total_steps=200)
    
    # Check phase transitions
    transitions = machine.schedule.get_phase_transitions()
    logger.info(f"Phase transitions: {transitions}")
    
    # Verify phase lookup
    for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
        phase = machine.schedule.at(t)
        logger.info(f"t={t:.2f} -> phase={phase.name}")
    
    logger.info("\n✅ Phase transition tracking example completed\n")


def example_4_time_controller():
    """Example 4: Time controller usage."""
    logger.info("\n" + "="*60)
    logger.info("Example 4: Time Controller")
    logger.info("="*60)
    
    from training import TimeController
    
    controller = TimeController(total_steps=1000)
    
    # Check conversions
    test_steps = [0, 100, 500, 999, 1000]
    
    logger.info("Step -> Normalized Time:")
    for step in test_steps:
        t = controller.normalized(step)
        logger.info(f"  step {step} -> t={t:.4f}")
    
    logger.info("\nNormalized Time -> Step:")
    test_times = [0.0, 0.1, 0.5, 0.9, 1.0]
    for t in test_times:
        step = controller.step_from_normalized(t)
        logger.info(f"  t={t:.4f} -> step {step}")
    
    # Test advancement
    controller.set_step(0)
    logger.info(f"\nStarting step: {controller.current_step}")
    
    for i in range(5):
        t = controller.get_current_time()
        logger.info(f"  Step {controller.current_step}: t={t:.4f}")
        controller.advance(200)
    
    logger.info("\n✅ Time controller example completed\n")


if __name__ == "__main__":
    # Run examples
    example_1_single_phase()
    example_2_two_phase_curriculum()
    example_3_phase_transitions()
    example_4_time_controller()
    
    logger.info("\n" + "="*60)
    logger.info("All examples completed successfully!")
    logger.info("="*60 + "\n")

