#!/usr/bin/env python3
"""
Curriculum Learning: Right-to-Left Masking with Progressive QLoRA

This example demonstrates using the unified framework components to implement:
1. Right-to-left training (prepend stacks)
2. Curriculum with increasing masking noise: 5% → 10% → 15%
3. Frozen trunk with QLoRA adapters added to new stacks
4. Three-phase progressive training

Uses the unified framework's Phase, PhaseSchedule, and TimeController
to coordinate data, model, and optimizer evolution through time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging

from training import Phase, PhaseSchedule, TimeController, ProgressiveRackBuilder, ProgressiveTrainer
from framework.specs import BatchSpec, RackSpec
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_masked_dataset(vocab_size: int, seq_len: int, num_samples: int, 
                         mask_rate: float, mask_token_id: int = 4):
    """
    Create a dataset with specified masking rate.
    
    Args:
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        num_samples: Number of samples
        mask_rate: Fraction of tokens to mask (0.0 to 1.0)
        mask_token_id: Token ID to use for masking
        
    Returns:
        DataLoader with masked data
    """
    # Generate random sequences
    input_ids = torch.randint(5, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    
    # Apply masking
    num_masked = int(seq_len * mask_rate)
    for i in range(num_samples):
        # Randomly select positions to mask
        mask_positions = torch.randperm(seq_len)[:num_masked]
        input_ids[i, mask_positions] = mask_token_id
    
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=4, shuffle=True)


def create_three_phase_schedule(total_steps: int):
    """
    Create a three-phase schedule for curriculum learning.
    
    Returns:
        PhaseSchedule covering [0,1] with three phases
    """
    phases = [
        Phase(
            start_t=0.0,
            end_t=1/3,
            name='phase_1_easy',
            task='mlm',
            data_config={'mask_rate': 0.05},  # 5% masking
            optimizer_config={'lr': 1e-4}
        ),
        Phase(
            start_t=1/3,
            end_t=2/3,
            name='phase_2_medium',
            task='mlm',
            data_config={'mask_rate': 0.10},  # 10% masking
            optimizer_config={'lr': 8e-5}
        ),
        Phase(
            start_t=2/3,
            end_t=1.0,
            name='phase_3_hard',
            task='mlm',
            data_config={'mask_rate': 0.15},  # 15% masking
            optimizer_config={'lr': 5e-5}
        )
    ]
    
    return PhaseSchedule(phases)


def demonstrate_curriculum_learning():
    """
    Demonstrate curriculum learning using the unified framework components.
    """
    logger.info("\n" + "="*70)
    logger.info("Curriculum Learning: Right-to-Left with Progressive QLoRA")
    logger.info("Using: Unified Framework (Phase, Schedule, TimeController)")
    logger.info("="*70)
    
    vocab_size = 1000
    d_model = 256
    total_steps = 300
    
    logger.info("\n📋 Curriculum Overview:")
    logger.info("  Phase 1 (0.00-0.33):  5% masking, 1 stack  (rightmost)")
    logger.info("  Phase 2 (0.33-0.67): 10% masking, 2 stacks (prepend)")
    logger.info("  Phase 3 (0.67-1.00): 15% masking, 3 stacks (prepend)")
    logger.info("  Strategy: Frozen trunk + QLoRA on new stacks")
    
    # Create StackWise configuration
    config = StackWiseConfig()
    config.model.vocab_size = vocab_size
    config.model.d_model = d_model
    config.model.n_heads = 8
    config.model.n_kv_heads = 4
    config.model.d_ff = d_model * 4
    
    # Enable progressive training with QLoRA
    config.training.progressive.enabled = True
    config.training.progressive.max_stacks = 3
    config.training.progressive.trunk_strategy = 'frozen'
    config.training.progressive.qlora_enabled = True
    config.training.progressive.qlora_rank = 16
    config.training.progressive.qlora_alpha = 32
    
    # Create progressive rack builder (prepend mode = right-to-left)
    logger.info("\n🏗️  Initializing Progressive Rack Builder (prepend mode)...")
    rack_builder = ProgressiveRackBuilder(
        config=config,
        building_mode="prepend",
        default_precision="full"
    )
    rack_builder.initialize_embeddings_and_head()
    
    # Create phase schedule using unified framework
    logger.info("\n📅 Creating Phase Schedule...")
    schedule = create_three_phase_schedule(total_steps)
    logger.info(f"✅ Schedule created with {len(schedule.phases)} phases")
    
    # Create time controller using unified framework
    logger.info("\n⏱️  Creating Time Controller...")
    time_controller = TimeController(total_steps=total_steps, start_step=0)
    logger.info(f"✅ Time controller created: {time_controller.total_steps} steps")
    
    # Phase 1: Build first stack (rightmost) with 5% masking
    logger.info("\n" + "-"*70)
    logger.info("📍 Phase 1: Easy Task (5% masking)")
    logger.info("-"*70)
    
    phase = schedule.phases[0]
    t_start = time_controller.normalized(0)
    t_end = time_controller.normalized(int(total_steps * 1/3))
    
    logger.info(f"Phase: {phase.name} [{t_start:.2f}, {t_end:.2f})")
    logger.info(f"Mask rate: {phase.data_config['mask_rate']*100:.0f}%")
    
    logger.info("Adding first stack (rightmost)...")
    stack_1 = rack_builder.prepend_stack(n_blocks=2, precision="full")
    logger.info(f"✅ Stack 0 added with {len(stack_1.blocks)} blocks")
    logger.info(f"   Position: Rightmost (closest to output)")
    logger.info(f"   QLoRA: {'Yes' if 0 in rack_builder.qlora_adapters else 'No'}")
    
    # Get spec for framework compatibility
    spec = rack_builder.to_spec()
    logger.info(f"   Rack spec: {spec.n_stacks} stacks, d_model={spec.d_model}")
    
    # Create dataset for phase 1
    dataloader_1 = create_masked_dataset(
        vocab_size=vocab_size,
        seq_len=128,
        num_samples=200,
        mask_rate=phase.data_config['mask_rate']
    )
    
    # Get batch spec for framework compatibility
    batch_iter = iter(dataloader_1)
    batch = next(batch_iter)
    batch_spec = BatchSpec(
        batch_size=batch[0].shape[0],
        seq_len=batch[0].shape[1],
        vocab_size=vocab_size
    )
    logger.info(f"✅ Dataset created: {batch_spec.batch_size} batches")
    
    logger.info("🎯 Simulating training on easy task...")
    for i in range(3):
        time_controller.advance(1)
        t = time_controller.get_current_time()
        current_phase = schedule.at(t)
        logger.info(f"   Step {time_controller.current_step}: t={t:.3f}, phase={current_phase.name}")
    logger.info("✅ Phase 1 demonstration complete")
    
    # Phase 2: Prepend second stack with 10% masking
    logger.info("\n" + "-"*70)
    logger.info("📍 Phase 2: Medium Task (10% masking)")
    logger.info("-"*70)
    
    phase = schedule.phases[1]
    t_start = time_controller.normalized(time_controller.current_step)
    t_end = time_controller.normalized(int(total_steps * 2/3))
    
    logger.info(f"Phase: {phase.name} [{t_start:.2f}, {t_end:.2f})")
    logger.info(f"Mask rate: {phase.data_config['mask_rate']*100:.0f}% (increased!)")
    
    logger.info("Freezing trunk (stack 0)...")
    rack_builder.freeze_trunk([0])
    logger.info(f"✅ Stack 0 frozen")
    
    logger.info("Prepending second stack (to the left)...")
    stack_2 = rack_builder.prepend_stack(n_blocks=2, precision="full")
    logger.info(f"✅ Stack 0 prepended with {len(stack_2.blocks)} blocks")
    logger.info(f"   Position: Left of previous stack")
    logger.info(f"   QLoRA: {'Yes' if 0 in rack_builder.qlora_adapters else 'No'}")
    
    # Update spec
    spec = rack_builder.to_spec()
    logger.info(f"   Rack spec: {spec.n_stacks} stacks, d_model={spec.d_model}")
    
    # Create dataset for phase 2
    dataloader_2 = create_masked_dataset(
        vocab_size=vocab_size,
        seq_len=128,
        num_samples=200,
        mask_rate=phase.data_config['mask_rate']
    )
    
    logger.info("🎯 Simulating training on medium task...")
    for i in range(3):
        time_controller.advance(1)
        t = time_controller.get_current_time()
        current_phase = schedule.at(t)
        logger.info(f"   Step {time_controller.current_step}: t={t:.3f}, phase={current_phase.name}")
    logger.info("✅ Phase 2 demonstration complete")
    
    # Phase 3: Prepend third stack with 15% masking
    logger.info("\n" + "-"*70)
    logger.info("📍 Phase 3: Hard Task (15% masking)")
    logger.info("-"*70)
    
    phase = schedule.phases[2]
    t_start = time_controller.normalized(time_controller.current_step)
    t_end = time_controller.normalized(total_steps)
    
    logger.info(f"Phase: {phase.name} [{t_start:.2f}, {t_end:.2f})")
    logger.info(f"Mask rate: {phase.data_config['mask_rate']*100:.0f}% (increased again!)")
    
    logger.info("Freezing trunk (stacks 0-1)...")
    rack_builder.freeze_trunk([0, 1])
    logger.info(f"✅ Stacks 0-1 frozen")
    
    logger.info("Prepending third stack (leftmost)...")
    stack_3 = rack_builder.prepend_stack(n_blocks=2, precision="full")
    logger.info(f"✅ Stack 0 prepended with {len(stack_3.blocks)} blocks")
    logger.info(f"   Position: Leftmost (furthest from output)")
    logger.info(f"   QLoRA: {'Yes' if 0 in rack_builder.qlora_adapters else 'No'}")
    
    # Update spec
    spec = rack_builder.to_spec()
    logger.info(f"   Rack spec: {spec.n_stacks} stacks, d_model={spec.d_model}")
    
    # Create dataset for phase 3
    dataloader_3 = create_masked_dataset(
        vocab_size=vocab_size,
        seq_len=128,
        num_samples=200,
        mask_rate=phase.data_config['mask_rate']
    )
    
    logger.info("🎯 Simulating training on hard task...")
    for i in range(3):
        time_controller.advance(1)
        t = time_controller.get_current_time()
        current_phase = schedule.at(t)
        logger.info(f"   Step {time_controller.current_step}: t={t:.3f}, phase={current_phase.name}")
    logger.info("✅ Phase 3 demonstration complete")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("📊 Training Complete - Final Model Summary")
    logger.info("="*70)
    
    rack_info = rack_builder.get_rack_info()
    logger.info(f"\n🏗️  Architecture (via ProgressiveRackBuilder):")
    logger.info(f"  Total stacks: {rack_info['current_stacks']}")
    logger.info(f"  Building mode: {rack_info['building_mode']}")
    
    logger.info(f"\n📚 Stack Details (left to right):")
    for i in range(rack_info['current_stacks']):
        stack_info = rack_builder.get_stack_info(i)
        if stack_info:
            logger.info(f"  Stack {i}:")
            logger.info(f"    Blocks: {stack_info['num_blocks']}")
            logger.info(f"    Precision: {stack_info['precision']}")
            logger.info(f"    Trainable params: {stack_info['trainable_parameters']:,}")
            logger.info(f"    Frozen: {stack_info['trainable_parameters'] == 0}")
            logger.info(f"    QLoRA: {stack_info['qlora_enabled']}")
    
    # Use framework specs for final summary
    spec = rack_builder.to_spec()
    logger.info(f"\n🎓 Framework Spec Summary:")
    logger.info(f"  Rack spec: {spec.n_stacks} stacks")
    logger.info(f"  d_model: {spec.d_model}")
    logger.info(f"  Frozen stacks: {spec.frozen_stacks}")
    
    logger.info(f"\n⏱️  Time Controller Summary:")
    logger.info(f"  Total steps: {time_controller.total_steps}")
    logger.info(f"  Current step: {time_controller.current_step}")
    logger.info(f"  Progress: {time_controller.get_progress():.2%}")
    
    # Build final rack
    logger.info(f"\n🎯 Building final rack...")
    final_rack = rack_builder.build_rack()
    total_params = sum(p.numel() for p in final_rack.parameters())
    trainable_params = sum(p.numel() for p in final_rack.parameters() if p.requires_grad)
    
    logger.info(f"✅ Final rack built:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable fraction: {100*trainable_params/total_params:.2f}%")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Curriculum learning demonstration complete!")
    logger.info("="*70 + "\n")
    
    return rack_builder, final_rack, schedule, time_controller


def visualize_curriculum():
    """Visualize the curriculum structure."""
    logger.info("\n" + "="*70)
    logger.info("📊 Curriculum Visualization")
    logger.info("="*70)
    
    logger.info("""
    Time Progression (t ∈ [0,1]):
    
    0.0         0.33        0.67        1.0
    |-----------|-----------|-----------|
    
    Phase 1     Phase 2     Phase 3
    5% mask     10% mask    15% mask
    ↓           ↓           ↓
    
    Stack Growth (right-to-left / prepend):
    
    Phase 1:           [Stack 0]
                       (rightmost)
    
    Phase 2:      [Stack 0][Stack 1*]
                           (frozen)  (new)
    
    Phase 3: [Stack 0][Stack 1][Stack 2*]
             (new)    (frozen) (frozen)
             
    * = has QLoRA adapters
    
    Unified Framework Components Used:
    - Phase: Define phases with config overrides
    - PhaseSchedule: Coordinate phases across time
    - TimeController: Manage step ↔ time conversion
    - Specs: Track model/data state (RackSpec, BatchSpec)
    - ProgressiveRackBuilder: Build and manage stacks
    """)


if __name__ == "__main__":
    logger.info("\n🚀 Starting Curriculum Learning Example")
    logger.info("="*70 + "\n")
    
    # Show visualization
    visualize_curriculum()
    
    # Run demonstration
    rack_builder, final_rack, schedule, time_controller = demonstrate_curriculum_learning()
    
    logger.info("\n💡 Key Takeaways:")
    logger.info("  1. Uses Unified Framework: Phase, Schedule, TimeController")
    logger.info("  2. Curriculum learning: gradual difficulty (5%→10%→15%)")
    logger.info("  3. Right-to-left: stacks prepended (prepend mode)")
    logger.info("  4. Frozen trunk: previous stacks frozen after training")
    logger.info("  5. Progressive QLoRA: new stacks get QLoRA adapters")
    logger.info("  6. Specs used: RackSpec, BatchSpec for framework compatibility")
    
    logger.info("\n🎓 This demonstrates the unified framework's ability to:")
    logger.info("  - Coordinate through Phase objects with config overrides")
    logger.info("  - Manage time with TimeController (steps ↔ t ∈ [0,1])")
    logger.info("  - Track state with Specs (model, data compatibility)")
    logger.info("  - Separate data evolution (masking rate)")
    logger.info("  - Separate model evolution (prepend stacks)")
    logger.info("  - Separate optimizer evolution (freeze + QLoRA)")
    logger.info("  - All synchronized through normalized time")
    
    logger.info("\n✅ Example complete!\n")
