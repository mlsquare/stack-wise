# Stack-Wise Trainer Module

## ⚠️ IMPORTANT WARNING

**BlockTrainer is NOT SUPPORTED and NOT TESTED.**
- This is legacy code that may not work correctly
- Use **FusionTrainer** for production training
- BlockTrainer is kept for reference only

## 🎉 RECENT UPDATES

**Phase 1-4 Complete:**
- ✅ **Disk Backup System**: Run ID-based file organization with validation
- ✅ **Configuration Support**: QLoRA, quantization, time-step masking parameters
- ✅ **Testing & Validation**: Comprehensive test suite with all tests passing
- ✅ **Memory Management**: Precision conversion with gradient clearing
- ✅ **Optimizer Setup**: Multi-learning rate parameter groups
- ✅ **GPT-2 Fusion Example**: Complete working example with data preparation
- ✅ **Documentation**: Updated README and comprehensive trainer documentation
- ✅ **Code Cleanup**: Removed legacy files, consolidated architecture
- ✅ **Git Integration**: Proper .gitignore for model checkpoints and weights

## 🚀 CURRENT STATUS

**✅ PRODUCTION READY**
- All core training modules implemented and tested
- GPT-2 fusion training example working
- Comprehensive documentation and examples
- Memory-efficient quantization and QLoRA support
- Disk backup system for model weights

**🎯 RECOMMENDED USAGE**
- **FusionTrainer**: Primary training module for production use
- **GPT-2 Example**: Complete working example in `examples/gpt2_fusion/`
- **Configuration**: Use `examples/gpt2_fusion/gpt2.yaml` as template

## Overview

The Stack-Wise Trainer Module implements a sophisticated layer-wise training system with mask-diffusion objectives and hybrid caching. It supports three distinct training modes: **layer-wise**, **block-wise**, and **fused training**, each optimized for different training scenarios and performance requirements.

## Core Goals

### 1. Progressive Mask-Diffusion Training
- **Variable masking rates** (15%-90%) over token positions
- **Progressive increase** from encoder-like to diffusion-based decoder behavior
- **Layer-specific masking** with configurable scheduling strategies
- **Bidirectional attention** during training for efficient representation learning

### 2. Efficient Activation Caching
- **Hybrid caching strategy** with unique mask storage and activation IDs
- **Deduplication** of identical mask patterns across samples
- **Memory optimization** with O(1) lookup performance
- **Dual-mode caching** (layerwise and fusion modes)

### 3. Flexible Training Modes
- **Layer-wise**: Individual layer training with cached activations
- **Block-wise**: Group training (4 layers per block) for better gradient flow
- **Fused**: Progressive training with frozen/trainable block options

### 4. Advanced Quantization & Adapter Strategy
- **QLoRA Adapters**: Optional low-rank adapters per block for efficient fine-tuning
- **Quantized Loading**: NF FP8/FP16 quantization for memory-efficient model loading
- **Mixed Precision Training**: Full precision training with quantized backbone
- **Selective Updates**: QLoRA-only updates for frozen blocks, full updates for trainable blocks

### 5. Time-Step-Based Masking Strategy
- **Triplet-Based Masking**: (input_id, mask_pattern, time_t) for precise mask generation
- **Discrete Time Steps**: Time bins for efficient storage and retrieval
- **Progressive Masking**: Time-step-dependent masking fractions
- **Memory-Efficient Storage**: Avoid storing all time-step activations

## Architecture

### Architectural Insight: Unified Block-Based Design

**Key Insight**: `LayerwiseTrainer` is a special case of `BlockwiseTrainer` with `block_size = 1`. This suggests a unified architecture where:

- **LayerwiseTrainer** = `BlockwiseTrainer(block_size=1)`
- **BlockwiseTrainer** = `BlockwiseTrainer(block_size=4)` (default)
- **FusedTrainer** = `BlockwiseTrainer` with progressive fusion strategies

This unified approach would:
- **Eliminate code duplication** between LayerwiseTrainer and BlockwiseTrainer
- **Simplify the architecture** with a single, configurable trainer
- **Enable seamless transitions** between training modes
- **Reduce maintenance overhead** with shared logic

### Core Components

#### 1. MaskScheduler
```python
class MaskScheduler:
    """Base class for mask fraction scheduling strategies"""
    - Linear progression: min_fraction → max_fraction
    - Exponential progression: accelerated masking
    - Cosine progression: smooth transitions
```

#### 2. MaskDiffusionObjective
```python
class MaskDiffusionObjective:
    """Mask-diffusion objective with progressive masking"""
    - Variable masking rates (15%-90%)
    - Progressive increase with layer depth
    - Configurable mask token handling
```

#### 3. FixedMaskAssigner
```python
class FixedMaskAssigner:
    """Assigns fixed masks to samples for consistent training"""
    - Sample-specific mask caching
    - Batch-optimized mask processing
    - Deterministic mask assignment
```

#### 4. HashBasedActivationCache
```python
class HashBasedActivationCache:
    """Hybrid activation cache with unique mask storage"""
    - O(1) activation lookup by ID
    - Unique mask pattern storage
    - Memory-efficient deduplication
```

### Training Classes

#### 1. LayerwiseTrainer
**Purpose**: Individual layer training with cached activations
**Note**: This is a special case of BlockwiseTrainer with block_size = 1

**Key Features**:
- Progressive masking per layer
- Activation caching between layers
- Language model head integration
- Mask-diffusion objective optimization

**Training Flow**:
```
Layer 0: Raw Inputs → Masking → Layer 0 → Cache Activations
Layer 1: Cached Activations → Layer 1 → Update Cache
Layer 2: Cached Activations → Layer 2 → Update Cache
```

#### 2. BlockwiseTrainer
**Purpose**: Group training of multiple layers (configurable layers per block)
**Note**: LayerwiseTrainer is a special case with block_size = 1

**Key Features**:
- Block-level gradient flow
- Faster training with fewer phases
- Better layer interaction during training
- Reduced memory overhead
- Configurable block sizes (1, 4, 8, etc.)

**Training Flow**:
```
Block 0: Raw Inputs → Layers 0-3 → Cache Block Activations
Block 1: Block 0 Activations → Layers 4-7 → Cache Block Activations
Block 2: Block 1 Activations → Layers 8-11 → Cache Block Activations
```

#### 3. FusedTrainer
**Purpose**: Progressive training with frozen/trainable block options

**Key Features**:
- **Frozen Mode**: Freeze previous blocks, train current block
- **Trainable Mode**: End-to-end training of all blocks
- **Resampling Strategy**: Fresh inputs for each block
- **Boundary Activation Storage**: Store activations at block boundaries

**Training Modes**:

**Frozen Mode**:
```
[Frozen Block 0] → [Frozen Block 1] → [Trainable Block 2]
```

**Trainable Mode**:
```
[Trainable Block 0] → [Trainable Block 0-1] → [Trainable Block 0-2]
```

#### 4. Quantization & Adapter Strategy
**Purpose**: Memory-efficient training with optional QLoRA adapters

**Key Components**:
- **QLoRA Adapters**: Low-rank adaptation matrices per block
- **Quantized Loading**: NVFP4/FP8/FP16 model loading for memory efficiency
- **Mixed Precision**: Full precision training with quantized backbone
- **Selective Updates**: QLoRA-only for frozen blocks, full updates for trainable blocks

**Training Strategies**:

**Frozen Block Training**:
```
Quantized Backbone + QLoRA Adapters → Update Only QLoRA
```

**Trainable Block Training**:
```
Quantized Backbone + QLoRA Adapters → Update Both Backbone + QLoRA
```

**Activation Generation**:
```
Quantized Fused Model → Generate Activations → Store for Next Block
```

#### 5. Time-Step-Based Masking Strategy
**Purpose**: Efficient progressive masking with time-step-dependent mask generation

**Key Components**:
- **Triplet-Based Masking**: (input_id, mask_pattern, time_t) for precise mask generation
- **Discrete Time Steps**: Time bins for efficient storage and retrieval
- **Progressive Masking**: Time-step-dependent masking fractions
- **Memory-Efficient Storage**: Avoid storing all time-step activations

**Masking Strategies**:

**Time-Step Mask Generation**:
```
(input_id, mask_pattern, time_t) → Generate Mask for Specific Time Step
```

**Progressive Masking**:
```
Time t=0: Low masking (15%) → Time t=T: High masking (90%)
```

**Memory-Efficient Storage**:
```
Store only current time-step activations → Avoid storing all time steps
```

**Training Flow with Time Steps**:
```
Load Fused Backbone (Quantized) → Load QLoRA Adapters → Train Block (Full Precision) → Generate Time-Step Activations
```

#### 6. UnifiedTrainer
**Purpose**: Unified interface supporting all training modes

**Key Features**:
- Automatic trainer selection based on configuration
- Consistent API across all training modes
- Flexible mode switching
- Unified training information

## Configuration Options

### Training Modes
```yaml
training:
  mode: "layerwise" | "blockwise" | "fused"
  block_size: 4                    # Layers per block
  fusion_mode: "frozen" | "trainable"  # For fused mode
```

### Mask-Diffusion Settings
```yaml
training:
  min_mask_fraction: 0.15         # Early layers (15%)
  max_mask_fraction: 0.90         # Late layers (90%)
  mask_schedule_type: "linear"    # linear | exponential | cosine
  mask_token_id: 0
```

### Caching Configuration
```yaml
training:
  cache_mode: "layerwise" | "fusion"
  cache_dir: "./cache"
  fusion_evaluation: false
  save_fused_checkpoints: false
```

### Training Parameters
```yaml
training:
  epochs_per_layer: 1
  learning_rate: 1.0e-4
  batch_size: 4
  seq_len: 512
```

### Quantization & Adapter Configuration
```yaml
training:
  # QLoRA Configuration
  qlora_enabled: true
  qlora_rank: 16
  qlora_alpha: 32
  qlora_dropout: 0.1
  
  # Quantization Configuration
  quantization_enabled: true
  quantization_type: "nf_fp8"  # nf_fp8 | fp16 | fp32
  load_quantized: true
  
  # Mixed Precision Training
  mixed_precision: true
  backbone_quantized: true
  adapters_full_precision: true
```

### Time-Step-Based Masking Configuration
```yaml
training:
  # Time-Step Masking
  time_step_masking: true
  num_time_steps: 10
  time_step_bins: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  
  # Progressive Masking by Time Step
  time_step_mask_fractions:
    0: 0.15  # Early time steps: low masking
    5: 0.50  # Middle time steps: medium masking
    9: 0.90  # Late time steps: high masking
  
  # Memory Management
  store_all_time_steps: false  # Only store current time step
  time_step_cache_size: 1      # Number of time steps to cache
```

## Usage Examples

### Layer-wise Training
```python
from src.training.layerwise_trainer import UnifiedTrainer

# Initialize trainer
trainer = UnifiedTrainer(config)

# Train all layers
trainer.train_all_layers(dataloader, model_layers)
```

### Block-wise Training
```python
# Configure for block-wise training
config.training.mode = "blockwise"
config.training.block_size = 4

trainer = UnifiedTrainer(config)
trainer.train_all_layers(dataloader, model_layers)
```

### Fused Training
```python
# Configure for fused training
config.training.mode = "fused"
config.training.fusion_mode = "frozen"  # or "trainable"

trainer = UnifiedTrainer(config)
trainer.train_all_layers(dataloader, model_layers)
```

### Quantized Training with QLoRA
```python
# Configure for quantized training with QLoRA
config.training.mode = "fused"
config.training.fusion_mode = "frozen"
config.training.quantization_enabled = True
config.training.quantization_type = "nf_fp8"
config.training.qlora_enabled = True
config.training.qlora_rank = 16

trainer = UnifiedTrainer(config)
trainer.train_all_layers(dataloader, model_layers)
```

### Mixed Precision Training
```python
# Configure for mixed precision training
config.training.mixed_precision = True
config.training.backbone_quantized = True
config.training.adapters_full_precision = True

trainer = UnifiedTrainer(config)
trainer.train_all_layers(dataloader, model_layers)
```

### Time-Step-Based Masking Training
```python
# Configure for time-step-based masking
config.training.time_step_masking = True
config.training.num_time_steps = 10
config.training.time_step_mask_fractions = {
    0: 0.15,  # Early time steps: low masking
    5: 0.50,  # Middle time steps: medium masking
    9: 0.90   # Late time steps: high masking
}
config.training.store_all_time_steps = False

trainer = UnifiedTrainer(config)
trainer.train_all_layers(dataloader, model_layers)
```

### Practical Training Regime: Quantized Backbone + QLoRA + Full Precision Blocks
```python
# Recommended training regime for memory efficiency and performance
config.training.mode = "fused"
config.training.fusion_mode = "frozen"
config.training.quantization_enabled = True
config.training.quantization_type = "nf_fp8"
config.training.qlora_enabled = True
config.training.qlora_rank = 16
config.training.time_step_masking = True
config.training.store_all_time_steps = False

trainer = UnifiedTrainer(config)
trainer.train_all_layers(dataloader, model_layers)
```

## Key Benefits

### 1. Progressive Learning
- **Encoder-like behavior** in early layers (low masking)
- **Diffusion-based behavior** in late layers (high masking)
- **Smooth transition** between training paradigms

### 2. Memory Efficiency
- **Hybrid caching** with deduplication
- **O(1) lookup** performance
- **Memory optimization** between layers

### 3. Training Flexibility
- **Multiple training modes** for different scenarios
- **Configurable block sizes** and fusion strategies
- **Progressive training** options

### 4. Performance Optimization
- **Batch-optimized** mask processing
- **Efficient activation caching**
- **Reduced computational overhead**

### 5. Memory & Computational Efficiency
- **Quantized Loading**: NF FP8/FP16 for memory-efficient model loading
- **QLoRA Adapters**: Low-rank adaptation for efficient fine-tuning
- **Mixed Precision**: Full precision training with quantized backbone
- **Selective Updates**: Only update necessary components (QLoRA vs full model)

### 6. Practical Training Regime Benefits
- **Memory Efficiency**: Quantized backbone reduces memory footprint by 2-4x
- **Layer Communication**: All layers remain connected through fused backbone
- **Lightweight Updates**: QLoRA adapters enable efficient fine-tuning
- **Full Precision Training**: Blocks trained in full precision for optimal performance
- **Time-Step Efficiency**: Avoid storing all time-step activations

## Advanced Features

### 1. Language Model Head Integration
- **Pre-trained embedding** reuse as output projection
- **Transposed embeddings** for language modeling
- **Cross-entropy loss** computation with proper logits

### 2. Activation Management
- **Unique activation IDs** based on sample + mask combinations
- **Efficient storage** with tensor-based keys
- **Automatic cleanup** between layers

### 3. Checkpoint Management
- **Layer-specific checkpoints** for layer-wise training
- **Block-specific checkpoints** for block-wise training
- **Fused model checkpoints** for fused training

### 4. Evaluation Support
- **Fusion evaluation** mode for model assessment
- **Boundary activation storage** for analysis
- **Training information** tracking

### 5. Quantization & Adapter Management
- **QLoRA Adapter Creation**: Low-rank adaptation matrices per block
- **Quantized Model Loading**: Memory-efficient loading with NF FP8/FP16
- **Mixed Precision Training**: Full precision adapters with quantized backbone
- **Selective Parameter Updates**: QLoRA-only for frozen blocks, full updates for trainable blocks
- **Activation Generation**: Quantized fused model for efficient activation computation

### 6. Time-Step-Based Masking Management
- **Triplet-Based Mask Generation**: (input_id, mask_pattern, time_t) for precise masking
- **Discrete Time Step Bins**: Efficient storage and retrieval of time-step information
- **Progressive Masking**: Time-step-dependent masking fractions
- **Memory-Efficient Storage**: Avoid storing all time-step activations
- **Selective Activation Storage**: Store only current time-step activations

## Performance Characteristics

### Layer-wise Training
- **Memory**: Low (single layer at a time)
- **Speed**: Medium (sequential training)
- **Flexibility**: High (individual layer control)

### Block-wise Training
- **Memory**: Medium (block-level caching)
- **Speed**: Fast (fewer training phases)
- **Flexibility**: Medium (block-level control)

### Fused Training
- **Memory**: High (multiple blocks)
- **Speed**: Variable (depends on frozen/trainable ratio)
- **Flexibility**: High (progressive strategies)

### Quantized Training with QLoRA
- **Memory**: Very Low (quantized backbone + low-rank adapters)
- **Speed**: Fast (efficient loading and training)
- **Flexibility**: High (selective updates, mixed precision)
- **Scalability**: Excellent (memory-efficient for large models)

### Practical Training Regime (Quantized Backbone + QLoRA + Full Precision Blocks)
- **Memory**: Very Low (quantized backbone + QLoRA adapters)
- **Speed**: Fast (efficient loading + lightweight updates)
- **Layer Communication**: Excellent (all layers connected through fused backbone)
- **Training Quality**: High (full precision block training)
- **Scalability**: Excellent (memory-efficient for large models)
- **Time-Step Efficiency**: High (avoid storing all time-step activations)

## Best Practices

### 1. Mode Selection
- **Layer-wise**: For fine-grained control and debugging
- **Block-wise**: For faster training with good gradient flow
- **Fused**: For progressive learning and parameter sharing

### 2. Configuration Tuning
- **Block size**: 4 layers for optimal balance
- **Mask fractions**: 15%-90% for progressive learning
- **Learning rates**: Lower for fused training

### 3. Memory Management
- **Cache cleanup** between training phases
- **Activation deduplication** for memory efficiency
- **Checkpoint management** for storage optimization

### 4. Quantization & Adapter Best Practices
- **QLoRA Rank**: Start with rank 16-32 for good performance/memory balance
- **Quantization Type**: Use NF FP8 for maximum memory savings, FP16 for better precision
- **Mixed Precision**: Keep adapters in full precision for better training stability
- **Selective Updates**: Use QLoRA-only for frozen blocks to maintain efficiency
- **Memory Monitoring**: Track memory usage with different quantization settings

### 5. Practical Training Regime Best Practices
- **Quantized Backbone**: Load fused model in quantized mode for memory efficiency
- **QLoRA Adapters**: Add low-rank adapters to each block for lightweight updates
- **Full Precision Blocks**: Train blocks in full precision for optimal performance
- **Time-Step Management**: Use discrete time bins to avoid storing all activations
- **Memory Efficiency**: Store only current time-step activations
- **Layer Communication**: Maintain all layer connections through fused backbone

## Clean Redesign: Modular Unified Architecture

### Architectural Redesign Plan
Based on the insights from time-step-based masking and practical training regimes, we propose a complete redesign with a modular, unified architecture.

### New Module Organization
```
src/training/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── unified_trainer.py      # Main unified trainer
│   ├── block_trainer.py        # Block-based training logic
│   └── fusion_trainer.py       # Fusion-specific logic
├── strategies/
│   ├── __init__.py
│   ├── masking/
│   │   ├── __init__.py
│   │   ├── time_step_masking.py    # Time-step-based masking
│   │   ├── progressive_masking.py # Progressive masking strategies
│   │   └── mask_scheduler.py       # Mask scheduling
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── qlora_manager.py        # QLoRA adapter management
│   │   ├── quantization_manager.py # Quantization handling
│   │   └── mixed_precision.py     # Mixed precision training
│   └── caching/
│       ├── __init__.py
│       ├── activation_cache.py     # Activation caching
│       ├── time_step_cache.py     # Time-step-specific caching
│       └── memory_manager.py      # Memory management
├── utils/
│   ├── __init__.py
│   ├── config_validator.py        # Configuration validation
│   ├── checkpoint_manager.py      # Checkpoint management
│   └── metrics.py                 # Training metrics
└── legacy/
    ├── __init__.py
    └── layerwise_trainer.py       # Legacy trainer (deprecated)
```

### Core Design Principles
- **Single Unified Trainer**: One trainer class handling all modes
- **Modular Components**: Pluggable components for different strategies
- **Configuration-Driven**: All behavior controlled by configuration
- **Memory-Efficient**: Optimized for large-scale training
- **Extensible**: Easy to add new training strategies

### Key Classes Design

#### UnifiedTrainer (Main Interface)
```python
class UnifiedTrainer:
    """Unified trainer supporting all training modes"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.block_size = config.training.block_size
        self.training_mode = config.training.mode
        
        # Initialize components
        self.masking_strategy = self._init_masking_strategy()
        self.quantization_manager = self._init_quantization_manager()
        self.cache_manager = self._init_cache_manager()
        self.block_trainer = self._init_block_trainer()
```

#### TimeStepMasking (Masking Strategy)
```python
class TimeStepMasking:
    """Time-step-based masking strategy"""
    
    def __init__(self, config):
        self.num_time_steps = config.training.num_time_steps
        self.time_step_bins = config.training.time_step_bins
        self.mask_fractions = config.training.time_step_mask_fractions
        
    def generate_mask(self, input_id, time_t):
        """Generate mask for specific input and time step"""
        pass
```

#### QLoRAManager (Adapter Management)
```python
class QLoRAManager:
    """QLoRA adapter management"""
    
    def __init__(self, config):
        self.qlora_enabled = config.training.qlora_enabled
        self.qlora_rank = config.training.qlora_rank
        
    def add_adapters_to_block(self, block_layers):
        """Add QLoRA adapters to block layers"""
        pass
```

### Training Flow Design
```python
def train_all_layers(self, dataloader, model_layers):
    """Unified training flow"""
    
    # 1. Initialize components
    self._setup_training_environment()
    
    # 2. Create blocks
    blocks = self._create_blocks(model_layers)
    
    # 3. Train each block
    for block_idx, block_layers in enumerate(blocks):
        self._train_block(block_idx, dataloader, block_layers)
        
    # 4. Finalize training
    self._finalize_training()
```

### Benefits of Clean Redesign
- **Modular Architecture**: Clear separation of concerns
- **Configuration-Driven**: All behavior controlled by config
- **Memory Efficiency**: Time-step-aware caching and quantization
- **Extensibility**: Easy to add new features
- **Maintainability**: Clean, organized code structure
- **Performance**: Optimized for large-scale training

### Implementation Strategy
1. **Phase 1**: Implement core unified trainer with basic functionality
2. **Phase 2**: Add modular components (masking, quantization, caching)
3. **Phase 3**: Implement advanced features (time-step masking, QLoRA)
4. **Phase 4**: Integration, testing, and migration from legacy code

## Future Extensions

### 1. Advanced Fusion Strategies
- **Hierarchical fusion** at different levels
- **Adaptive fusion** based on performance
- **Dynamic block sizing** based on complexity

### 2. Enhanced Caching
- **Distributed caching** for multi-GPU training
- **Compression techniques** for activation storage
- **Smart eviction** policies for memory management

### 3. Training Optimization
- **Gradient accumulation** across blocks
- **Learning rate scheduling** per block
- **Advanced optimization** strategies

## Summary: Advanced Quantization & Adapter Strategy with Time-Step-Based Masking

### Key Features Added:
1. **QLoRA Adapters**: Optional low-rank adaptation matrices per block for efficient fine-tuning
2. **Quantized Loading**: NF FP8/FP16 model loading for memory-efficient training
3. **Mixed Precision Training**: Full precision adapters with quantized backbone
4. **Selective Updates**: QLoRA-only for frozen blocks, full updates for trainable blocks
5. **Quantized Activation Generation**: Memory-efficient activation computation using quantized fused models
6. **Time-Step-Based Masking**: Triplet-based masking (input_id, mask_pattern, time_t) for precise mask generation
7. **Progressive Masking**: Time-step-dependent masking fractions for efficient training
8. **Memory-Efficient Storage**: Avoid storing all time-step activations

### Practical Training Regime:
```
Load Fused Backbone (Quantized) → Add QLoRA Adapters → Train Block (Full Precision) → Generate Time-Step Activations
```

### Training Flow with Quantization & Time Steps:
```
Load Quantized Model → Add QLoRA Adapters → Train in Mixed Precision → Generate Time-Step Activations → Store Current Time Step Only
```

### Memory & Performance Benefits:
- **Memory Efficiency**: Quantized backbone reduces memory footprint by 2-4x
- **Training Speed**: QLoRA adapters enable faster fine-tuning
- **Scalability**: Support for larger models with limited memory
- **Flexibility**: Selective updates based on training requirements
- **Time-Step Efficiency**: Avoid storing all time-step activations
- **Layer Communication**: All layers remain connected through fused backbone

### Recommended Training Strategy:
- **Quantized Backbone**: Load fused model in quantized mode for memory efficiency
- **QLoRA Adapters**: Add low-rank adapters to each block for lightweight updates
- **Full Precision Blocks**: Train blocks in full precision for optimal performance
- **Time-Step Management**: Use discrete time bins to avoid storing all activations

## Usage Examples

### Basic FusionTrainer Usage

```python
from config.base import StackWiseConfig
from training.core.fusion_trainer import FusionTrainer

# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Initialize FusionTrainer
fusion_trainer = FusionTrainer(
    config=config,
    masking_strategy=None,  # Will be set by UnifiedTrainer
    quantization_manager=None,  # Will be set by UnifiedTrainer
    cache_manager=None,  # Will be set by UnifiedTrainer
    lexical_kernel_manager=None  # Will be set by UnifiedTrainer
)

# Create dummy blocks for testing
dummy_blocks = []
for block_idx in range(2):
    block = []
    for layer_idx in range(2):
        layer = torch.nn.Linear(128, 128)
        block.append(layer)
    dummy_blocks.append(block)

# Test disk backup system
fusion_trainer._save_full_precision_weights_to_disk(dummy_blocks, "fp16")

# Test validation
run_id = config.training.run_id
is_valid = fusion_trainer._validate_saved_weights(run_id, "fp16", 2)

# Test restoration
restored_blocks = fusion_trainer._restore_full_precision_weights_from_disk(
    run_id, "fp16", [0, 1]
)

# Test model reconstruction
reconstructed = fusion_trainer._reconstruct_model_from_disk(run_id, "fp16")
```

### Configuration Example

```yaml
# config.yaml
training:
  # Run identification
  run_id: "my_training_run"
  total_blocks: 4
  
  # QLoRA and quantization
  qlora_enabled: true
  qlora_lr: 1.0e-5
  current_block_lr: 1.0e-4
  quantization_enabled: true
  quantization_type: "fp16"  # fp4 | fp8 | fp16 | fp32
  
  # Time-step-based masking
  time_step_masking: true
  num_time_steps: 8
  time_step_mask_fractions: [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
  
  # Training modes
  mode: "fused"  # layerwise | blockwise | fused
  fusion_mode: "frozen"  # frozen | trainable
```

### Testing the System

```python
# Run the comprehensive test suite
python3 test_fusion_direct.py

# Expected output:
# ✅ Configuration loading successful
# ✅ FusionTrainer initialization successful
# ✅ Disk backup system test successful
# ✅ Memory management test successful
# ✅ Optimizer setup test successful
# 🎉 All tests passed successfully!
```

## Summary

This trainer module provides a comprehensive solution for layer-wise transformer training with mask-diffusion objectives, offering flexibility, efficiency, and scalability for various training scenarios. The addition of quantization, QLoRA adapters, and time-step-based masking makes it particularly suitable for memory-constrained environments and large-scale model training.
