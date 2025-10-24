# Stack-Wise Trainer Module

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

#### 4. UnifiedTrainer
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

## Proposed Refactoring: Unified Block-Based Architecture

### Current Architecture Issues
- **Code Duplication**: LayerwiseTrainer and BlockwiseTrainer share similar logic
- **Maintenance Overhead**: Changes need to be applied to multiple classes
- **Inconsistent APIs**: Different interfaces for similar functionality

### Proposed Unified Architecture
```python
class UnifiedBlockTrainer:
    """Unified trainer supporting all training modes through block_size parameter"""
    
    def __init__(self, config):
        self.block_size = config.training.block_size
        self.fusion_mode = config.training.fusion_mode
        
        # Single trainer handles all modes:
        # - block_size=1: Layer-wise training
        # - block_size=4: Block-wise training  
        # - fusion_mode: Fused training strategies
```

### Benefits of Unified Architecture
- **Single Codebase**: One trainer class for all modes
- **Consistent API**: Same interface across all training modes
- **Easy Mode Switching**: Change block_size to switch modes
- **Reduced Complexity**: Simpler mental model and maintenance

### Migration Strategy
1. **Phase 1**: Implement UnifiedBlockTrainer alongside existing classes
2. **Phase 2**: Update UnifiedTrainer to use UnifiedBlockTrainer
3. **Phase 3**: Deprecate LayerwiseTrainer and BlockwiseTrainer
4. **Phase 4**: Remove deprecated classes

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

This trainer module provides a comprehensive solution for layer-wise transformer training with mask-diffusion objectives, offering flexibility, efficiency, and scalability for various training scenarios.
