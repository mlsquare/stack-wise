# 🚀 Progressive Training System

This document explains the progressive training system for StackWise, which enables building deep models progressively with sophisticated training strategies.

## 📋 Overview

The progressive training system allows you to:
- **Build models progressively** by adding stacks one by one
- **Handle time interpretation** in two ways (time-as-input vs time-as-depth)
- **Manage precision** across different stacks (full, half, bfloat16, nvfp4, QLoRA)
- **Stream activations** through trunk with optional caching
- **Train with different strategies** (frozen trunk vs QLoRA trunk)

## 🔄 Dual-LoRA Approach

The progressive training system implements a sophisticated dual-LoRA approach:

### **Stack LoRA (Always Added)**
- ✅ **Added to each stack** as it's created when `qlora_enabled = True`
- ✅ **Progressive configuration** based on strategy (simplified/progressive/variable)
- ✅ **Per-stack parameters** (rank, alpha) can be customized

### **Progressive QLoRA (Conditionally Added)**
- ✅ **Added to entire trunk** when `progressive_qlora = True` AND there are existing stacks
- ✅ **Smaller rank** for trunk QLoRA (e.g., rank=8 vs rank=16 for stack LoRA)
- ✅ **All existing stacks** get progressive QLoRA adapters when new stacks are added

### **Configuration Example**
```yaml
training:
  progressive:
    # Stack LoRA parameters (added to each stack)
    qlora_enabled: true
    qlora_rank: 16
    qlora_alpha: 32
    
    # Progressive QLoRA parameters (added to trunk when new stacks are added)
    progressive_qlora: true
    progressive_qlora_rank: 8
    progressive_qlora_alpha: 16
```

### **Training Strategies**
- **Frozen Trunk**: All parameters (including LoRA) are frozen
- **QLoRA Trunk**: Only LoRA adapters are trainable (both stack LoRA + progressive QLoRA)

## 🎯 Training Objectives

The progressive training system supports different training objectives:

### **MLM (Masked Language Modeling)**
- **Targets**: Same as inputs (automatically determined from masks)
- **Use case**: BERT-style training, diffusion models
- **Configuration**: `training_objective: "mlm"`

### **CLM (Causal Language Modeling)**
- **Targets**: Shifted inputs (next token prediction)
- **Use case**: GPT-style training, autoregressive models
- **Configuration**: `training_objective: "clm"`
- **⚠️ Warning**: Last target token may not be available (targets[-1] = inputs[-1])
- **Solution**: Use CLM mask to exclude last position from loss computation

### **Custom Objectives**
- **Targets**: Explicitly provided in batch
- **Use case**: Custom tasks, fine-tuning
- **Configuration**: `training_objective: "custom"`

## 🏗️ Architecture Components

### **1. ProgressiveRackBuilder**
Builds Racks progressively with support for append/prepend modes and dual-LoRA approach:

```python
from training.progressive_rack_builder import ProgressiveRackBuilder

# Create builder (fully config-driven)
builder = ProgressiveRackBuilder(
    config=config,  # All parameters read from config (including max_stacks)
    building_mode="append",  # or "prepend"
    default_precision="full"
)

# Add stacks progressively with dual-LoRA approach
stack1 = builder.append_stack(n_blocks=4, precision="full")
stack2 = builder.append_stack(n_blocks=4, precision="half")
stack3 = builder.prepend_stack(n_blocks=4, precision="qlora")

# Build final rack
rack = builder.build_rack()
```

### **1.1. Config-Driven Stack Creation**
You can also create stacks directly using the config-driven approach:

```python
from model.architecture import create_stack_from_config

# Create stack using config (no redundant parameters)
stack = create_stack_from_config(
    stack_id=0,
    n_blocks=4,
    config=config  # All parameters read from config
)

# Or use the builder (recommended)
stack = builder.append_stack(n_blocks=4)  # Uses config internally
```

### **2. ProgressiveDataLoader**
Enhanced dataloader with time interpretation and activation caching:

```python
from training.progressive_dataloader import ProgressiveDataLoader
from training.strategies.masking.time_step_masking import TimeStepMasking

# Create masking strategy
masking_strategy = TimeStepMasking(config)

# Create progressive dataloader
progressive_dataloader = ProgressiveDataLoader(
    base_dataloader=dataloader,
    masking_strategy=masking_strategy,
    stack_idx=0,
    trunk_activations=cached_activations,
    cache_activations=True
)
```

### **3. ProgressiveTrainer**
Orchestrates progressive training with different strategies:

```python
from training.progressive_trainer import ProgressiveTrainer

# Create trainer
trainer = ProgressiveTrainer(config)

# Train progressively
results = trainer.train_progressively(
    rack_builder=builder,
    dataloader=dataloader,
    target_stacks=6
)
```

## 🕐 Time Interpretation

### **Time-as-Input (Standard Diffusion)**
Time is added as an input parameter via positional encoding. For each batch, time is randomly sampled from uniform [0, 1]:

```python
# Configuration
config.training.progressive.time_interpretation = "input"
config.training.progressive.time_embedding_dim = 512
config.training.progressive.time_encoding_type = "sinusoidal"  # or "learned"

# Usage - time is automatically sampled per batch
masking_strategy = TimeStepMasking(config)
# Time is randomly sampled from uniform [0, 1] for each batch
# and converted to discrete time steps (0 to num_time_steps-1)
```

### **Time-as-Depth (Progressive Training)**
Time is tied to stack index:

```python
# Configuration
config.training.progressive.time_interpretation = "depth"
config.training.progressive.stack_time_mapping = "linear"  # or "exponential"
config.training.progressive.time_per_stack = 100

# Usage
masking_strategy = TimeStepMasking(config)
masks = masking_strategy.generate_masks_for_stack(batch, stack_idx=2)
time_step = masking_strategy.get_time_step_for_stack(stack_idx=2)
```

## 🎯 Training Strategies

### **Strategy 1: Frozen Trunk + Full Precision New Stack**
```python
config.training.progressive.trunk_strategy = "frozen"
config.training.progressive.new_stack_precision = "full"

# Previous stacks are frozen, only new stack is trained
trainer = ProgressiveTrainer(config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=6)
```

### **Strategy 2: Simplified QLoRA Trunk + Full Precision New Stack**
```python
config.training.progressive.trunk_strategy = "qlora"
config.training.progressive.new_stack_precision = "full"
config.training.progressive.cache_activations = True

# Simplified QLoRA approach:
# - QLoRA adapters added to ALL stacks
# - When trunk is frozen: all params (including LoRA) are frozen
# - When QLoRA trunk: only LoRA adapters are updated
trainer = ProgressiveTrainer(config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=6)
```

## 🔧 Configuration

### **Progressive Training Configuration**
```yaml
training:
  progressive:
    enabled: true
    trunk_strategy: "frozen"        # "frozen" or "qlora"
    new_stack_precision: "full"    # "full", "half", "bfloat16", "qlora"
    cache_activations: true
    
    # Simplified QLoRA configuration
    qlora_enabled: true             # Enable QLoRA adapters on all stacks
    qlora_rank: 16                  # QLoRA rank for all stacks
    qlora_alpha: 32                 # QLoRA alpha parameter
    qlora_strategy: "simplified"    # "simplified" - add LoRA to all stacks
    
    max_stacks: 12
    building_mode: "append"          # "append" or "prepend"
    
    # Time interpretation
    time_interpretation: "depth"     # "input" or "depth"
    time_embedding_dim: 512        # For time-as-input
    time_encoding_type: "sinusoidal" # "sinusoidal" or "learned"
    stack_time_mapping: "linear"    # "linear", "exponential", "custom"
    time_per_stack: 100            # For time-as-depth
```

## 🚀 Progressive QLoRA Strategy

### **Key Concept:**
The progressive QLoRA approach adds LoRA adapters to each stack as it's created, allowing for different QLoRA configurations per stack:

### **How It Works:**

#### **1. Progressive QLoRA Setup (Per-Stack Configuration)**
```python
# QLoRA adapters are added to each stack during creation
# with stack-specific configurations
config.training.progressive.qlora_enabled = True
config.training.progressive.qlora_strategy = "progressive"  # or "simplified", "variable"
```

#### **2. Frozen Trunk Strategy**
```python
# When trunk is frozen: ALL parameters (including LoRA) are frozen
rack_builder.freeze_trunk(trunk_indices)
# Result: No training on trunk stacks
```

#### **3. QLoRA Trunk Strategy**
```python
# When QLoRA trunk: only LoRA adapters are trainable
rack_builder.freeze_all_but_qlora(trunk_indices)
# Result: Only LoRA adapters are updated, original params frozen
```

#### **4. Training Configuration**
```python
# The trainer automatically configures which parameters are trainable
# based on trunk_strategy setting
if trunk_strategy == "frozen":
    # All parameters (including LoRA) are frozen
elif trunk_strategy == "qlora":
    # Only LoRA adapters are trainable
```

### **QLoRA Strategies:**

#### **1. Simplified Strategy**
```python
# All stacks get the same QLoRA configuration
config.training.progressive.qlora_strategy = "simplified"
config.training.progressive.qlora_rank = 16
config.training.progressive.qlora_alpha = 32
```

#### **2. Progressive Strategy**
```python
# QLoRA parameters change as stacks are added
config.training.progressive.qlora_strategy = "progressive"
config.training.progressive.qlora_rank_pattern = "increasing"  # 16, 32, 64, 128...
config.training.progressive.qlora_alpha_pattern = "constant"   # 32, 32, 32, 32...
```

#### **3. Variable Strategy**
```python
# Custom QLoRA configuration per stack
config.training.progressive.qlora_strategy = "variable"
config.training.progressive.qlora_configs = {
    0: {"rank": 8, "alpha": 16},    # Stack 0: small QLoRA
    1: {"rank": 16, "alpha": 32},  # Stack 1: medium QLoRA
    2: {"rank": 32, "alpha": 64},  # Stack 2: large QLoRA
    3: {"rank": 64, "alpha": 128}  # Stack 3: very large QLoRA
}
```

### **Benefits:**

1. **Flexible Configuration**: Different QLoRA settings per stack
2. **Progressive Patterns**: Increasing/decreasing/linear patterns
3. **Custom Control**: Fine-grained control over each stack
4. **Memory Efficient**: Only needed adapters per stack
5. **Training Flexibility**: Easy switching between frozen and QLoRA modes

## 🚀 Usage Examples

### **Example 1: Basic Progressive Training**
```python
from training import ProgressiveTrainer, ProgressiveRackBuilder
from config.base import StackWiseConfig

# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")
config.validate()

# Create components
rack_builder = ProgressiveRackBuilder(
    vocab_size=config.model.vocab_size,
    d_model=config.model.d_model,
    d_ff=config.model.d_ff,
    n_heads=config.model.n_heads
)

trainer = ProgressiveTrainer(config)

# Train progressively
results = trainer.train_progressively(
    rack_builder=rack_builder,
    dataloader=dataloader,
    target_stacks=6
)
```

### **Example 2: QLoRA Trunk Training**
```python
# Configure for QLoRA trunk
config.training.progressive.trunk_strategy = "qlora"
config.training.progressive.cache_activations = True

# Train with QLoRA trunk
trainer = ProgressiveTrainer(config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=6)
```

### **Example 3: Time-as-Input Training**
```python
# Configure for time-as-input
config.training.progressive.time_interpretation = "input"
config.training.progressive.time_encoding_type = "learned"

# Train with time-as-input
trainer = ProgressiveTrainer(config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=6)
```

### **Example 4: Different Training Objectives**
```python
# MLM objective (default)
config.training.progressive.training_objective = "mlm"
# Targets are automatically determined from inputs and masks

# CLM objective
config.training.progressive.training_objective = "clm"
# Targets are shifted inputs for next token prediction
# WARNING: Last target token may not be available!

# Custom objective
config.training.progressive.training_objective = "custom"
# Targets must be provided in the batch
```

### **Example 5: Handling CLM Limitations**
```python
# For CLM, use the provided CLM mask to exclude last position
for batch in progressive_dataloader:
    if batch['training_objective'] == 'clm':
        # Use combined_masks which excludes the last position
        loss_mask = batch['combined_masks']
        # Or use clm_mask for all positions except last
        clm_mask = batch['clm_mask']
        
        # Compute loss only on valid positions
        loss = compute_loss(logits, targets, loss_mask)
```

## 📊 Benefits

### **Memory Efficiency**
- **Frozen Trunk**: Minimal memory usage for previous stacks
- **QLoRA Trunk**: Reduced memory with QLoRA adapters
- **Activation Caching**: Essential for frozen trunk to avoid recomputation
- **Smart Caching**: Cached activations used as inputs for new stacks

### **Training Flexibility**
- **Progressive Building**: Add stacks as needed
- **Precision Control**: Different precision per stack
- **Time Interpretation**: Choose between diffusion and progressive paradigms

### **Research Friendly**
- **Easy Experimentation**: Test different depths and strategies
- **Modular Design**: Mix and match components
- **Comprehensive Logging**: Track training progress

## 🔄 Training Workflow

### **Progressive Training Pipeline**
1. **Initialize** ProgressiveRackBuilder with configuration
2. **Add Stack** using append/prepend mode
3. **Configure Trunk** (frozen or QLoRA)
4. **Create Dataloader** with time interpretation
5. **Train Stack** with appropriate strategy
6. **Cache Activations** if needed
7. **Repeat** until target depth reached

### **Activation Caching Strategy**

#### **Frozen Trunk + Caching**
- **Cache activations** from each stack after training
- **Use cached activations** as inputs for next stack
- **Avoid recomputation** of frozen layers
- **Memory trade-off**: Store activations vs recompute

#### **QLoRA Trunk + Caching**
- **Cache activations** for efficiency
- **Stream through QLoRA trunk** when needed
- **Flexible caching** based on memory constraints

#### **No Caching**
- **Recompute activations** for each new stack
- **Higher memory usage** but no storage overhead
- **Suitable for** small models or abundant memory

### **Memory Management**
- **Activation Caching**: Store activations for trunk training
- **Precision Control**: Use appropriate precision per stack
- **QLoRA Integration**: Reduce memory for trunk stacks

## 🎯 Best Practices

### **Configuration**
- Start with `trunk_strategy="frozen"` for simplicity
- Use `time_interpretation="depth"` for progressive training
- Enable `cache_activations=True` for QLoRA trunk

### **Training**
- Monitor memory usage with activation caching
- Use appropriate precision for each stack
- Experiment with different building modes

### **Debugging**
- Check training history for each stack
- Monitor activation cache size
- Verify time step assignments

## 🚀 Advanced Features

### **Custom Time Mapping**
```python
# Custom time mapping for time-as-depth
def custom_time_mapping(stack_idx, max_stacks, num_time_steps):
    # Custom logic here
    return min(stack_idx * 50, num_time_steps - 1)
```

### **Mixed Precision Training**
```python
# Different precision for different stacks
rack_builder.append_stack(precision="full")    # First stack
rack_builder.append_stack(precision="half")    # Second stack
rack_builder.append_stack(precision="qlora")   # Third stack
```

### **Activation Streaming**
```python
# Stream activations through trunk
progressive_dataloader = ProgressiveDataLoader(
    base_dataloader=dataloader,
    masking_strategy=masking_strategy,
    stack_idx=current_stack,
    trunk_activations=cached_activations,
    cache_activations=True
)
```

This progressive training system provides a powerful and flexible framework for building deep models with sophisticated training strategies! 🎉
