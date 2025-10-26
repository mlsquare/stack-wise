# API Reference

This document provides a comprehensive reference for the Stack-Wise training system APIs.

## Core Classes

### StackWiseConfig
Main configuration class for the entire system.

```python
from src.config.base import StackWiseConfig

# Load from YAML
config = StackWiseConfig.from_yaml("config.yaml")

# Validate configuration
config.validate()

# Access model config
model_config = config.model
training_config = config.training
```

### ProgressiveTrainer
Main progressive training interface for the new system.

```python
from src.training import ProgressiveTrainer

# Initialize trainer
trainer = ProgressiveTrainer(config=config)

# Train rack progressively
results = trainer.train_rack(rack_builder, dataloader, target_stacks=3)
```

### ProgressiveRackBuilder
Config-driven progressive rack building with dual-LoRA support.

```python
from src.training import ProgressiveRackBuilder

# Initialize builder
rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")

# Add stacks progressively
stack1 = rack_builder.append_stack(n_blocks=4, precision="full")
stack2 = rack_builder.append_stack(n_blocks=4, precision="half")

# Build final rack
rack = rack_builder.build_rack()
```

### Trainer (Hierarchical)
Hierarchical trainer for Block/Stack/Rack training.

```python
from src.training import Trainer

# Initialize trainer
trainer = Trainer(config=config)

# Train specific components
trainer.train_block(block, dataloader)
trainer.train_stack(stack, dataloader)
trainer.train_rack(rack, dataloader)
```

## Model Components

### MLGKALayer
Multi-Latent Grouped Kernel Attention layer.

```python
from src.model.layers import MLGKALayer

# Create layer
layer = MLGKALayer(
    d_model=768,
    n_heads=12,
    n_kv_heads=4,
    d_ff=3072,
    attention_type="gqa",
    attention_mode="bidirectional"
)

# Forward pass
output = layer(input_tensor)
```

### LexicalKernelManager
Manages lexical embeddings and provides language model head.

```python
from src.model.layers import LexicalKernelManager

# Initialize with pre-trained embeddings
manager = LexicalKernelManager(
    family="gpt2",
    embedding_option="embed_tokens",
    freeze_embeddings=True,
    target_model_dim=768
)

# Get language model head
lm_head = manager.get_lm_head()

# Get embeddings
embeddings = manager.get_embeddings()
```

### SwiGLUFFN
SwiGLU feed-forward network with optional frozen up-projections.

```python
from src.model.layers import SwiGLUFFN

# Create FFN
ffn = SwiGLUFFN(
    d_model=768,
    d_ff=3072,
    freeze_up_proj=True
)

# Forward pass
output = ffn(input_tensor)
```

## Training Strategies

### ProgressiveMasking
Progressive masking strategy with variable mask fractions.

```python
from src.training.strategies.masking import ProgressiveMasking

# Create masking strategy
masking_strategy = ProgressiveMasking(config)

# Generate masks for a stack
masks = masking_strategy.generate_masks_for_stack(batch, stack_idx)

# Generate masks with layer index
masks = masking_strategy.generate_masks(batch, layer_idx)
```

**Key Features:**
- Progressive masking fraction from 15% to 90%
- Configurable schedule (linear, exponential, cosine)
- Depth-based time interpretation

### TimeStepMasking (Experimental)
Time-step-based masking with dual time interpretations.

```python
from src.training.strategies.masking import TimeStepMasking

# Create masking strategy
masking_strategy = TimeStepMasking(config)

# Set time interpretation
masking_strategy.time_interpretation = "depth"  # or "input"

# Generate masks for time step
masks = masking_strategy.generate_masks_for_time_step(batch, time_t)
```

**Key Features:**
- Supports time-as-depth and time-as-input interpretations
- Discrete time steps with progressive masking
- Mask caching for efficiency
- ⚠️ **Experimental** - use with caution

### BaseMaskingStrategy
Abstract base class for masking strategies.

```python
from src.training.strategies.masking import BaseMaskingStrategy

# All masking strategies inherit from this base class
```

### Deprecated Modules

⚠️ **Note**: The following modules are deprecated and no longer functional:

- `QuantizationManager` - Deprecated, quantization handled by ProgressiveRackBuilder
- `QLoRAManager` - Deprecated, QLoRA handled by ProgressiveRackBuilder
- `CacheManager` - Deprecated, caching implemented in ProgressiveDataLoader
- `TimeStepCache` - Deprecated, caching implemented in ProgressiveDataLoader
- `ActivationCache` - Deprecated, caching implemented in ProgressiveDataLoader

## Configuration Classes

### ModelConfig
Model architecture configuration.

```python
@dataclass
class ModelConfig(BaseConfig):
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 12
    d_ff: int = 3072
    attention_type: str = "mha"
    attention_mode: str = "causal"
    use_rope: bool = False
    tie_embeddings: bool = True
```

### TrainingConfig
Training parameters configuration.

```python
@dataclass
class TrainingConfig(BaseConfig):
    mode: str = "layerwise"
    block_size: int = 4
    fusion_mode: str = "frozen"
    total_blocks: int = 2
    batch_size: int = 8
    max_steps: int = 1000
    qlora_enabled: bool = True
    quantization_type: str = "fp16"
    time_step_masking: bool = True
```

## Data Loaders

### ProgressiveDataLoader
Enhanced DataLoader for progressive training with activation caching.

```python
from src.training import ProgressiveDataLoader
from src.training.strategies.masking import ProgressiveMasking

# Create masking strategy
masking_strategy = ProgressiveMasking(config)

# Create progressive data loader
dataloader = ProgressiveDataLoader(
    base_dataloader=original_dataloader,
    masking_strategy=masking_strategy,
    stack_idx=0,
    trunk_activations=None,
    enable_trunk_cache=True
)

# Iterate over batches
for batch in dataloader:
    # Batch includes input_ids, targets, masks, and trunk_activations
    pass
```

**Key Features:**
- Automatic mask generation for each batch
- Trunk activation caching and injection
- Support for time-as-depth and time-as-input interpretations
- Dictionary-based activation caching

### CachedDataLoader
Simple wrapper for cached data loading.

```python
from src.training import CachedDataLoader

# Wrap an existing dataloader
cached_dataloader = CachedDataLoader(base_dataloader)
```

## Precision Management

### PrecisionManager
Manages model precision conversions for memory efficiency.

```python
from src.training import PrecisionManager

# Create precision manager
precision_manager = PrecisionManager()

# Convert model to half precision
model_fp16 = precision_manager.to_half(model)

# Convert model to full precision
model_fp32 = precision_manager.to_full(model)

# Mixed precision setup
with precision_manager.mixed_precision_context():
    loss = model(input_ids)
```

## Utility Classes

### ConfigValidator
Configuration validation utility.

```python
from src.training import ConfigValidator

# Create validator
validator = ConfigValidator(config)

# Validate configuration
validator.validate()

# Get errors
if not validator.is_valid():
    errors = validator.get_errors()
    print(f"Configuration errors: {errors}")
```

### CheckpointManager
Manages training checkpoints.

```python
from src.training import CheckpointManager

# Create checkpoint manager
checkpoint_manager = CheckpointManager(config)

# Save checkpoint
checkpoint_manager.save_checkpoint(
    block_idx=0,
    model_layers=layers,
    optimizer=optimizer,
    epoch=1,
    loss=0.5
)

# Load checkpoint
state = checkpoint_manager.load_checkpoint(block_idx=0)
```

### TrainingMetrics
Training metrics tracking.

```python
from src.training import TrainingMetrics

# Create metrics tracker
metrics = TrainingMetrics()

# Log metrics
metrics.log_loss(loss_value)
metrics.log_step(step_number)

# Get metrics
all_metrics = metrics.get_metrics()
```

## Error Handling

### Common Exceptions
```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

class TrainingError(Exception):
    """Raised when training fails."""
    pass

class QuantizationError(Exception):
    """Raised when quantization fails."""
    pass
```

### Validation
```python
# Validate configuration
try:
    config.validate()
except ConfigurationError as e:
    print(f"Configuration error: {e}")

# Validate model
try:
    model.validate()
except ModelError as e:
    print(f"Model error: {e}")
```

## Examples

### Basic Progressive Training
```python
# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Initialize rack builder
rack_builder = ProgressiveRackBuilder(config, building_mode="append")

# Add stacks progressively
for i in range(3):
    stack = rack_builder.append_stack(n_blocks=4, precision="half")
    print(f"Added stack {i+1}")

# Build final rack
rack = rack_builder.build_rack()

# Initialize trainer
trainer = ProgressiveTrainer(config)

# Train rack progressively
results = trainer.train_rack(rack_builder, dataloader, target_stacks=3)
```

### Progressive Training with Masking
```python
# Create masking strategy
from src.training.strategies.masking import ProgressiveMasking

masking_strategy = ProgressiveMasking(config)

# Create progressive data loader
dataloader = ProgressiveDataLoader(
    base_dataloader=base_dataloader,
    masking_strategy=masking_strategy,
    stack_idx=0
)

# Train with progressive masking
trainer = ProgressiveTrainer(config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=3)
```

### Precision Management
```python
# Create precision manager
precision_manager = PrecisionManager()

# Convert rack to half precision for training
rack_fp16 = precision_manager.to_half(rack)

# Train with half precision
trainer = ProgressiveTrainer(config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=3)

# Convert back to full precision for inference
rack_fp32 = precision_manager.to_full(rack_fp16)
```
