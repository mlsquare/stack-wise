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

### Masking Strategy
Handles mask generation and application.

```python
from src.training.strategies.masking import MaskScheduler

# Create scheduler
scheduler = MaskScheduler(
    min_fraction=0.1,
    max_fraction=0.9,
    schedule_type="linear"
)

# Get mask fraction for layer
fraction = scheduler.get_mask_fraction(layer_idx, total_layers)
```

### Quantization Manager
Manages model quantization and precision conversion.

```python
from src.training.strategies.quantization import QuantizationManager

# Initialize manager
quant_manager = QuantizationManager(
    precision="fp16",
    enable_qlora=True
)

# Convert model to precision
quantized_model = quant_manager.convert_to_precision(model, "fp16")
```

### Cache Manager
Handles activation caching and storage.

```python
from src.training.strategies.caching import CacheManager

# Initialize manager
cache_manager = CacheManager(
    cache_mode="fusion",
    max_cache_size=1000
)

# Store activations
cache_manager.store_activations(block_idx, activations, masks)

# Retrieve activations
activations = cache_manager.get_activations(block_idx)
```

## Configuration Classes

### ModelConfig
Model architecture configuration.

```python
@dataclass
class ModelConfig(BaseConfig):
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_ff: int = 3072
    attention_type: str = "standard"
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
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_steps: int = 1000
    qlora_enabled: bool = True
    quantization_type: str = "fp16"
    time_step_masking: bool = True
```

## Utility Functions

### Model Creation
```python
def create_gpt2_model(config: ModelConfig) -> List[MLGKALayer]:
    """Create GPT-2 style model layers."""
    layers = []
    for i in range(config.n_layers):
        layer = MLGKALayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_ff=config.d_ff,
            attention_type=config.attention_type,
            attention_mode=config.attention_mode
        )
        layers.append(layer)
    return layers
```

### Data Loading
```python
def create_data_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Implementation details...
    pass
```

## Advanced Features

### QLoRA Adapters
```python
from src.training.core.fusion_trainer import QLoRAAdapter

# Create QLoRA adapter
adapter = QLoRAAdapter(
    original_layer=layer,
    rank=16,
    alpha=32.0,
    dropout=0.1
)

# Forward pass with adaptation
output = adapter(input_tensor)
```

### Disk Backup System
```python
# Save full-precision weights
trainer._save_full_precision_weights_to_disk(trained_blocks, "fp16")

# Restore from disk
restored_blocks = trainer._restore_full_precision_weights_from_disk(
    run_id="my_experiment",
    cache_precision="fp16",
    block_indices=[0, 1, 2]
)
```

### Multi-Learning Rate Optimizer
```python
# Setup optimizer with parameter groups
optimizer = trainer._setup_fusion_optimizer(
    trainable_blocks=all_blocks,
    qlora_enabled=True,
    current_block_idx=1
)

# Parameter groups:
# - qlora_backbone: lr=1e-5
# - current_block: lr=5e-4
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

### Basic Training Loop
```python
# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Initialize trainer
trainer = FusionTrainer(config)

# Train model
trainer.train_with_frozen_backbone()
```

### Custom Model Architecture
```python
# Create custom model
layers = []
for i in range(12):
    layer = MLGKALayer(
        d_model=768,
        n_heads=12,
        n_kv_heads=4,
        d_ff=3072,
        attention_type="gqa",
        attention_mode="bidirectional"
    )
    layers.append(layer)

# Train with custom model
trainer.train_with_custom_model(layers)
```

### Quantized Training
```python
# Enable quantization
config.training.quantization_enabled = True
config.training.quantization_type = "fp16"
config.training.qlora_enabled = True

# Train with quantization
trainer = FusionTrainer(config)
trainer.train_with_frozen_backbone()
```
