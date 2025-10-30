# Unified Time-Based Training Framework Implementation

## Overview

This document describes the implementation of the unified time-based training framework that orchestrates data, model, and optimizer evolution according to the design documents:
- `stackwise_framework_reformulation.md` - Conceptual synthesis
- `framework_glue_design.md` - Technical specifications

## Architecture

### Core Components

#### 1. Specs (`src/framework/specs.py`)
Lightweight dataclasses that define contracts between components:
- `BatchSpec`: Data batch specification
- `EmbeddingSpec`: Embedding layer specification  
- `RackSpec`: Rack (stack collection) specification
- `HeadSpec`: Output head specification
- `OptimizerSpec`: Optimizer configuration
- `DataSpec`: Data loader specification

All specs support serialization (`to_dict()`, `from_dict()`) and validation.

#### 2. Adapters (`src/framework/adapters.py`)
Bridge existing implementations to framework specs:
- `DataAdapter`: Wraps DataLoaders with BatchSpec
- `ModelAdapter`: Wraps models (Embedding + Rack + Head) with specs
- `OptimizerAdapter`: Wraps optimizers with OptimizerSpec

Adapters provide consistent interfaces while preserving existing functionality.

#### 3. Factories (`src/framework/factories.py`)
Build components from config or existing objects:
- `build_data_loader()`: Creates data adapters
- `build_model()`: Creates model adapters
- `build_optimizer()`: Creates optimizer adapters
- `validate_training_state()`: Ensures component compatibility

Factories delegate to existing builders (e.g., `create_rack_from_config`).

#### 4. Schedule (`src/training/schedule.py`)
Time-based curriculum management:
- `Phase`: Training phase with time range and config overrides
- `PhaseSchedule`: Sequence of phases covering [0,1]
- `TimeController`: Converts steps ↔ normalized time t ∈ [0,1]

Supports both step-based and normalized-time-based phase definitions.

#### 5. Training Machine (`src/training/training_machine.py`)
Core orchestrator that:
1. Maintains time controller and phase schedule
2. Assembles (Data_t, Model_t, Optimizer_t) at each step
3. Coordinates training loops with phase transitions
4. Tracks metrics and manages checkpoints

### Integration Points

#### Progressive Components
Extended with spec hooks:
- `ProgressiveDataLoader.get_batch_spec()`: Returns BatchSpec
- `ProgressiveRackBuilder.to_spec()`: Returns RackSpec
- `Rack.to_spec()`: Returns RackSpec

These enable framework integration without breaking existing code.

## Usage

### Basic Single-Phase Training

```python
from training import TrainingMachine

config = {
    'model': {
        'vocab_size': 1000,
        'd_model': 256,
        'n_heads': 8,
        'architecture': {'n_stacks': 2, 'blocks_per_stack': 2}
    },
    'training': {
        'batch_size': 4,
        'seq_len': 128,
        'optimizer': {'optimizer_type': 'AdamW', 'lr': 1e-4}
    },
    'data': {'use_dummy_data': True}
}

machine = TrainingMachine(config, total_steps=1000)
results = machine.train()
```

### Multi-Phase Curriculum

```python
config = {
    # ... model, training, data config ...
    'schedule': [
        {
            'start_t': 0.0,
            'end_t': 0.5,
            'name': 'warmup',
            'optimizer_config': {'lr': 1e-5}
        },
        {
            'start_t': 0.5,
            'end_t': 1.0,
            'name': 'main',
            'optimizer_config': {'lr': 1e-4}
        }
    ]
}

machine = TrainingMachine(config, total_steps=1000)
results = machine.train()
```

### Using with Existing Progressive Components

```python
from training import ProgressiveTrainer, ProgressiveRackBuilder
from config.base import StackWiseConfig

config = StackWiseConfig.from_yaml("config.yaml")
builder = ProgressiveRackBuilder(config)
trainer = ProgressiveTrainer(config)

# Existing code continues to work
results = trainer.train_rack(builder, dataloader, target_stacks=4)
```

## Key Features

### 1. Time as First-Class Concept
- Normalized time t ∈ [0,1] governs training
- Automatic step ↔ time conversion
- Phase transitions at specified times

### 2. Three-Plane Architecture
- **Data Plane**: Dataset selection, transforms, augmentation
- **Model Plane**: Architecture evolution, freeze/unfreeze
- **Optimizer Plane**: Learning dynamics, schedules

### 3. Curriculum Learning
- Multi-phase training with per-phase configs
- Smooth transitions between phases
- Config overrides at phase boundaries

### 4. Backward Compatibility
- Existing progressive training continues to work
- Spec hooks added without breaking changes
- Legacy code paths preserved

### 5. Validation and Safety
- Spec compatibility checks
- Phase schedule validation
- Input/output shape verification

## Benefits

### Design Benefits
1. **Explicit Time**: Training state tracked by normalized time
2. **Separation of Concerns**: Data/Model/Optimizer evolve independently
3. **Testability**: Components have clear interfaces and specs
4. **Composability**: Phases can be combined for complex curricula

### Implementation Benefits
1. **Non-Breaking**: Existing code continues to work
2. **Minimal Changes**: Adapter pattern preserves implementations
3. **Optional Adoption**: Can use progressively or all at once
4. **Extensible**: Easy to add new phases or components

## Examples

See `examples/training_machine_example.py` for:
- Single-phase training
- Two-phase curriculum
- Phase transition tracking
- Time controller usage

## Testing

Unit tests in `tests/unit/test_unified_framework.py` cover:
- Spec creation and validation
- Phase schedule operations
- Time controller conversions
- Adapter functionality
- Training machine basic operations

## Migration Path

### Phase 1: Use Specs (Completed)
- Add spec dataclasses
- No changes to existing code

### Phase 2: Add Adapters (Completed)
- Wrap existing components
- Maintain compatibility

### Phase 3: Introduce Schedule (Completed)
- Add time-based scheduling
- Keep progressive training

### Phase 4: Deploy Training Machine (Completed)
- New code uses unified framework
- Legacy code continues working

### Future: Deprecation (Optional)
- Gradually migrate to unified framework
- Remove legacy paths when no longer needed

## Design Critique

### Current Design (Progressive Training)
**Strengths:**
- Strong progressive modules (rack builder, dataloader)
- Config-driven approach
- Flexible QLoRA and precision management

**Weaknesses:**
- Time implicit across configs
- No single source of truth for training state at step t
- Phase transitions not explicit
- Spec drift possible between components

### Proposed Design (Unified Framework)
**Strengths:**
- Time explicit and first-class
- Single source of truth via TrainingMachine
- Clear separation of concerns (data/model/optimizer planes)
- Testable via specs and adapters
- Backward compatible via adapters

**Potential Risks:**
- Spec drift requires maintenance
- Adapter layer adds indirection
- Additional concepts to learn
- Mitigated by: comprehensive tests, clear examples, gradual adoption

## Conclusion

The unified time-based training framework provides a robust foundation for curriculum learning while preserving backward compatibility with existing progressive training code. The implementation successfully separates concerns, makes time explicit, and enables complex training curricula through phase-based configuration.

