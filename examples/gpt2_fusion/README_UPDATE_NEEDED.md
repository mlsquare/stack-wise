# GPT-2 Fusion Examples - UPDATE NEEDED

## ⚠️ Status: BROKEN

The GPT-2 fusion examples in this directory are currently broken because they depend on the old `FusionTrainer` from `src/training/core/` which has been removed.

## 🔧 Required Updates

The following files need to be updated to use the new progressive training system:

- `train_gpt2_fusion.py` - Main training script
- `simple_train.py` - Simplified training script  
- `evaluate_gpt2.py` - Evaluation script
- `setup.py` - Setup script

## 🚀 Migration Path

Replace the old `FusionTrainer` with the new progressive training components:

### Old Approach (Broken):
```python
from src.training.core.fusion_trainer import FusionTrainer

fusion_trainer = FusionTrainer(config=config, ...)
```

### New Approach (Working):
```python
from src.training import ProgressiveTrainer, ProgressiveRackBuilder

# Create progressive rack builder
rack_builder = ProgressiveRackBuilder(config=config)

# Add stacks progressively
stack1 = rack_builder.append_stack(n_blocks=4)

# Train with progressive trainer
trainer = ProgressiveTrainer(config=config)
results = trainer.train_rack(rack_builder, dataloader, target_stacks=1)
```

## 📝 TODO

- [ ] Update `train_gpt2_fusion.py` to use progressive training
- [ ] Update `simple_train.py` to use progressive training
- [ ] Update `evaluate_gpt2.py` to use progressive training
- [ ] Update `setup.py` to use progressive training
- [ ] Test all updated examples
- [ ] Update documentation

## 🎯 Benefits of Migration

- ✅ **Dual-LoRA support** - Advanced LoRA strategies
- ✅ **Progressive building** - Add stacks incrementally
- ✅ **Multiple precision modes** - full, half, bfloat16, nvfp4
- ✅ **Enhanced configuration** - More flexible setup
- ✅ **Better performance** - Optimized training strategies
