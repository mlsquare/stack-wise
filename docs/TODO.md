# 📋 StackWise TODO List

This document outlines all pending tasks, improvements, and critical issues that need to be addressed in the StackWise framework.

## ✅ Recently Completed

### Dual-LoRA Implementation (Completed)
- ✅ **Implemented dual-LoRA approach** with stack LoRA + progressive QLoRA
- ✅ **Added `_add_qlora_to_stack()`** for adding LoRA to individual stacks
- ✅ **Added `_add_qlora_to_trunk()`** for adding QLoRA to entire trunk
- ✅ **Updated both `append_stack()` and `prepend_stack()`** with consistent logic
- ✅ **Added progressive QLoRA configuration** with `progressive_qlora` parameter

### Precision Support (Completed)
- ✅ **Added NVFP4 precision support** (NVIDIA FP4 format)
- ✅ **Fixed QLoRA documentation** (QLoRA is not a precision)
- ✅ **Updated precision modes** to include `nvfp4`
- ✅ **Added proper handling** for NVFP4 in `PrecisionManager`

### Code Cleanup (Completed)
- ✅ **Fixed undefined `max_stacks` variable** in `ProgressiveRackBuilder`
- ✅ **Removed duplicate methods** and redundant code
- ✅ **Updated method naming** for consistency
- ✅ **Deleted temporary test files**
- ✅ **Updated configuration** with new parameters

### Documentation Updates (Completed)
- ✅ **Updated PROGRESSIVE_TRAINING.md** with dual-LoRA approach
- ✅ **Updated CONFIGURATION_GUIDE.md** with new parameters
- ✅ **Updated CHANGELOG.md** with recent changes
- ✅ **Updated README.md** with new features

## 🚨 Critical Architecture Issues

### High Priority - Core Functionality Gaps

#### 1. RoPE + MLA Compatibility
- **Issue**: Rotary Position Embedding (RoPE) may not be compatible with Multi-Latent Attention (MLA)
- **Impact**: Position encoding could be broken in MLA layers, causing training instability
- **Risk Level**: **CRITICAL**
- **Status**: Pending
- **Files to Check**: `src/model/attention/`, `src/model/layers.py`

#### 2. Diffusion Noise + MASK Token Handling
- **Issue**: Missing proper diffusion noise injection and MASK token processing at input
- **Impact**: Core diffusion training functionality is incomplete
- **Risk Level**: **CRITICAL**
- **Status**: Pending
- **Files to Implement**: `src/training/strategies/masking/diffusion_noise.py`

#### 3. Embedding Adaptation
- **Issue**: No input/output embedding adaptation mechanisms
- **Impact**: Limited flexibility for different model architectures
- **Risk Level**: **HIGH**
- **Status**: Pending
- **Files to Implement**: `src/model/embeddings/`

#### 4. Denoising Schedule
- **Issue**: No proper denoising schedule implementation for diffusion training
- **Impact**: Diffusion training may not converge properly
- **Risk Level**: **HIGH**
- **Status**: Pending
- **Files to Implement**: `src/training/strategies/masking/denoising_schedule.py`

## 🔧 Infrastructure & Code Quality

### Medium Priority - System Improvements

#### 5. Fix Deprecated torch_dtype Warning
- **Issue**: `torch_dtype` parameter is deprecated, should use `dtype`
- **Impact**: Deprecation warnings in model creation
- **Risk Level**: **LOW**
- **Status**: Pending
- **Files to Fix**: `src/model/layers.py`, `examples/gpt2_fusion/train_gpt2_fusion.py`

#### 6. Implement Missing Modules
- **Issue**: Referenced modules don't exist:
  - `mask_scheduler.py`
  - `mixed_precision.py` 
  - `memory_manager.py`
- **Impact**: Import errors and incomplete functionality
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Create**: 
  - `src/training/strategies/masking/mask_scheduler.py`
  - `src/training/strategies/quantization/mixed_precision.py`
  - `src/training/strategies/caching/memory_manager.py`

#### 7. Improve Error Handling
- **Issue**: Limited error handling and validation throughout the system
- **Impact**: Poor user experience, difficult debugging
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Update**: All training modules

#### 8. Add Comprehensive Tests
- **Issue**: Limited test coverage for components
- **Impact**: Bugs may go undetected, difficult to refactor
- **Risk Level**: **HIGH**
- **Status**: Pending
- **Files to Create**: `tests/` directory expansion

#### 9. Optimize Memory Usage
- **Issue**: Memory usage could be more efficient for large models
- **Impact**: Limited scalability, potential OOM errors
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Optimize**: `src/training/core/fusion_trainer.py`

#### 10. Performance Optimization
- **Issue**: Training performance could be improved
- **Impact**: Slower training, higher computational costs
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Optimize**: All training modules

#### 11. Configuration Validation Improvements
- **Issue**: Better error messages needed for configuration issues
- **Impact**: Difficult to debug configuration problems
- **Risk Level**: **LOW**
- **Status**: Pending
- **Files to Update**: `src/config/base.py`

#### 12. Intelligent Saving System
- **Issue**: Current saving system is inefficient - saves entire rack every time
- **Impact**: Wastes storage space and time re-saving unchanged stacks
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Implement**: 
  - `src/training/utils/intelligent_checkpointing.py`
  - `src/training/utils/delta_compression.py`
- **Proposed Solution**: 
  - Only save stacks that have been modified since last checkpoint
  - Implement delta-based rack saving (save only changes)
  - Track stack modification timestamps/hashes
  - Maintain checkpoint metadata for reconstruction

#### 13. Documentation Updates
- **Issue**: Documentation needs to reflect current implementation
- **Impact**: Users may follow outdated instructions
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Update**: All `docs/` files

## 🤖 Model Support & Variants

### Medium Priority - Framework Expansion

#### 14. tinyLlaMA + nanoGPT Variants
- **Issue**: Limited support for different model architectures
- **Impact**: Framework not truly universal
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Create**: `examples/tinyllama/`, `examples/nanogpt/`

#### 15. BERT Variant Support
- **Issue**: No encoder-only model support
- **Impact**: Cannot train BERT-style models
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Create**: `examples/bert_fusion/`

#### 16. Decoder Head + CLM Training
- **Issue**: Missing autoregressive training capabilities
- **Impact**: Cannot do proper generative training
- **Risk Level**: **HIGH**
- **Status**: Pending
- **Files to Implement**: `src/model/decoder_head.py`

#### 17. Context Length Management
- **Issue**: No variable sequence length handling
- **Impact**: Limited to fixed context lengths
- **Risk Level**: **MEDIUM**
- **Status**: Pending
- **Files to Implement**: `src/training/strategies/context_length.py`

## 📊 Evaluation & Benchmarking

### High Priority - Performance Measurement

#### 18. Encoder Benchmarks
- **Issue**: No evaluation framework for NLP/NLU tasks
- **Impact**: Cannot measure model performance on downstream tasks
- **Risk Level**: **HIGH**
- **Status**: Pending
- **Files to Create**: `benchmarks/encoder_tasks.py`

#### 19. Perplexity Benchmarks
- **Issue**: No generative mode evaluation
- **Impact**: Cannot measure language modeling performance
- **Risk Level**: **HIGH**
- **Status**: Pending
- **Files to Create**: `benchmarks/perplexity.py`

## 🎯 Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. RoPE + MLA compatibility
2. Diffusion noise + MASK handling
3. Denoising schedule
4. Decoder head + CLM training

### Phase 2: Infrastructure (Short-term)
5. Fix deprecated warnings
6. Implement missing modules
7. Improve error handling
8. Add comprehensive tests
9. Intelligent saving system

### Phase 3: Model Support (Medium-term)
10. tinyLlaMA + nanoGPT variants
11. BERT variant support
12. Context length management
13. Embedding adaptation

### Phase 4: Evaluation (Long-term)
14. Encoder benchmarks
15. Perplexity benchmarks
16. Performance optimization
17. Documentation updates

## 📝 Notes

- **Total Tasks**: 19
- **Critical Issues**: 4
- **High Priority**: 8
- **Medium Priority**: 6
- **Low Priority**: 0

This TODO list represents a comprehensive roadmap for making StackWise a production-ready, feature-complete framework for layer-wise transformer training with diffusion objectives.

## 🔄 Status Legend

- **Pending**: Not started
- **In Progress**: Currently being worked on
- **Completed**: Finished
- **Cancelled**: No longer needed

---

*Last Updated: 2024-10-24*
*Total Tasks: 19*
