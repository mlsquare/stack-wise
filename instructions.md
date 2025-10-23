# 🧠 StackWise — Layer-Wise Transformer with Mask-Diffusion Objective
### Software Requirements Document (SRD)

---

## 1. Project Overview

StackWise develops a **layer-wise trainable Transformer** with a **bidirectional attention training paradigm** that can be fine-tuned for both **masked language modeling (MLM)** and **causal language modeling (CLM)** tasks.

Unlike standard end-to-end training, each Transformer layer is trained *independently* using a **supervised layer-wise schedule** with **bidirectional attention** (BERT-style) during the training phase. Activations from previously trained layers are cached and used as fixed input for the next layer. The final model can be fine-tuned for either **autoregressive generation** (CLM) or **bidirectional understanding** (MLM) tasks.

The training objective generalizes **causal (CLM)** and **masked language modeling (MLM)** by varying the masking rate from partial (≈15%) to nearly full (≈90%) token masking. The diffusion process operates over **token positions**, not embedding noise — aligning closely with *Roberta-Diffusion* and *[MMDiffBERT](https://github.com/mlsquare/mmDiffBERT)* implementations.

---

## 2. Key Architectural Innovation: Bidirectional Training → Task-Specific Fine-tuning

### 2.1 Training Paradigm
- **Training Phase**: Use **bidirectional attention** (like BERT) for more efficient learning
- **Fine-tuning Phase**: Switch to **causal attention** (like GPT) for autoregressive tasks
- **Flexibility**: Support both MLM and CLM fine-tuning from the same base model

### 2.2 Attention Mode Switching
```python
# Training: Bidirectional attention (no causal mask)
attention_mask = None  # Full bidirectional attention

# Inference: Causal attention (autoregressive)
attention_mask = create_causal_mask(seq_len)  # Lower triangular mask
```

---

## 3. Objectives and Rationale

| Goal | Description |
|------|-------------|
| **Bidirectional pre-training efficiency** | Use bidirectional attention for more efficient representation learning during layer-wise training |
| **Task-specific fine-tuning flexibility** | Enable fine-tuning for both MLM (BERT-style) and CLM (GPT-style) tasks |
| **Layer-wise training benefits** | Enable incremental, layer-by-layer training with minimal memory pressure |
| **Explore mask-diffusion supervision** | Replace fixed-mask MLM with variable masking schedules | Following the style of [MMDiffBERT](https://github.com/mlsquare/mmDiffBERT)
| **Integrate modern attention variants** | Incorporate **Grouped Query Attention (GQA)** and **Multi-Latent Attention (MLA)** |
| **Freeze redundant parameters** | Keep **up-projection matrices fixed and non-trainable** |
| **Facilitate research extensibility** | Allow rapid experimentation with different attention, normalization, and diffusion strategies |
| **Kernel View of Attention** | Replace attention with random-kitchen sinks, i.e, after creatin Q,K and V matrices, use a standard random matrix drawn from a chosen family like Gaussian or Laplacian. Effectively, support different type of attention models |


---

## 4. Functional Requirements

### 4.1 Core Components
1. **Model Architecture**
   - Decoder-only Transformer (GPT-style) with **bidirectional attention during training**
   - Configurable number of layers, heads, and model width
   - Options for *standard*, *GQA*, or *MLA* attention
   - **RMSNorm** normalization and **SwiGLU** feed-forward modules
   - Frozen up-projections in the MLP
   - Rotary Positional Embeddings (RoPE)

2. **Attention Mode Configuration**
   - **Training Mode**: Bidirectional attention (no causal mask)
   - **Inference Mode**: Causal attention (autoregressive mask) or Diffusion 
   - **Fine-tuning Mode**: Configurable for MLM or CLM tasks or Diffusion

3. **Layer-wise Supervised Training**
   - Train one layer (and embeddings, if layer-0) at a time
   - Keep the embedding as well as last language head trainable
   - Cache activations after each layer for reuse after the layers are trained
   - At any time only one layer's parameters are updated based on the activations from previous layers
   - **Use bidirectional attention during training for better representation learning**

4. **Fusion Phase**
   - Sequentially merge layer checkpoints into a unified Transformer
   - Optionally run a short **joint fine-tuning** stage
   - Support both MLM and CLM fine-tuning modes

5. **Mask-Diffusion Objective**
   - Randomly mask between *min_mask_fraction* and *max_mask_fraction* of tokens
   - Supervise only masked tokens with cross-entropy
   - Mask tokens, not embeddings — preserving original hidden states
   - **Works with bidirectional attention for better context understanding**

6. **Tied Embeddings**
   - Input and output embeddings share parameters
   - Optional untied mode for ablation

7. **Experiment Control**
   - YAML-based configuration
   - Layer-wise checkpoints and activation caching after layer pre-training
   - Dummy dataset for smoke testing
   - **Attention mode configuration** (bidirectional vs causal)

---

## 5. Non-Functional Requirements

| Category | Requirement |
|-----------|-------------|
| **Scalability** | Handle up to 70B-parameter configurations using layer-wise staging |
| **Reproducibility** | Deterministic initialization and fixed random up-projections |
| **Extensibility** | Modular design supporting future additions like LoRA, MoE, or rotary embeddings |
| **Efficiency** | Reduced optimizer state per layer, minimal GPU memory footprint |
| **Transparency** | Easy checkpoint inspection, clear training logs per layer |
| **Compatibility** | PyTorch ≥ 2.1, CUDA 12+, YAML config files, minimal dependencies |
| **Flexibility** | Support both bidirectional and causal attention modes |

---

## 6. System Design and Flow

### 6.1 Training Stages
1. **Stage-0 (Embeddings + Layer-1):**
   - Train with **bidirectional attention** and diffusion mask objective
   - Cache activations (acts_layer1.pt)

2. **Stage-k (Layer-k):**
   - Load cached activations from previous layer
   - Train only the current layer's parameters with **bidirectional attention**

3. **Fusion Phase:**
   - Merge all layers sequentially
   - Optionally fine-tune entire model on small dataset
   - **Support both MLM and CLM fine-tuning modes**

### 6.2 Data Flow
```
input_ids → Embeddings → Layer_1 (bidirectional) → act_1
                          ↓
                        train Layer_2 (bidirectional) → act_2
                          ↓
                        train Layer_3 (bidirectional) → act_3
                          ↓
                        ...
fuse all layers → Joint Fine-Tuning → Final LM
                          ↓
                    Fine-tune for MLM or CLM tasks
```

### 6.3 Attention Mode Flow
```
Training Phase:     Bidirectional Attention (BERT-style)
                    ↓
Fusion Phase:      Bidirectional Attention (BERT-style)
                    ↓
Fine-tuning Phase:  Causal Attention (GPT-style) OR Bidirectional (BERT-style) with simple MLM or Diffusion MLM
```

---

## 7. Key Innovations

| Innovation | Description |
|-------------|-------------|
| **Bidirectional layer-wise training** | Use bidirectional attention during layer-wise training for better representation learning |
| **Task-specific fine-tuning flexibility** | Support both MLM and CLM fine-tuning from the same base model |
| **Layer-wise supervised learning** | Treat each layer as an independent supervised learner using cached activations |
| **Diffusion-mask objective** | Variable masking rate generalizing MLM/CLM without embedding noise |
| **GQA integration** | Shares K/V projections across head groups to reduce memory and parameters |
| **MLA integration** | Low-rank factorization of Q/K/V projections, reducing FLOPs and cache footprint |
| **Frozen up-projections** | Random but fixed linear projections prevent over-fitting while maintaining expressivity |

---

## 8. Configuration Parameters

| Parameter | Default | Description |
|------------|----------|-------------|
| `vocab_size` | 128000 | Vocabulary size |
| `d_model` | 4096 | Model dimension |
| `n_layers` | 8 | Number of Transformer layers |
| `n_heads` | 32 | Attention heads |
| `n_kv_heads` | 8 | Key/Value heads for GQA |
| `d_ff` | 14336 | Feed-forward dimension |
| `mla_rq` | 1024 | MLA latent dimension for Q |
| `mla_rkv` | 512 | MLA latent dimension for K/V |
| `mask_fraction_min` | 0.15 | Minimum token masking ratio |
| `mask_fraction_max` | 0.90 | Maximum token masking ratio |
| `dropout` | 0.0 | Dropout probability |
| `tie_embeddings` | true | Whether to share input/output embeddings |
| `attention_mode` | "bidirectional" | "bidirectional" for training, "causal" for inference |
| `fine_tune_mode` | "clm" | "clm" for autoregressive, "mlm" for bidirectional fine-tuning |
| `attention_type` | "standard" | or Kernel-based
| `lr` | 1e-4 | Learning rate per layer |
| `batch_size` | 4 | Batch size |
| `seq_len` | 512 | Context length |

---

## 9. Implementation Strategy

### 9.1 Modular Architecture (✅ COMPLETED)
```
src/
├── config/           # Configuration management
├── model/           # Model components (attention, normalization, etc.)
├── training/        # Training pipelines (layerwise, fusion, fine-tuning)
├── data/           # Data handling and preprocessing
├── utils/          # Utilities and helpers
└── scripts/        # Entry point scripts
```

### 9.2 Implementation Phases

#### **Phase 1: Foundation (IN PROGRESS)**
1. **Configuration System** ⬅️ *Current Focus*
   - Hierarchical configuration classes
   - Validation and defaults
   - Support for all new parameters

2. **Core Attention System**
   - Base attention class with mode switching
   - All attention variants (standard, GQA, MLA, kernel)
   - Bidirectional vs causal attention modes

#### **Phase 2: Training Pipeline**
3. **Layer-wise Training**
   - Scalable training loop for all layers
   - Proper activation caching
   - Memory optimization

4. **Fusion Phase**
   - Model combination logic
   - Joint fine-tuning support

#### **Phase 3: Advanced Features**
5. **Fine-tuning Modules**
   - CLM, MLM, and Diffusion fine-tuning
   - Task-specific loss functions

6. **Production Features**
   - Error handling, logging, checkpointing
   - Metrics and evaluation

### 9.3 Configuration Updates
```yaml
# Model configuration
attention_mode: "bidirectional"  # bidirectional | causal
attention_type: "standard"       # standard | gqa | mla | kernel
fine_tune_mode: "clm"           # clm | mlm | diffusion
use_rope: true                   # true | false
```

---

## 10. Deliverables

1. **Repository Scaffold**
   - `/src/model/core.py` — model definitions with attention mode support
   - `/src/train_layerwise.py` — trainer logic with bidirectional training
   - `/src/fine_tune.py` — fine-tuning logic for MLM/CLM tasks
   - `/src/data/dummy_dataset.py` — mock dataset for smoke tests
   - `/config.yaml` — configuration template with attention modes
   - `/README.md` — documentation and usage guide

2. **Artifacts**
   - Layer checkpoints (`layer_k.pt`)
   - Cached activations (`acts_layerk.pt`)
   - Final fused model checkpoint
   - Fine-tuned models for MLM/CLM tasks

---

## 11. Usage Examples

### 11.1 Layer-wise Training (Bidirectional)
```bash
python -m src.train_layerwise --config config.yaml --attention_mode bidirectional
```

### 11.2 Fine-tuning for CLM (Autoregressive)
```bash
python -m src.fine_tune --config config.yaml --mode clm --checkpoint fused_model.pt
```

### 11.3 Fine-tuning for MLM (Bidirectional)
```bash
python -m src.fine_tune --config config.yaml --mode mlm --checkpoint fused_model.pt
```

---

## 12. Future Extensions

1. **LoRA/Adapter Fusion** for parameter-efficient fine-tuning
2. **Mixture-of-Experts** layers within layer-wise training framework
3. **Hybrid diffusion objectives** operating jointly in embedding and token space
4. **Curriculum masking schedules** where masking rate increases over epochs
5. **Multilingual data handling** with shared embeddings and diffusion consistency loss
6. **Attention pattern analysis** for understanding bidirectional vs causal learning

---

## 13. Licensing & Attribution

- Base implementation under **MIT License**
- Integrates conceptual ideas from:
  - *Roberta-Diffusion (Nathan)*
  - *DeepSeek-V2/V3* (MLA formulation)
  - *MMDiffBird (mlsquare)* internal research prototype
  - *BERT* (bidirectional attention paradigm)
  - *GPT* (causal attention paradigm)

---

**End of SRD**
