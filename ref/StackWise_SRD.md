# 🧠 Software Requirements Document (SRD)
### Project: *StackWise — Layer-Wise Transformer with Mask-Diffusion Objective*

---

## 1. Project Overview

StackWise develops a **layer-wise trainable Transformer**—architecturally similar to GPT-2/LLaMA but incorporating **modern attention mechanisms** and a **diffusion-style token masking objective**.

Unlike standard end-to-end training, each Transformer layer is trained *independently* using a **supervised layer-wise schedule**. Activations from previously trained layers are cached and used as fixed input for the next layer. The final model is assembled by sequentially fusing all trained layers and optionally fine-tuned jointly.

The training objective generalizes **causal (CLM)** and **masked language modeling (MLM)** by varying the masking rate from partial (≈15%) to nearly full (≈90%) token masking. The diffusion process operates over **token positions**, not embedding noise — aligning closely with *Roberta-Diffusion* and *MMDiffBird* implementations.

---

## 2. Objectives and Rationale

| Goal | Description |
|------|--------------|
| **Simplify deep transformer training** | Enable incremental, layer-by-layer training with minimal memory pressure. |
| **Explore mask-diffusion supervision** | Replace fixed-mask MLM with variable masking schedules. |
| **Integrate modern attention variants** | Incorporate **Grouped Query Attention (GQA)** and **Multi-Latent Attention (MLA)** to reduce parameters and improve efficiency. |
| **Freeze redundant parameters** | Keep **up-projection matrices fixed and non-trainable**, mimicking low-rank factorization. |
| **Facilitate research extensibility** | Allow rapid experimentation with different attention, normalization, and diffusion strategies. |

---

## 3. Functional Requirements

### 3.1 Core Components
1. **Model Architecture**
   - Decoder-only Transformer (GPT-style).
   - Configurable number of layers, heads, and model width.
   - Options for *standard*, *GQA*, or *MLA* attention or both.
   - **RMSNorm** normalization and **SwiGLU** feed-forward modules.
   - Frozen up-projections in the MLP.
   - Rotary Positional Embeddings (RoPE) support.

2. **Layer-wise Supervised Training**
   - Train one layer (and embeddings, if layer-0) at a time.
   - Keep the embedding as well as last language head trainable.
   - Cache activations after each layer for reuse, after ths layer is trained
   - At any time only one layer's parameters are updated based on the activations from previous layers.

3. **Fusion Phase**
   - Sequentially merge layer checkpoints into a unified Transformer.
   - Optionally run a short **joint fine-tuning** stage.

4. **Mask-Diffusion Objective**
   - Randomly mask between *min_mask_fraction* and *max_mask_fraction* of tokens.
   - Supervise only masked tokens with cross-entropy.
   - Mask tokens, not embeddings — preserving original hidden states.

5. **Tied Embeddings**
   - Input and output embeddings share parameters.
   - Optional untied mode for ablation.

6. **Experiment Control**
   - YAML-based configuration.
   - Layer-wise checkpoints and activation caching after layer pre-training.
   - Dummy dataset for smoke testing.

---

## 4. Non-Functional Requirements

| Category | Requirement |
|-----------|--------------|
| **Scalability** | Handle up to 70B-parameter configurations using layer-wise staging. |
| **Reproducibility** | Deterministic initialization and fixed random up-projections. |
| **Extensibility** | Modular design supporting future additions like LoRA, MoE, or rotary embeddings. |
| **Efficiency** | Reduced optimizer state per layer, minimal GPU memory footprint. |
| **Transparency** | Easy checkpoint inspection, clear training logs per layer. |
| **Compatibility** | PyTorch ≥ 2.1, CUDA 12+, YAML config files, minimal dependencies. |

---

## 5. System Design and Flow

### 5.1 Training Stages
1. **Stage-0 (Embeddings + Layer-1):**
   - Train with diffusion mask objective.
   - Cache activations (acts_layer1.pt).

2. **Stage-k (Layer-k):**
   - Load cached activations from previous layer.
   - Train only the current layer’s parameters.

3. **Fusion Phase:**
   - Merge all layers sequentially.
   - Optionally fine-tune entire model on small dataset.

### 5.2 Data Flow
```
input_ids → Embeddings → Layer_1 → act_1
                          ↓
                        train Layer_2 → act_2
                          ↓
                        train Layer_3 → act_3
                          ↓
                        ...
fuse all layers → Joint Fine-Tuning → Final LM
```

---

## 6. Key Innovations

| Innovation | Description |
|-------------|-------------|
| **Layer-wise supervised learning** | Treat each layer as an independent supervised learner using cached activations. |
| **Diffusion-mask objective** | Variable masking rate generalizing MLM/CLM without embedding noise. |
| **GQA integration** | Shares K/V projections across head groups to reduce memory and parameters. |
| **MLA integration** | Low-rank factorization of Q/K/V projections, reducing FLOPs and cache footprint. |
| **Frozen up-projections** | Random but fixed linear projections prevent over-fitting while maintaining expressivity. These projections are used in LlaMA-3 architecture |

---

## 7. Example Configuration Parameters

| Parameter | Default | Description |
|------------|----------|--------------|
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
| `lr` | 1e-4 | Learning rate per layer |
| `batch_size` | 4 | Batch size |
| `seq_len` | 512 | Context length |

---

## 8. Deliverables

1. **Repository Scaffold**
   - `/src/model/core.py` — model definitions (MHA, GQA, MLA, SwiGLU, etc.)  
   - `/src/train_layerwise.py` — trainer logic  
   - `/src/data/dummy_dataset.py` — mock dataset for smoke tests  
   - `/config.yaml` — configuration template  
   - `/README.md` — documentation and usage guide  

2. **Artifacts**
   - Layer checkpoints (`layer_k.pt`)  
   - Cached activations (`acts_layerk.pt`)  
   - Final fused model checkpoint  

---

---

## 11. Future Extensions

1. **LoRA/Adapter Fusion** for parameter-efficient fine-tuning.  
2. **Mixture-of-Experts** layers within layer-wise training framework.  
3. **Hybrid diffusion objectives** operating jointly in embedding and token space.  
4. **Curriculum masking schedules** where masking rate increases over epochs.  
5. **Multilingual data handling** with shared embeddings and diffusion consistency loss.  

---

## 11. Licensing & Attribution

- Base implementation under **MIT License**.  
- Integrates conceptual ideas from:
  - *Roberta-Diffusion (Nathan)*  
  - *DeepSeek-V2/V3* (MLA formulation)  
  - *MMDiffBird (mlsquare)* internal research prototype.  

---

**End of SRD**
