# StackWise Unified Framework — Conceptual Synthesis

## 1️⃣ Time as the Organizing Axis

StackWise training is governed by **normalized time**
`t ∈ [0,1]` (or equivalently, `t ∈ [0,T]` for real training steps).
Time is **continuous**, but the system operates in discrete causal steps.

At each `t`, the framework produces a **complete training state** consisting of:
- the active **task** (what objective we’re solving),
- the **data configuration** (dataset, transforms, samplers),
- the **model configuration** (embedding block, rack, head),
- and the **optimizer state** (learning dynamics).

---

## 2️⃣ The Three Training Planes

| Plane | Responsibility | What evolves | Runs on |
|--------|----------------|---------------|---------|
| **Data Plane** | Defines the input distribution and how data are presented to the model. | Dataset selection, data transforms, masking, corruption, batching, augmentation. | CPU / data pipeline nodes. |
| **Model Plane** | Defines the architecture and parameter space being trained. | Stack evolution inside the rack, frozen/unfrozen regions, adapter/head state. | GPU / accelerator. |
| **Optimizer Plane** | Defines the dynamics of learning applied to the model. | Learning rate, scheduler, parameter groups, optimizer type, momenta. | GPU + control logic. |

All three planes are **time-indexed** and **task-aware**, i.e. `Data(t, task_t)`, `Model(t, task_t)`, and `Optimizer(t, Model_t, task_t)`.

---

## 3️⃣ The Causal Path at Time *t*

1. **Select task:** determine the active objective (e.g. MLM, diffusion, seq2seq, alignment).  
2. **Prepare data plane:** build a time-aware, task-aware data loader.  
3. **Prepare model plane:** assemble the model as `Embedding Block → Rack → Head`.  
4. **Prepare optimizer plane:** generate optimizer state reactive to model size and task difficulty.  
5. **Training step:** sample a batch, forward pass, compute loss, and update.

---

## 4️⃣ Structural Hierarchy of the Model Plane

```
Embedding Block → Rack (Stack[Block₁, Block₂, …, Blockₙ]) → Head
```

- **Embedding Block:** input-side model converting raw data to embeddings.
- **Rack:** evolving sequence of stacks; each stack is an ordered collection of atomic blocks.
- **Head:** output-side model projecting internal representations to task outputs.

The **Rack** evolves continuously in time, while the **Embedding Block** and **Head** evolve discretely at task or modality boundaries.

---

## 5️⃣ Evolution Over Time

| Aspect | Description |
|--------|--------------|
| **Data evolution** | Dataset and transform schedule — what data and corruption level are shown. |
| **Model evolution** | Architectural growth within the rack — how many blocks, which are trainable. |
| **Optimizer evolution** | Learning dynamics — how fast, how aggressive, which params are controlled. |

Each plane evolves independently but remains synchronized through normalized time.

---

## 6️⃣ Segmented Time / Multi-Phase Training

The global timeline can be segmented into phases `[0, t₁), [t₁, t₂), …, [t_{K-1}, 1]`,
where each phase has its own task, data strategy, and optimizer regime.

---

## 7️⃣ Causal Alignment Across Planes

At any instant, the triplet `(Data_t, Model_t, Optimizer_t)` represents the full *training state*.

---

## 8️⃣ Conceptual Picture

- **Time (t):** the global scheduler.  
- **Data plane:** what the model sees.  
- **Model plane:** what the model is.  
- **Optimizer plane:** how the model learns.  

Together, they form a *causal training machine* that evolves the model smoothly through time.
Structurally, each model snapshot at time `t` is `Embedding Block + Rack + Head`.
