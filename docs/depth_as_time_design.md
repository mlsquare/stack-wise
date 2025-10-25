# Depth-as-Time: Design Options & Coding Checklist

This document summarizes the two **training curricula** and the two **time/noise schedules**, and provides code-ready checklists for data prep, training, and sampling in PyTorch.

---

## 0) Notation

- Block ordered **left → right**: `Block 1, Block 2, …, Block L`.
- Mask ratio / noise level per block: `r_k`, decreasing left→right.
- Clean tokens: `x0`; corrupted tokens: `x(r)` or `x_t`.
- Shared LM head tapped at any depth unless stated otherwise.

---

## 1) Axes of Choice

### A. Curriculum Direction (Model Growth)

1. **Left→Right (capacity growth)**
   - Start with a shallow stack; **append** blocks.
   - All blocks can see **all** noise levels or a fixed `r`.
   - Final representation may **drift** unless the head is anchored.

2. **Right→Left (reverse-diffusion curriculum)**
   - Train **rightmost block** `L` on **light** noise (e.g., `r_L ≈ 0.15`).
   - **Prepend** `L-1` with **heavier** noise (`r_{L-1} > r_L`); **freeze** suffix `L..` as semantic anchor.
   - Each new block learns to map noisier inputs into the **fixed downstream** latent manifold.

> In both curricula, **leftmost** blocks see the **highest** noise; **rightmost** the **lowest**.

---

### B. Time/Noise Parameterization

1. **Time-as-Input (standard diffusion)**
   - One denoiser \(f_\theta(x_t,t)\) shared across all timesteps.
   - Each batch samples a random `t`; explicit time embedding.
   - Depth increases **capacity**, not discrete time resolution.

2. **Time-as-Depth (depth = reverse time)**
   - Tie noise schedule to block ID: `Block k ↔ r_k`.
   - No explicit `t` input required.
   - A single forward pass performs the entire **reverse diffusion** trajectory.
   - Works best with **Right→Left** curriculum.

---

## 2) Schedules & Mapping

Example schedule (L=6):
```
r = [1.00, 0.75, 0.55, 0.40, 0.25, 0.15]
```

Mapping:
- **Depth→Noise:** `Block k` ↔ noise `r_k` (Time-as-Depth)
- **Input-Time:** random `t` per batch (Time-as-Input)

---

## 3) Data Corruption (Token Space)

- Corrupt with **masking** at ratio `r`: replace selected tokens with `<MASK>` or Gaussian-noised embeddings.
- Compute loss **only** on masked positions.
- Use deterministic masks via `(sample_id, r, seed)` for reproducibility.

---

## 4) Training Recipes

### 4.1 Right→Left + Time-as-Depth (recommended for large models)

**Goal:** progressive reverse-diffusion unroll with fixed semantic anchor.

For `k = L → 1`:
1. **Freeze** suffix `blocks[k+1..L]` + LM head.  
2. Corrupt inputs with mask ratio `r_k`.  
3. Train **Block k** so full subnetwork `k..L` reconstructs masked tokens.  
4. Optionally add **consistency loss** between post-Block-k hidden and expected suffix input.  
5. Use LoRA on frozen suffix for tiny alignment (optional).

**Stability:**
- Spectral norm on linears (`Wq, Wk, Wv, Wo, W1, W2`).
- Residual gain `γ∈[0.3,0.7]`.
- Grad clip (1–5).
- Quantize frozen suffix (QLoRA).

---

### 4.2 Left→Right + Time-as-Input

**Goal:** classical diffusion-style denoising with shared time embedding.

For `d = 1 → L`:
1. Add new block; optionally freeze earlier ones.  
2. Sample `t ~ U(0,1)` per batch; corrupt accordingly.  
3. Train to predict noise (MSE) or clean tokens (CE).  
4. Periodically fine-tune jointly for continuity.

---

### 4.3 Hybrid

- **Right→Left + Time-as-Input:** each block sees narrow band of `t` near `r_k`.
- **Left→Right + Time-as-Depth:** rare; tie depth to noise but grow forward.

---

## 5) Sampling (Generation)

### 5.1 Time-as-Depth (single forward unroll)

- Start with **all masks**.
- For `k = 1 → L` (high→low noise):
  1. Run block `k` (or suffix `k..L`).
  2. Read logits; unmask top/confident tokens.
  3. Write back (self-conditioning).
  4. Stop early if no masks remain.

> Use suffix execution at first; step-local after consistency training.

### 5.2 Time-as-Input (iterative schedule)

- Start all masks.
- For `t_L → t_0`:
  - Run shared denoiser with `(x_t, t)`.
  - Update masked tokens.
  - Repeat.

---

## 6) Pros & Cons

| Aspect | Right→Left + Depth=Time | Left→Right + Time-as-Input |
|---|---|---|
| **Endpoint semantics** | Fixed (frozen head) | May drift |
| **Continuity in time** | Architectural (per-block) | Learned via shared weights |
| **Memory & scale** | Train one block at a time | Full/partial joint training |
| **Sampling** | One forward pass | Multiple time iterations |
| **Interpretability** | Block = reverse step | Depth = capacity |
| **Ease** | Needs prepend logic | Standard sequential |

---

## 7) PyTorch Structure

```python
class DenoiserStack(nn.Module):
    def __init__(self, d_model, vocab_size, tok_embed):
        super().__init__()
        self.tok_embed = tok_embed
        self.blocks = nn.ModuleList([])  # left→right
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # shared LM head

    def prepend_block(self, block):       # right→left
        self.blocks.insert(0, block)

    def freeze_suffix(self, start_idx):   # freeze blocks[start_idx:]
        for b in self.blocks[start_idx:]:
            for p in b.parameters():
                p.requires_grad = False

    def forward_from(self, h, start_idx=0, end_idx=None, attn_mask=None):
        if end_idx is None: end_idx = len(self.blocks)
        for b in self.blocks[start_idx:end_idx]:
            h = b(h, attn_mask=attn_mask)
        return h

    def forward(self, input_ids, start_idx=0, end_idx=None, attn_mask=None):
        h = self.tok_embed(input_ids)
        h = self.forward_from(h, start_idx, end_idx, attn_mask)
        h = self.ln_out(h)
        logits = self.head(h)
        return logits, h
```

### Training (Right→Left)
```python
# Train block k with suffix frozen
model.freeze_suffix(start_idx=1)
for batch in loader_k:
    mask = gen_mask(batch.ids, r_k, seed_fn(batch.ids, r_k))
    corrupted = apply_mask(batch.ids, mask, r_k)
    logits, _ = model(corrupted)
    loss = ce(logits[mask], batch.ids[mask])
    loss.backward(); opt.step(); opt.zero_grad()
```

---

## 8) Stability & Regularization

- Spectral norm on linears (n_power_iterations=1).  
- Residual gain γ per block.  
- Grad clip 1–5.  
- Consistency loss between block output and suffix expectation.  
- QLoRA for frozen suffix.  
- Deterministic masks `(sample_id, r, seed)`.

---

## 9) Checklists

### Build (Right→Left + Depth=Time)
- [ ] Define `r_k` schedule.  
- [ ] Train Block L + head on `r_L`; freeze.  
- [ ] `prepend_block(L-1)`; freeze suffix; train on `r_{L-1}`.  
- [ ] Repeat to Block 1.  
- [ ] Optional consistency or adapter fine-tune.

### Sample (Depth=Time)
- [ ] Init all masks.  
- [ ] For k=1→L: run suffix `k..L`; unmask; write back; early-exit.

### Build (Left→Right + Time-as-Input)
- [ ] Add blocks over time; all see `t` embedding.  
- [ ] Train on random `t` per batch.  
- [ ] Periodic joint fine-tune; anchor head if needed.

---

## 10) When to Pick Which

| Goal | Best Choice |
|------|--------------|
| **Train massive model with limited memory; stable endpoint** | Right→Left + Depth=Time (with QLoRA & spectral norm) |
| **Classical diffusion modeling; shared denoiser** | Left→Right + Time-as-Input |

---

*This guide provides the structure, schedules, and PyTorch patterns needed to implement both “depth-as-time” and “time-as-input” variants of layer-wise diffusion Transformers.*
