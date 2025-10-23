# Layer‑Wise Transformer (Mask‑Diffusion Objective)

This repo is a **from‑scratch PyTorch scaffold** for training GPT‑2/LLaMA‑style decoder models **layer‑by‑layer** with:
- **Tied embeddings**
- **GQA** or **MLA** attention (select via config)
- **SwiGLU MLP** with **up‑projections frozen** (random, fixed)
- **Mask‑diffusion style objective**: randomly mask a variable fraction of tokens (e.g., 15% to 90%) and train to predict the masked tokens
- **Layer‑wise supervised training**: train layer 0 (with embeddings), cache activations, then train layer 1, …, fuse & (optionally) joint‑tune at the end

> Mask‑diffusion here means **masking tokens** (not adding Gaussian noise to embeddings). That generalizes **MLM/CLM** by varying the mask ratio and supervision policy.

## Features
- Decoder block with **RMSNorm + residuals**
- **Attention options**: standard MHA, **GQA**, **MLA** (low‑rank Q/K/V factors)
- **Frozen up‑projections** in SwiGLU
- **Layer‑wise activation caching** and training
- Final stage **fusion + joint tuning** (optional)

## Quickstart
```bash
# 1) Create/activate an environment with PyTorch >= 2.1
pip install -r requirements.txt

# 2) (Optional) Edit config.yaml to change width, heads, attention type, etc.
# 3) Run a smoke test with the dummy dataset (random token ids):
python -m src.train_layerwise --config config.yaml --dummy True --max_steps 50
```

## Configuration
Edit **config.yaml**. Key fields:
- `vocab_size` (default 128000)
- `d_model` (e.g., 4096 for 4k width)
- `n_layers` (# of decoder blocks)
- `n_heads`, `n_kv_heads` (for GQA/MLA), and `mla_rq`, `mla_rkv` (low‑rank sizes)
- `d_ff` (SwiGLU MLP width; e.g., 14336 ~ 3.5×d_model or ~2.5× for DeepSeek‑style)
- `attn_type`: `"standard"`, `"gqa"`, or `"mla"`
- `mask_fraction_min/max`: min/max masking ratio per batch (e.g., 0.15 … 0.90)
- `tie_embeddings`: typically `true`

## Notes
- **Frozen up‑projections** reduce trainable params and optimizer state; if you want partial adaptability, add LoRA or small gates later.
- **GQA** is most useful for decoder inference KV‑cache reduction; **MLA** can reduce parameters/compute and is essential in DeepSeek‑like stacks.
- The dummy dataset is just to validate the plumbing—replace it with your dataloader.
