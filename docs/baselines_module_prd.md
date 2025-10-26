# Baselines Module PRD: Encoder–Decoder Families & Scaling-Benchmark Framework

---

## 1️⃣ Objective

Establish a **baseline and scaling benchmark module** within the main framework (`baselines/`) to validate the *depth-as-time* and *curriculum-flexible* architecture against well-known encoder (BERT-family) and decoder (GPT-family) models under comparable compute budgets.

The module enables reproducible scaling, benchmarking, and cross-family evaluation at multiple model sizes — from Tiny to Large.

---

## 2️⃣ Scope

| Family | Type | Reference Models | Goal |
|--------|------|------------------|------|
| **Encoders** | Bidirectional / Masked | Tiny-BERT, MiniLM, DistilBERT, ModernBERT, BERT-base/large | Evaluate embedding quality & NLU tasks |
| **Decoders** | Autoregressive / Causal | Tiny-GPT, GPT-2 (124M–1.5B), LLaMA family | Evaluate text generation, reasoning, perplexity |
| **Unified / Diffusion** | Mask-Diffusion (depth-as-time) | Compare scaling laws vs classical baselines |

---

## 3️⃣ Design Goals

1. **Fair scaling** — match models by compute/token budgets, not just parameter count.  
2. **Composable configs** — per-family YAML configs defining width, depth, heads, FFN ratio, mask ratio.  
3. **Unified trainers** — both *classical* and *depth-as-time* training regimes selectable via flag.  
4. **Standardized benchmarks** — unified evaluation harness for NLU and NLG.  
5. **Extensible registry** — easy to add new model families, configs, and benchmarks.

---

## 4️⃣ Directory Structure

```
/baselines/
 ├── encoder/
 │   ├── bert_family/
 │   │   ├── tiny.yaml
 │   │   ├── base.yaml
 │   │   ├── large.yaml
 │   │   └── modern.yaml
 │   └── modernbert_family/
 ├── decoder/
 │   ├── gpt2_family/
 │   │   ├── nano.yaml
 │   │   ├── small.yaml
 │   │   ├── medium.yaml
 │   │   ├── large.yaml
 │   │   └── xl.yaml
 │   └── llama_family/
 ├── trainer/
 │   ├── classical_trainer.py
 │   └── depth_time_trainer.py
 ├── benchmarks/
 │   ├── datasets.yaml
 │   └── evaluate.py
 └── configs/
     └── scaling/
         ├── nano.yaml
         ├── small.yaml
         ├── base.yaml
         ├── large.yaml
         └── xl.yaml
```

---

## 5️⃣ Training Regimes

| Regime | Description |
|---------|-------------|
| **Classic Pre-training** | Standard masked LM (encoder) or causal LM (decoder). |
| **Depth-as-Time** | Layer-wise diffusion-style denoising curriculum (Right→Left). |
| **Hybrid** | Fine-tune classical checkpoint under diffusion curriculum. |

Command-line usage:
```
train.py --family bert --config base.yaml --curriculum right2left --noise-schedule cosine
```

---

## 6️⃣ Model Scaling

| Scale | Params | Layers | d_model | FFN ratio | Purpose |
|-------|---------|---------|----------|------------|----------|
| Nano | 15M | 4 | 256 | 2 | Sanity checks |
| Tiny | 40M | 6 | 384 | 3 | Light prototype |
| Small | 80M | 12 | 512 | 4 | Tiny-BERT / GPT-2-small equivalent |
| Base | 120M | 12 | 768 | 4 | Reference baseline |
| Large | 350M–1B | 24–36 | 1024–1536 | 4 | High-capacity proof |
| XL | 1–3B+ | 48+ | 2048+ | 4 | Stretch-goal (HPC) |

---

## 7️⃣ Benchmarks

### Encoder Tasks (NLU)

| Task | Dataset | Metric |
|-------|----------|--------|
| Semantic Textual Similarity | STS-B | Spearman / Pearson |
| Natural Language Inference | MNLI, SNLI | Accuracy |
| Question Answering | SQuAD v1/v2 | F1 / EM |
| Sentiment | SST-2 | Accuracy |
| Named Entity Recognition | CoNLL-2003 | F1 |

### Decoder Tasks (NLG)

| Task | Dataset | Metric |
|-------|----------|--------|
| Language Modeling | WikiText-103 | Perplexity |
| QA / Completion | LAMBADA | Accuracy |
| Reasoning | PIQA, HellaSwag | Accuracy |
| Summarization | CNN/DailyMail | ROUGE-L |
| Translation | WMT-14 (en-de) | BLEU |

---

## 8️⃣ Compute & Budgeting

- Equalize **training FLOPs**: `FLOPs = tokens × FLOPs/token × epochs`  
- Define per-scale compute targets.  
- Auto-adjust steps and batch size to maintain equal compute.  
- Support mixed precision (bf16/fp16) via `accelerate` or `DeepSpeed`.

---

## 9️⃣ Evaluation Harness

```
evaluate.py --model bert-base --tasks mnli,sst2,stsb
evaluate.py --model gpt2-small --tasks wikitext103,hellaswag
```
Features:
- Auto dataset loading (via HuggingFace `datasets`).  
- Tokenizer auto-select by family.  
- Unified metric interface.  
- Logs to TensorBoard / Weights & Biases.  
- Supports zero-shot + fine-tuned eval.  
- Embedding extraction API for STS tasks.

---

## 🔟 Integration with Depth-as-Time

- Compatible with the `DenoiserStack` architecture.  
- Encoder families can use *mask-diffusion objectives* (conditional denoising).  
- Decoder families can use *autoregressive unmasking schedules*.  
- Depth-as-time mode implemented as `depth_time_trainer.py` in `/trainer/`.

---

## 11️⃣ Milestones

| Phase | Deliverable | Duration |
|-------|--------------|----------|
| **P1** | Tiny-BERT & Tiny-GPT prototypes, single-node runs | 2–3 weeks |
| **P2** | Base & Small models across both families, full benchmarks | 4–6 weeks |
| **P3** | Integrate depth-as-time variants, scaling-law plots | 3 weeks |
| **P4** | Comparative report: compute vs performance | 2 weeks |

---

## 12️⃣ Metrics of Success

- Tiny/Base models achieve within **5–10%** of standard baselines (BERT-base, GPT-2-small).  
- Scaling-law exponents consistent with known literature (α ≈ 0.05–0.10).  
- Depth-as-time variants maintain comparable downstream performance with reduced compute.  

---

## 13️⃣ Repository Decision

Start as **`baselines/` module inside main framework**, not a separate repo.

**Pros:**
- Reuse shared components (`DenoiserStack`, trainers, datasets).  
- Easier integrated experiments across curricula.  
- Central logging & evaluation utilities.

**Later Option:**
Export as standalone repo (`model-baselines/`) for publication or public benchmarking once stable.

---

## ✅ Summary

This module ensures:
- Comparable encoder & decoder baselines under equal compute.  
- Joint benchmarking for *classical* and *depth-as-time* regimes.  
- Unified configuration, dataset, and evaluation registry.  
- Reproducible scaling studies to validate model performance across scales.

---

*(End of PRD)*
