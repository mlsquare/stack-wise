# src/train_layerwise.py
import argparse, yaml, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model.core import ModelConfig, LayeredDecoder
from src.data.dummy_dataset import DummyTokenDataset

SPECIAL_MASK_ID = 4  # placeholder; ensure tokenizer reserves it

def random_mask(input_ids, mask_fraction):
    B,T = input_ids.shape
    mask = (torch.rand(B, T, device=input_ids.device) < mask_fraction).long()
    masked = input_ids.clone()
    masked[mask==1] = SPECIAL_MASK_ID
    return masked, mask

def masked_diffusion_loss(model, h, input_ids, mask):
    logits = model.output_logits(h)               # (B,T,V)
    loss_all = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1),
        reduction="none"
    ).view_as(input_ids)
    loss = (loss_all * mask).sum() / mask.sum().clamp(min=1)
    return loss

def set_trainable_layer(model: LayeredDecoder, k: int):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.blocks[k].parameters(): p.requires_grad_(True)
    if k == 0:
        for p in model.tok.parameters(): p.requires_grad_(True)

def main(args):
    with open(args.config, "r") as f:
        cfgd = yaml.safe_load(f)
    cfg = ModelConfig(**cfgd)
    device = cfgd.get("device","cuda" if torch.cuda.is_available() else "cpu")
    model = LayeredDecoder(cfg).to(device)

    if args.dummy:
        ds = DummyTokenDataset(num_samples=128, seq_len=cfgd["seq_len"], vocab_size=cfgd["vocab_size"])
    else:
        raise NotImplementedError("Plug your real dataset here.")
    dl = DataLoader(ds, batch_size=cfgd["batch_size"], shuffle=True, drop_last=True)

    # Stage 0
    k = 0
    set_trainable_layer(model, k)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfgd["lr"], betas=tuple(cfgd["betas"]), weight_decay=cfgd["weight_decay"])
    steps, max_steps = 0, int(args.max_steps or cfgd["max_steps"])
    model.train()
    cached_acts = []

    it = iter(dl)
    while steps < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        input_ids = batch["input_ids"].to(device)
        prev_act = model.tok(input_ids)
        h_k = model.forward_layer(prev_act, k, attn_mask=None)
        frac = float(cfg.mask_fraction_min + (cfg.mask_fraction_max - cfg.mask_fraction_min) * torch.rand(1).item())
        masked_ids, mask = random_mask(input_ids, frac)
        loss = masked_diffusion_loss(model, h_k, input_ids, mask)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        steps += 1
        if steps % 10 == 0:
            print(f"[layer {k}] step {steps} loss={loss.item():.4f} (mask_frac={frac:.2f})")
        cached_acts.append(h_k.detach().cpu())

    act0_path = "acts_layer0.pt"
    torch.save(torch.cat(cached_acts, dim=0), act0_path)
    torch.save(model.blocks[0].state_dict(), "layer_0.pt")
    print(f"Saved layer_0 and cached activations → {act0_path}")

    # Stage 1 (demo)
    if cfg.n_layers > 1:
        acts = torch.load(act0_path, map_location="cpu")
        dl2 = DataLoader(acts, batch_size=cfgd["batch_size"], shuffle=True, drop_last=True)
        k = 1
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.blocks[k].parameters(): p.requires_grad_(True)
        opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfgd["lr"], betas=tuple(cfgd["betas"]), weight_decay=cfgd["weight_decay"])
        steps = 0
        it2 = iter(dl2); it = iter(dl)
        while steps < max_steps:
            try:
                prev_act = next(it2)
            except StopIteration:
                it2 = iter(dl2); prev_act = next(it2)
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            prev_act = prev_act.to(device)
            input_ids = batch["input_ids"].to(device)
            h_k = model.forward_layer(prev_act, k, attn_mask=None)
            frac = float(cfg.mask_fraction_min + (cfg.mask_fraction_max - cfg.mask_fraction_min) * torch.rand(1).item())
            masked_ids, mask = random_mask(input_ids, frac)
            loss = masked_diffusion_loss(model, h_k, input_ids, mask)
            opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()
            steps += 1
            if steps % 10 == 0:
                print(f"[layer {k}] step {steps} loss={loss.item():.4f} (mask_frac={frac:.2f})")
        torch.save(model.blocks[k].state_dict(), f"layer_{k}.pt")
        print(f"Saved layer_{k}.pt")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--dummy", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=True)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    main(args)
