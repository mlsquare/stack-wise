# src/data/dummy_dataset.py
import torch
from torch.utils.data import Dataset

class DummyTokenDataset(Dataset):
    def __init__(self, num_samples=64, seq_len=512, vocab_size=128000, pad_id=0):
        self.N = num_samples; self.T = seq_len; self.V = vocab_size; self.pad_id = pad_id
    def __len__(self): return self.N
    def __getitem__(self, idx):
        x = torch.randint(5, self.V, (self.T,), dtype=torch.long)  # avoid tiny ids
        return {"input_ids": x, "labels": x.clone(), "attn_mask": None}
