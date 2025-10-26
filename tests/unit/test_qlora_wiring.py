


# Dynamically load the progressive_rack_builder module to avoid importing
# package-level modules that pull heavy optional dependencies at import-time.
import sys
import importlib
import os
from pathlib import Path

import torch
import torch.nn as nn


# Ensure repo root is on sys.path so relative imports inside src work
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Import builder through package import so relative imports resolve correctly
prb = importlib.import_module('src.training.progressive_rack_builder')
ProgressiveRackBuilder = prb.ProgressiveRackBuilder
import sys
import importlib
import os
from pathlib import Path

import torch
import torch.nn as nn


# Ensure repo root is on sys.path so relative imports inside src work
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Import builder through package import so relative imports resolve correctly
prb = importlib.import_module('src.training.progressive_rack_builder')
ProgressiveRackBuilder = prb.ProgressiveRackBuilder


class DummyBlock(nn.Module):
    def __init__(self, d_model: int = 16):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x


def count_linears(module: nn.Module) -> int:
    return sum(1 for _ in module.modules() if isinstance(_, nn.Linear))


class SimpleStack(nn.Module):
    """Minimal stack-like container with blocks list and modules()."""

    def __init__(self, blocks, stack_id=0):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.stack_id = stack_id

    def forward(self, x, attention_mask=None):
        out = x
        for b in self.blocks:
            out = b(out, attention_mask)
        return out


class DummyModelCfg:
    vocab_size = 1000
    d_model = 16
    d_ff = 64
    n_heads = 8
    tokenizer_embedding = {}


class DummyTrainCfg:
    def __init__(self):
        # Progressive settings are nested under training.progressive in the real config
        self.progressive = {
            'qlora_enabled': True,
            'qlora_rank': 2,
            'qlora_alpha': 4,
        }


class DummyConfig:
    def __init__(self):
        self.model = DummyModelCfg()
        self.training = DummyTrainCfg()

    def to_dict(self):
        return {}


def test_add_qlora_adapters_and_forward_effect():
    d_model = 16
    import sys
    from pathlib import Path
    import importlib

    import torch
    import torch.nn as nn

    # Ensure repo root is on sys.path so package imports resolve correctly
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    # Import builder through package import so relative imports resolve correctly
    prb = importlib.import_module('src.training.progressive_rack_builder')
    ProgressiveRackBuilder = prb.ProgressiveRackBuilder


    class DummyBlock(nn.Module):
        def __init__(self, d_model: int = 16):
            super().__init__()
            self.lin1 = nn.Linear(d_model, d_model)
            self.lin2 = nn.Linear(d_model, d_model)

        def forward(self, x, attention_mask=None):
            x = self.lin1(x)
            x = torch.relu(x)
            x = self.lin2(x)
            return x


    def count_linears(module: nn.Module) -> int:
        return sum(1 for _ in module.modules() if isinstance(_, nn.Linear))


    class SimpleStack(nn.Module):
        """Minimal stack-like container with blocks list and modules()."""

        def __init__(self, blocks, stack_id=0):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)
            self.stack_id = stack_id

        def forward(self, x, attention_mask=None):
            out = x
            for b in self.blocks:
                out = b(out, attention_mask)
            return out


    class DummyModelCfg:
        vocab_size = 1000
        d_model = 16
        d_ff = 64
        n_heads = 8
        tokenizer_embedding = {}


    class DummyTrainCfg:
        def __init__(self):
            # Progressive settings are nested under training.progressive in the real config
            self.progressive = {
                'qlora_enabled': True,
                'qlora_rank': 2,
                'qlora_alpha': 4,
            }


    class DummyConfig:
        def __init__(self):
            self.model = DummyModelCfg()
            self.training = DummyTrainCfg()

        def to_dict(self):
            return {}


    def test_add_qlora_adapters_and_forward_effect():
        d_model = 16
        batch = 2
        seq_len = 4

        # Build a simple stack with two dummy blocks
        blocks = [DummyBlock(d_model=d_model) for _ in range(2)]
        stack = SimpleStack(blocks, stack_id=0)

        # input
        x = torch.randn(batch, seq_len, d_model)

        # baseline output
        out_before = stack(x)

        # builder with dummy config
        builder = ProgressiveRackBuilder(config=DummyConfig())

        # add qlora adapters
        builder._add_qlora_to_stack(stack, 0, rank=2, alpha=4)

        # After adding adapters (initialized with zero up weights) output should be unchanged
        out_after = stack(x)
        assert torch.allclose(out_before, out_after, atol=1e-6)

        # Check adapter metadata exists
        meta = builder.qlora_adapters.get(0)
        assert meta is not None
        adapters = meta.get('adapters', {})
        # There should be as many adapters as Linear modules in the stack
        assert len(adapters) == count_linears(stack)

        # Now set adapter weights to non-zero and ensure output changes
        for info in adapters.values():
            adapter = info['adapter']
            # small non-zero init
            adapter.down.weight.data.fill_(0.01)
            adapter.up.weight.data.fill_(0.01)

        out_modified = stack(x)
        # Expect outputs to differ
        assert not torch.allclose(out_before, out_modified, atol=1e-6)


    def test_freeze_trunk_but_enable_qlora():
        d_model = 16
        blocks = [DummyBlock(d_model=d_model) for _ in range(1)]
        stack = SimpleStack(blocks, stack_id=0)

        builder = ProgressiveRackBuilder(config=DummyConfig())
        builder._add_qlora_to_stack(stack, 0, rank=2, alpha=4)

        # Freeze original params but enable qlora adapters
        builder._freeze_single_stack_but_qlora(stack, 0)

        # All original Linear parameters should be frozen
        for m in stack.modules():
            if isinstance(m, nn.Linear):
                # the linear might have a child adapter module; skip adapter params
                for name, p in m.named_parameters(recurse=False):
                    # parameters directly on the Linear (weight, bias) should be frozen
                    assert p.requires_grad is False

        # Adapter params should be trainable
        meta = builder.qlora_adapters.get(0)
        adapters = meta.get('adapters', {})
        assert len(adapters) > 0
        for info in adapters.values():
            adapter = info['adapter']
            for p in adapter.parameters():
                assert p.requires_grad is True

                d_model = 16
                d_ff = 64
                n_heads = 8
                tokenizer_embedding = {}


            class DummyTrainCfg:
                def __init__(self):
                    # Enable qlora for tests
                    self.progressive = {
                        'qlora_enabled': True,
                        'qlora_rank': 2,
                        'qlora_alpha': 4
                    }


            class DummyConfig:
                def __init__(self):
                    self.model = DummyModelCfg()
                    self.training = DummyTrainCfg()

                def to_dict(self):
                    return {}


            def test_add_qlora_adapters_and_forward_effect():
                d_model = 16
                batch = 2
                seq_len = 4

                # Build a simple stack with two dummy blocks
                blocks = [DummyBlock(d_model=d_model) for _ in range(2)]
                stack = SimpleStack(blocks, stack_id=0)

                # input
                x = torch.randn(batch, seq_len, d_model)

                # baseline output
                out_before = stack(x)

                # builder with dummy config
                builder = ProgressiveRackBuilder(config=DummyConfig())

                # add qlora adapters
                builder._add_qlora_to_stack(stack, 0, rank=2, alpha=4)

                # After adding adapters (initialized with zero up weights) output should be unchanged
                out_after = stack(x)
                assert torch.allclose(out_before, out_after, atol=1e-6)

                # Check adapter metadata exists
                meta = builder.qlora_adapters.get(0)
                assert meta is not None
                adapters = meta.get('adapters', {})
                # There should be as many adapters as Linear modules in the stack
                assert len(adapters) == count_linears(stack)

                # Now set adapter weights to non-zero and ensure output changes
                for info in adapters.values():
                    adapter = info['adapter']
                    # small non-zero init
                    adapter.down.weight.data.fill_(0.01)
                    adapter.up.weight.data.fill_(0.01)

                out_modified = stack(x)
                # Expect outputs to differ
                assert not torch.allclose(out_before, out_modified, atol=1e-6)


            def test_freeze_trunk_but_enable_qlora():
                d_model = 16
                blocks = [DummyBlock(d_model=d_model) for _ in range(1)]
                stack = SimpleStack(blocks, stack_id=0)

                builder = ProgressiveRackBuilder(config=DummyConfig())
                builder._add_qlora_to_stack(stack, 0, rank=2, alpha=4)

                # Freeze original params but enable qlora adapters
                builder._freeze_single_stack_but_qlora(stack, 0)

                # All original Linear parameters should be frozen
                for m in stack.modules():
                    if isinstance(m, nn.Linear):
                        # the linear might have a child adapter module; skip adapter params
                        for name, p in m.named_parameters(recurse=False):
                            # parameters directly on the Linear (weight, bias) should be frozen
                            assert p.requires_grad is False

                # Adapter params should be trainable
                meta = builder.qlora_adapters.get(0)
                adapters = meta.get('adapters', {})
                assert len(adapters) > 0
                for info in adapters.values():
                    adapter = info['adapter']
                    for p in adapter.parameters():
                        assert p.requires_grad is True
