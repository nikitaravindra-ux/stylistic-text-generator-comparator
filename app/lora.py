"""Milestone 3: LoRA written by hand (no peft).

LoRA freezes a pretrained weight W and learns a low-rank update:   y = x W + (alpha / r) * x A^T B^T
with A: (r, in_features) and B: (out_features, r). B starts at zero, so the adapted model is *identical* to the
base model before training.

GPT-2 gotcha: its attention projections (c_attn, c_proj) are transformers' `Conv1D`, not `nn.Linear`. Conv1D stores
its weight as (in_features, out_features) and computes `x @ W + b` -- the transpose of nn.Linear. We therefore read
the feature sizes according to the layer type and keep A/B in the *logical* (out, in) orientation; the low-rank
update is added to the layer's *output*, so it is correct for both layouts. `delta_weight` returns the update in
the layer's own storage layout so merging / testing is also correct.
"""
import contextlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

DEFAULT_TARGETS = ("c_attn", "c_proj")  # attention qkv + attention output (+ MLP output, which is also named c_proj)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if isinstance(base, Conv1D):
            self.is_conv1d = True
            in_features, out_features = base.weight.shape          # Conv1D: (in, out)
        elif isinstance(base, nn.Linear):
            self.is_conv1d = False
            out_features, in_features = base.weight.shape          # nn.Linear: (out, in)
        else:
            raise TypeError(f"cannot wrap {type(base).__name__}; expected nn.Linear or Conv1D")
        self.base, self.r, self.alpha = base, r, alpha
        self.scaling = alpha / r
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The update scaling * B @ A, in the same layout as base.weight (transposed for Conv1D)."""
        delta = (self.lora_B @ self.lora_A) * self.scaling         # (out, in)
        return delta.T if self.is_conv1d else delta


def inject_lora(model: nn.Module, target_modules=DEFAULT_TARGETS, r: int = 8, alpha: float = 16.0,
                dropout: float = 0.05) -> list[str]:
    """Freeze every base weight, then wrap each matching Linear/Conv1D in a LoRALinear. Returns wrapped names."""
    for p in model.parameters():
        p.requires_grad = False
    targets = set(target_modules)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, (nn.Linear, Conv1D)):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    if not replaced:
        raise ValueError(f"no modules named {sorted(targets)} found to adapt")
    model.lora_config = {"target_modules": sorted(targets), "r": r, "alpha": alpha, "dropout": dropout}
    return replaced


def lora_layers(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def assert_only_lora_trainable(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        is_lora = "lora_" in name
        if p.requires_grad != is_lora:
            raise AssertionError(f"{name}: requires_grad={p.requires_grad}, expected {is_lora}")


def trainable_summary(model: nn.Module) -> str:
    trainable, total = count_parameters(model)
    return f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)"


@contextlib.contextmanager
def adapters_disabled(model: nn.Module):
    """Temporarily bypass every adapter, i.e. run the untouched base model."""
    layers = lora_layers(model)
    before = [l.enabled for l in layers]
    for l in layers:
        l.enabled = False
    try:
        yield model
    finally:
        for l, e in zip(layers, before):
            l.enabled = e


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(model: nn.Module, path: str | Path) -> None:
    """Save only the adapter tensors plus the config needed to rebuild them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.lora_config, "state": lora_state_dict(model)}, path)
    path.with_suffix(".json").write_text(json.dumps(model.lora_config, indent=2))


def load_lora(model: nn.Module, path: str | Path) -> nn.Module:
    """Inject adapters (using the saved config) into a fresh base model and load the saved tensors."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not lora_layers(model):
        cfg = blob["config"]
        inject_lora(model, cfg["target_modules"], cfg["r"], cfg["alpha"], cfg["dropout"])
    missing, unexpected = model.load_state_dict(blob["state"], strict=False)
    if unexpected or any("lora_" in k for k in missing):
        raise RuntimeError(f"adapter does not match model: missing={missing[:3]} unexpected={unexpected[:3]}")
    device = next(model.parameters()).device
    return model.to(device)
