"""Milestone 4: fine-tune only the LoRA adapter parameters on the commentary corpus."""
import json
import math
import random
from pathlib import Path

import torch

from app import lora


def encode_stream(lines: list[str], tok) -> list[int]:
    """One long token stream: each commentary line followed by the end-of-text token."""
    ids: list[int] = []
    for line in lines:
        ids.extend(tok.encode(line))
        ids.append(tok.eos_token_id)
    return ids


def make_blocks(ids: list[int], block_size: int, offset: int = 0) -> torch.Tensor:
    """Cut the stream into (N, block_size) training blocks, starting at `offset`."""
    ids = ids[offset:]
    n = len(ids) // block_size
    if n == 0:
        raise ValueError(f"corpus ({len(ids)} tokens) is shorter than one block ({block_size})")
    return torch.tensor(ids[: n * block_size]).view(n, block_size)


@torch.no_grad()
def eval_loss(model, tok, lines: list[str], block_size: int, device) -> float:
    """Mean next-token cross-entropy on held-out lines (nats / token). Natural log; perplexity = exp(loss)."""
    model.eval()
    blocks = make_blocks(encode_stream(lines, tok), block_size).to(device)
    losses = [model(b.unsqueeze(0), labels=b.unsqueeze(0)).loss.item() for b in blocks]
    return sum(losses) / len(losses)


def train_lora(model, tok, train_lines: list[str], val_lines: list[str], epochs: int = 8, batch_size: int = 8,
               lr: float = 5e-4, block_size: int = 64, seed: int = 0, device=None, log=print) -> dict:
    """Train adapter params only. Returns {"step_loss": [...], "epoch_loss": [...], "val_loss": [...]}; val_loss[0]
    is measured *before* training, i.e. the untouched base model (adapter B is zero-initialised)."""
    device = device or next(model.parameters()).device
    lora.assert_only_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    stream = encode_stream(train_lines, tok)
    model.to(device)
    hist = {"step_loss": [], "epoch_loss": [], "val_loss": [eval_loss(model, tok, val_lines, block_size, device)]}
    log(f"  epoch 0/{epochs}  val_loss {hist['val_loss'][0]:.3f}  (base model)")
    for epoch in range(1, epochs + 1):
        model.train()
        blocks = make_blocks(stream, block_size, offset=rng.randrange(block_size))  # new block boundaries each epoch
        order = list(range(len(blocks)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            batch = blocks[order[i:i + batch_size]].to(device)
            loss = model(batch, labels=batch).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
            hist["step_loss"].append(loss.item())
        hist["epoch_loss"].append(sum(losses) / len(losses))
        hist["val_loss"].append(eval_loss(model, tok, val_lines, block_size, device))
        log(f"  epoch {epoch}/{epochs}  train_loss {hist['epoch_loss'][-1]:.3f}  val_loss {hist['val_loss'][-1]:.3f}")
    model.eval()
    return hist


def render_loss_curve(values: list[float], width: int = 64, height: int = 10, label: str = "train loss") -> str:
    """A small ASCII line chart so the loss curve can be shown in a terminal / markdown code block."""
    n_steps = len(values)
    if len(values) > width:  # average into `width` buckets
        step = len(values) / width
        values = [sum(values[int(i * step):max(int((i + 1) * step), int(i * step) + 1)]) /
                  len(values[int(i * step):max(int((i + 1) * step), int(i * step) + 1)]) for i in range(width)]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    rows = [[" "] * len(values) for _ in range(height)]
    for x, v in enumerate(values):
        y = int(round((v - lo) / span * (height - 1)))
        rows[height - 1 - y][x] = "*"
    out = [f"{label}"]
    for i, row in enumerate(rows):
        val = hi - span * i / (height - 1)
        out.append(f"{val:6.2f} | " + "".join(row))
    out.append("       +" + "-" * len(values))
    out.append(f"        step 1{' ' * max(1, len(values) - 6 - len(str(n_steps)))}step {n_steps}")
    return "\n".join(out)


def save_training_log(hist: dict, summary: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_log.json").write_text(json.dumps({**summary, **hist}, indent=2))
    (out_dir / "loss_curve.txt").write_text(render_loss_curve(hist["step_loss"]) + "\n")


def perplexity(loss: float) -> float:
    return math.exp(loss)
