"""Milestone 5: our own autoregressive sampling loop (temperature + top-k), not `model.generate`."""
import torch


@torch.no_grad()
def generate(model, tok, prompt: str = "", max_new_tokens: int = 40, temperature: float = 0.8, top_k: int = 40,
             seed: int | None = 0) -> str:
    """Sample a continuation of `prompt`. Stops at the end-of-text token (the fine-tuned model learns to emit one at
    the end of each commentary line). Deterministic for a fixed seed. Returns only the new text."""
    model.eval()
    device = next(model.parameters()).device
    ids = tok.encode(prompt) if prompt else []
    if not ids:
        ids = [tok.eos_token_id]  # start-of-text
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0 if seed is None else seed)
    max_ctx = model.config.n_positions
    new: list[int] = []
    for _ in range(max_new_tokens):
        window = torch.tensor([(ids + new)[-max_ctx:]], device=device)
        logits = model(window).logits[0, -1].float().cpu()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        if nxt == tok.eos_token_id:
            break
        new.append(nxt)
    return tok.decode(new).strip()
