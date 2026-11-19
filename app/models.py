"""Model + tokenizer loading. The real run uses distilgpt2; tests/--tiny use a random 2-layer GPT-2 and a byte tokenizer
so everything works offline."""
import os

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def pick_device() -> torch.device:
    """MPS on Apple Silicon when available, otherwise CPU. Set STG_DEVICE=cpu (or mps/cuda) to override."""
    if os.environ.get("STG_DEVICE"):
        return torch.device(os.environ["STG_DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ByteTokenizer:
    """Byte-level tokenizer: ids 0-255 are bytes, 256 is end-of-text. Same encode/decode API as a HF tokenizer."""
    vocab_size = 257
    eos_token_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids if int(i) < 256).decode("utf-8", errors="replace")


def tiny_model(seed: int = 0) -> tuple[GPT2LMHeadModel, ByteTokenizer]:
    tok = ByteTokenizer()
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=128, vocab_size=tok.vocab_size,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return GPT2LMHeadModel(cfg).eval(), tok


def load_pretrained(name: str = "distilgpt2"):
    """Download (first run only) and load a small pretrained causal LM + its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model.eval(), tok


def load_model(name: str = "distilgpt2"):
    """`name == "tiny"` gives the offline test model; anything else is a Hugging Face model id."""
    return tiny_model() if name == "tiny" else load_pretrained(name)
