"""One function per milestone. Each is a thin orchestrator over a focused module:

    1 corpus.py    2 ngram.py    3 lora.py + models.py    4 finetune.py    5 generate.py    6 compare.py
"""
from pathlib import Path

from app import corpus
from app import compare, finetune, generate as gen, lora
from app.models import load_model, pick_device
from app.ngram import NGramModel

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "corpus" / "raw"
CLEAN_PATH = ROOT / "corpus" / "clean.txt"
OUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
ADAPTER_PATH = OUT_DIR / "adapter" / "lora_adapter.pt"

# --- Milestone 1: Prepare a small corpus of sports commentary text.
# Hint: Focus on collecting and cleaning a relevant text dataset.
def milestone_1(raw_dir=RAW_DIR, out_path=CLEAN_PATH, stats_path=OUT_DIR / "corpus_stats.json") -> dict:
    """Clean the raw transcripts into corpus/clean.txt and return corpus stats (lines, tokens, vocab size)."""
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    return corpus.build_corpus(raw_dir, out_path, stats_path)


# --- Milestone 2: Implement the n-gram model and generate text from it.
# Hint: Apply the n-gram modeling technique from the drill to build your model and generate text.
def milestone_2(n: int = 3, k: float = 0.05, corpus_path=CLEAN_PATH) -> NGramModel:
    """Fit an n-gram model on the training split of the cleaned corpus (the LoRA model trains on the same split)."""
    train, _ = corpus.split_lines(corpus.load_corpus(corpus_path))
    return NGramModel(n=n, k=k).fit(train)


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
# Hint: Use the technique from the drill to inject LoRA adapters into a pre-trained LLM.
def milestone_3(model_name: str = "distilgpt2", r: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                targets=lora.DEFAULT_TARGETS):
    """Load a small pretrained LLM and inject hand-written LoRA adapters. Returns (model, tokenizer)."""
    model, tok = load_model(model_name)
    lora.inject_lora(model, targets, r=r, alpha=alpha, dropout=dropout)
    lora.assert_only_lora_trainable(model)  # base frozen, adapters trainable -- fail loudly otherwise
    return model, tok


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
# Hint: Train the LoRA adapters on your prepared corpus, following the fine-tuning process from the drill.
def milestone_4(model, tok, epochs: int = 8, batch_size: int = 8, lr: float = 5e-4, block_size: int = 64, seed: int = 0,
                corpus_path=CLEAN_PATH, adapter_path=ADAPTER_PATH, out_dir=OUT_DIR, device=None, log=print) -> dict:
    """Fine-tune the adapter (only) on the training split, save the adapter + training log, return the loss history."""
    train, val = corpus.split_lines(corpus.load_corpus(corpus_path))
    trainable, total = lora.count_parameters(model)
    log(lora.trainable_summary(model))
    hist = finetune.train_lora(model, tok, train, val, epochs=epochs, batch_size=batch_size, lr=lr,
                               block_size=block_size, seed=seed, device=device, log=log)
    lora.save_lora(model, adapter_path)
    summary = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "block_size": block_size, "seed": seed,
               "trainable_params": trainable, "total_params": total, "lora": model.lora_config,
               "train_lines": len(train), "val_lines": len(val)}
    finetune.save_training_log(hist, summary, out_dir)
    return hist


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
# Hint: Utilize your fine-tuned LoRA-adapted LLM to generate new text.
def milestone_5(model, tok, prompt: str = "", max_new_tokens: int = 40, temperature: float = 0.8, top_k: int = 40,
                seed: int = 0, adapters: bool = True) -> str:
    """Generate from the LoRA-adapted LLM (adapters=False runs the untouched base model for comparison)."""
    if adapters:
        return gen.generate(model, tok, prompt, max_new_tokens, temperature, top_k, seed)
    with lora.adapters_disabled(model):
        return gen.generate(model, tok, prompt, max_new_tokens, temperature, top_k, seed)


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
# Hint: Design a simple display to show the outputs from both models side-by-side for easy evaluation.
def load_adapted_model(model_name: str = "distilgpt2", adapter_path=ADAPTER_PATH, device=None):
    """Fresh base model + the saved adapter (no retraining needed). Returns (model, tokenizer)."""
    model, tok = load_model(model_name)
    lora.load_lora(model, adapter_path)
    lora.assert_only_lora_trainable(model)
    return model.to(device or pick_device()).eval(), tok


def milestone_6(model, tok, ngram: NGramModel, prompts=None, max_new_tokens: int = 40, temperature: float = 0.8,
                top_k: int = 40, seed: int = 0, out_path=OUT_DIR / "comparison.md", model_name: str = "distilgpt2",
                corpus_path=CLEAN_PATH, discussion_path=DOCS_DIR / "discussion.md", width: int | None = None) -> str:
    """Generate from N-gram, base LLM (adapters off) and LoRA LLM for the same prompts; print-ready side-by-side text,
    plus a markdown report written to out_path. Returns the side-by-side text."""
    prompts = list(prompts or compare.DEFAULT_PROMPTS)
    kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)
    rows = compare.run_comparison(
        prompts,
        lambda p: ngram.generate(p, max_tokens=max_new_tokens, temperature=temperature, seed=seed),
        lambda p: milestone_5(model, tok, p, adapters=False, **kw),
        lambda p: milestone_5(model, tok, p, adapters=True, **kw))
    _, val = corpus.split_lines(corpus.load_corpus(corpus_path))
    device, block = next(model.parameters()).device, 64
    with lora.adapters_disabled(model):
        base_ppl = finetune.perplexity(finetune.eval_loss(model, tok, val, block, device))
    lora_ppl = finetune.perplexity(finetune.eval_loss(model, tok, val, block, device))
    trainable, total = lora.count_parameters(model)
    meta = {"temperature": temperature, "top_k": top_k, "seed": seed, "max_new_tokens": max_new_tokens, "n": ngram.n,
            "k": ngram.k, "model_name": model_name, "ngram_ppl": ngram.perplexity(val), "base_ppl": base_ppl,
            "lora_ppl": lora_ppl, "trainable": trainable, "total": total, "pct": 100 * trainable / total,
            **{k: model.lora_config[k] for k in ("r", "alpha")}, "targets": ", ".join(model.lora_config["target_modules"])}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(compare.render_markdown(rows, meta, compare.load_discussion(discussion_path)), encoding="utf-8")
    return compare.render_side_by_side(rows, width)
