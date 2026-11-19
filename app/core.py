"""One function per milestone. Each is a thin orchestrator over a focused module:

    1 corpus.py    2 ngram.py    3 lora.py + models.py    4 finetune.py    5 generate.py    6 compare.py
"""
from pathlib import Path

from app import corpus
from app.ngram import NGramModel

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "corpus" / "raw"
CLEAN_PATH = ROOT / "corpus" / "clean.txt"
OUT_DIR = ROOT / "outputs"

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
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
# Hint: Train the LoRA adapters on your prepared corpus, following the fine-tuning process from the drill.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
# Hint: Utilize your fine-tuned LoRA-adapted LLM to generate new text.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
# Hint: Design a simple display to show the outputs from both models side-by-side for easy evaluation.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
