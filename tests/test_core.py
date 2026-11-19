"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from app import core, corpus

RAW = """=== FOOTBALL BROADCAST - AUTO TRANSCRIPT ===
[music]

[00:00:31] And here comes Marlowe down the left wing, he&#39;s got space, and he&rsquo;s looking for the cross!
  00:00:35\tSARAH OKOYE: Oh, what a [crowd noise] ball from Duarte, threaded through the  defence! 
(34') <b>He shoots... and it&#39;s in! &amp; Unbelievable scenes</b>
Mike: The keeper gets a hand to it&nbsp;, but it's a goal for the home side!
This half-time report is brought to you by Northgate Insurance. Get a quote today.
AD READ: Zephyr Sneakers, run faster!
[applause]
[00:00:35] COMMENTATOR 1: Oh, what a ball from Duarte, threaded through the defence!
"""


# Milestone 1: Prepare a small corpus of sports commentary text.
def test_milestone_1(tmp_path):
    lines = corpus.clean_text(RAW)
    joined = "\n".join(lines)
    for noise in ("00:00", "[", "]", "&#39;", "&rsquo;", "&amp;", "&nbsp;", "<b>", "crowd noise", "SARAH", "Mike:",
                  "brought to you", "Zephyr", "(34')", "  ", "=== "):
        assert noise not in joined, noise
    assert "he's got space" in joined and "he's looking for the cross" in joined
    assert "He shoots... and it's in! & Unbelievable scenes" in joined
    assert "a hand to it, but" in joined
    assert len(lines) == 4  # four commentary lines survive; the duplicate "Duarte" line is dropped
    # real pipeline end to end: writes clean file + stats
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    (raw_dir / "a.txt").write_text(RAW)
    stats = core.milestone_1(raw_dir, tmp_path / "clean.txt", tmp_path / "stats.json")
    assert stats["lines"] == 4 and stats["tokens"] > 30 and 0 < stats["vocab_size"] <= stats["tokens"]
    assert corpus.load_corpus(tmp_path / "clean.txt") == lines


# Milestone 2: Implement the n-gram model and generate text from it.
def test_milestone_2():
    from app.ngram import NGramModel
    lines = ["the striker scores a goal!", "the keeper makes a save!"] * 5 + ["the striker misses a sitter!"]
    m = NGramModel(n=3, k=0.1).fit(lines)
    # context matters: after "striker" a goal/miss is possible, never "makes"; after "keeper" only "makes"
    after_striker = m.next_distribution(["the", "striker"], temperature=1.0)
    assert set(after_striker) == {"scores", "misses"} and after_striker["scores"] > after_striker["misses"]
    assert set(m.next_distribution(["the", "keeper"])) == {"makes"}
    # smoothing: unseen words still get non-zero probability, and probabilities are sane
    assert 0 < m.prob("penalty", ["the", "striker"]) < m.prob("scores", ["the", "striker"]) < 1
    # backoff: an unseen two-token context falls back to a shorter one instead of crashing
    assert m.next_distribution(["zzz", "striker"]) == m.next_distribution(["striker"])
    # lower temperature is sharper
    assert (m.next_distribution(["the", "striker"], 0.3)["scores"] > after_striker["scores"])
    # deterministic under a seed, honours the prompt, stops at end of sentence
    a = m.generate("the keeper", seed=3)
    assert a == m.generate("the keeper", seed=3) and a == "makes a save!"
    # perplexity: in-domain text beats scrambled text; n is configurable
    assert m.perplexity(["the striker scores a goal!"]) < m.perplexity(["goal a scores striker the!"])
    assert NGramModel(n=2).fit(lines).n == 2
    # core wiring, on the real corpus
    real = core.milestone_2(n=2)
    assert real.n == 2 and len(real.generate("The crowd", seed=1)) > 0


# Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
def test_milestone_3(tmp_path):
    import torch
    from app import lora
    from app.models import tiny_model
    base, tok = tiny_model()
    ids = torch.tensor([tok.encode("what a goal!")])
    with torch.no_grad():
        before = base(ids).logits.clone()
    model, _ = core.milestone_3("tiny", r=4, alpha=8, dropout=0.0)
    # GPT-2 attention layers are Conv1D, with weight shaped (in, out): the wrapper must handle that layout
    layer = model.transformer.h[0].attn.c_attn
    assert isinstance(layer, lora.LoRALinear) and layer.is_conv1d
    assert layer.base.weight.shape == (64, 192) and layer.lora_A.shape == (4, 64) and layer.lora_B.shape == (192, 4)
    # only adapter params train, and they are a small fraction
    lora.assert_only_lora_trainable(model)
    trainable, total = lora.count_parameters(model)
    assert 0 < trainable < 0.5 * total and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    # B starts at zero => identical to the base model until trained
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, before, atol=1e-6)
    # a non-zero adapter changes the forward pass, and disabling adapters restores the base output exactly
    with torch.no_grad():
        for l in lora.lora_layers(model):
            l.lora_B.normal_(0, 0.5)
        adapted = model(ids).logits
        assert not torch.allclose(adapted, before, atol=1e-3)
        with lora.adapters_disabled(model):
            assert torch.allclose(model(ids).logits, before, atol=1e-6)
        assert torch.allclose(model(ids).logits, adapted)  # re-enabled afterwards
        # layout maths: x @ (W + dW) + b  ==  adapter forward (Conv1D stores W as (in, out))
        x = torch.randn(3, 64)
        merged = x @ (layer.base.weight + layer.delta_weight()) + layer.base.bias
        assert torch.allclose(layer(x), merged, atol=1e-5)
    # base weights untouched by injection; adapter round-trips through save/load
    lora.save_lora(model, tmp_path / "adapter.pt")
    fresh, _ = tiny_model()
    lora.load_lora(fresh, tmp_path / "adapter.pt")
    with torch.no_grad():
        assert torch.allclose(fresh(ids).logits, adapted, atol=1e-6)
    # nn.Linear layers work too
    lin = lora.LoRALinear(torch.nn.Linear(5, 7), r=2)
    assert lin.lora_A.shape == (2, 5) and lin.lora_B.shape == (7, 2) and lin(torch.randn(1, 5)).shape == (1, 7)


# Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
def test_milestone_4(tmp_path):
    import json
    import torch
    from app import lora
    model, tok = core.milestone_3("tiny", r=8, alpha=16, dropout=0.0)
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    small = tmp_path / "small.txt"
    small.write_text("\n".join(corpus.load_corpus(core.CLEAN_PATH)[:30]) + "\n")
    hist = core.milestone_4(model, tok, epochs=20, batch_size=8, lr=3e-2, block_size=32, corpus_path=small,
                            adapter_path=tmp_path / "a.pt", out_dir=tmp_path, device=torch.device("cpu"), log=lambda *_: None)
    # loss goes down, both on the training batches and on held-out text
    first = sum(hist["step_loss"][:3]) / 3
    last = sum(hist["step_loss"][-3:]) / 3
    assert last < first - 0.1, (first, last)
    assert hist["val_loss"][-1] < hist["val_loss"][0] - 0.2, hist["val_loss"]
    # only adapter weights moved; every base weight is bit-identical
    for n, p in model.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, base_before[n]) and not p.requires_grad, n
    assert any(float(p.detach().abs().sum()) > 0 for n, p in model.named_parameters() if "lora_B" in n)
    # artefacts: adapter file + training log + loss curve
    assert (tmp_path / "a.pt").exists() and (tmp_path / "a.pt").stat().st_size < 1_000_000
    log = json.loads((tmp_path / "train_log.json").read_text())
    assert log["trainable_params"] < log["total_params"] and (tmp_path / "loss_curve.txt").read_text().count("*") > 5


# Milestone 5: Generate text from the LoRA-adapted LLM.
def test_milestone_5():
    model, tok = core.milestone_3("tiny", dropout=0.0)
    import torch
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.2)  # make the adapter non-trivial
    a = core.milestone_5(model, tok, "The crowd", max_new_tokens=20, seed=7)
    assert a == core.milestone_5(model, tok, "The crowd", max_new_tokens=20, seed=7)       # deterministic under a seed
    assert a != core.milestone_5(model, tok, "The crowd", max_new_tokens=20, seed=8)       # ... and seed matters
    assert isinstance(a, str) and 0 < len(a.encode()) <= 20 * 4
    # adapters off => base model output, differs from adapted output; adapters come back on afterwards
    base = core.milestone_5(model, tok, "The crowd", max_new_tokens=20, seed=7, adapters=False)
    assert base != a and core.milestone_5(model, tok, "The crowd", max_new_tokens=20, seed=7) == a
    # greedy decoding and empty prompt work
    assert core.milestone_5(model, tok, "", max_new_tokens=5, temperature=0) == core.milestone_5(model, tok, "", max_new_tokens=5, temperature=0)


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
def test_milestone_6(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("STG_DEVICE", "cpu")
    import torch
    model, tok = core.milestone_3("tiny", dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.2)
    ngram = core.milestone_2(n=3)
    out = core.milestone_6(model, tok, ngram, ["The crowd", "Break point,"], max_new_tokens=15, seed=1,
                           out_path=tmp_path / "comparison.md", model_name="tiny", width=150,
                           discussion_path=tmp_path / "missing.md")
    # the side-by-side view labels all three models and shows each prompt
    for label in ("N-gram", "distilgpt2 (base)", "distilgpt2 + LoRA", "'The crowd'", "'Break point,'"):
        assert label in out
    md = (tmp_path / "comparison.md").read_text()
    assert md.count("### ") == 2 and "perplexity" in md and "| N-gram |" in md and "| distilgpt2 + LoRA |" in md
    # the n-gram column really is the n-gram model's output for that prompt
    assert "The crowd " + ngram.generate("The crowd", max_tokens=15, temperature=0.8, seed=1) in md
    # CLI: whole pipeline offline on the tiny model, then generate from the saved adapter
    from app.cli import main
    main(["all", "--model-name", "tiny", "--epochs", "1", "--out-dir", str(tmp_path / "run"), "--max-new-tokens", "10",
          "--prompt", "The crowd"])
    printed = capsys.readouterr().out
    assert "milestone 1" in printed and "trainable params" in printed and "report written" in printed
    assert (tmp_path / "run" / "adapter" / "lora_adapter.pt").exists() and (tmp_path / "run" / "comparison.md").exists()
    main(["generate", "--which", "base", "--model-name", "tiny", "--out-dir", str(tmp_path / "run"), "--prompt", "Hi",
          "--max-new-tokens", "5"])
    main(["ngram", "--samples", "2", "--prompt", "The crowd"])
