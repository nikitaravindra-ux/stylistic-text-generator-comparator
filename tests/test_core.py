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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Generate text from the LoRA-adapted LLM.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

