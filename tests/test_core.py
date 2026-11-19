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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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

