"""Milestone 1: load raw broadcast transcripts and clean them into one commentary sentence-or-two per line."""
import html
import json
import re
import unicodedata
from pathlib import Path

TAG_RE = re.compile(r"<[^>]+>")
# [00:01:23] / [03:24] / 00:01:23 / (34')  at the start of a line
TIMESTAMP_RE = re.compile(r"^\s*(?:\[\d{1,2}:\d{2}(?::\d{2})?\]|\d{1,2}:\d{2}(?::\d{2})?|\(\d+'\))\s*")
# "MIKE TANNER:", "Mike:", "COMMENTATOR 1:"  (1-3 words, short, followed by a colon)
SPEAKER_RE = re.compile(r"^\s*(?:COMMENTATOR\s*\d+|[A-Za-z][A-Za-z.'-]*(?:\s[A-Za-z][A-Za-z.'-]*){0,2})\s*:\s+")
MARKER_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [crowd noise], (laughter), [applause] ...
AD_RE = re.compile(r"brought to you by|^\s*(?:ad\s*read|ad|advertisement|sponsor(?:ed)?(?:\s+message)?)\s*[:\]]|"
                   r"\[sponsor|promo code|terms and conditions|stay tuned after the break|\.example\b", re.I)
HEADER_RE = re.compile(r"^\s*===.*===\s*$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?|\.\.\.|[.,!?;:-]")


def clean_line(raw: str) -> str:
    """Clean one raw transcript line. Returns '' if the line is noise (marker-only, ad read, header...)."""
    if HEADER_RE.match(raw) or AD_RE.search(html.unescape(raw)):
        return ""
    text = html.unescape(raw)
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    text = TAG_RE.sub("", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = TIMESTAMP_RE.sub("", text, count=1)
    text = MARKER_RE.sub(" ", text)
    text = SPEAKER_RE.sub("", text, count=1)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # "word ," left over from &nbsp;,
    return text


def clean_text(raw: str, min_words: int = 4) -> list[str]:
    """Clean a whole raw transcript: per-line cleaning, drop short fragments and exact duplicates."""
    seen, out = set(), []
    for line in raw.splitlines():
        c = clean_line(line)
        if len(c.split()) < min_words or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if not out:
            out = t
        elif t in {".", ",", "!", "?", ";", ":", "..."}:
            out += t
        else:
            out += " " + t
    return out


def corpus_stats(lines: list[str]) -> dict:
    toks = [t for line in lines for t in tokenize(line)]
    return {"lines": len(lines), "tokens": len(toks), "vocab_size": len(set(toks)),
            "avg_tokens_per_line": round(len(toks) / max(1, len(lines)), 1)}


def build_corpus(raw_dir: str | Path, out_path: str | Path, stats_path: str | Path | None = None) -> dict:
    """Clean every .txt in raw_dir, write one line per commentary line to out_path, return stats."""
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no raw transcripts in {raw_dir}; run scripts/make_raw_corpus.py")
    lines, raw_lines = [], 0
    for f in files:
        raw = f.read_text(encoding="utf-8")
        raw_lines += len(raw.splitlines())
        lines.extend(clean_text(raw))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats = {"raw_files": len(files), "raw_lines": raw_lines, **corpus_stats(lines)}
    if stats_path:
        Path(stats_path).write_text(json.dumps(stats, indent=2))
    return stats


def load_corpus(path: str | Path) -> list[str]:
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def split_lines(lines: list[str], val_every: int = 10) -> tuple[list[str], list[str]]:
    """Deterministic train/validation split: every `val_every`-th line is held out."""
    train = [l for i, l in enumerate(lines) if (i + 1) % val_every]
    val = [l for i, l in enumerate(lines) if not (i + 1) % val_every]
    return train, val
