"""Milestone 2: an n-gram language model from scratch.

* counts of every order 1..n over sentence-padded token streams (<s> ... </s>)
* add-k smoothing on the highest order whose context was seen in training
* backoff: if the (n-1)-token context was never seen, drop the oldest token and try the shorter context
* temperature sampling over the continuations observed after the matched context
* perplexity on held-out lines
"""
import math
import random
from collections import Counter, defaultdict

from app.corpus import detokenize, tokenize

BOS, EOS = "<s>", "</s>"


class NGramModel:
    def __init__(self, n: int = 3, k: float = 0.05):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.k = n, k
        # counts[order][context tuple (order-1 tokens)] -> Counter(next token)
        self.counts: dict[int, dict[tuple, Counter]] = {o: defaultdict(Counter) for o in range(1, n + 1)}
        self.vocab: set[str] = set()

    def fit(self, lines: list[str]) -> "NGramModel":
        for line in lines:
            toks = [BOS] * (self.n - 1) + tokenize(line) + [EOS]
            self.vocab.update(toks)
            for i in range(self.n - 1, len(toks)):
                for order in range(1, self.n + 1):
                    ctx = tuple(toks[i - order + 1:i])
                    self.counts[order][ctx][toks[i]] += 1
        self.vocab.discard(BOS)
        return self

    # -- probabilities -------------------------------------------------
    def _matched(self, context: list[str]) -> tuple[tuple, Counter]:
        """Longest context (<= n-1 tokens) seen in training, with its next-token counts. Backs off to ()."""
        context = ([BOS] * (self.n - 1) + list(context))[-(self.n - 1):] if self.n > 1 else []
        for order in range(self.n, 0, -1):
            ctx = tuple(context[len(context) - (order - 1):]) if order > 1 else ()
            if ctx in self.counts[order]:
                return ctx, self.counts[order][ctx]
        raise RuntimeError("model has not been fitted")

    def prob(self, token: str, context: list[str]) -> float:
        """Add-k smoothed P(token | context) at the longest seen context."""
        _, counts = self._matched(context)
        total = sum(counts.values())
        return (counts[token] + self.k) / (total + self.k * (len(self.vocab) + 1))

    def perplexity(self, lines: list[str]) -> float:
        logp, count = 0.0, 0
        for line in lines:
            toks = [BOS] * (self.n - 1) + tokenize(line) + [EOS]
            for i in range(self.n - 1, len(toks)):
                logp += math.log(self.prob(toks[i], toks[:i]))
                count += 1
        return math.exp(-logp / max(1, count))

    # -- generation ----------------------------------------------------
    def next_distribution(self, context: list[str], temperature: float = 1.0) -> dict[str, float]:
        """Distribution over observed continuations of the matched context, sharpened/flattened by temperature."""
        _, counts = self._matched(context)
        t = max(temperature, 1e-3)
        weights = {w: (c + self.k) ** (1.0 / t) for w, c in counts.items()}
        z = sum(weights.values())
        return {w: v / z for w, v in weights.items()}

    def generate(self, prompt: str = "", max_tokens: int = 40, temperature: float = 0.8,
                 seed: int | None = None) -> str:
        """Continue `prompt` (may be empty). Stops at end-of-sentence or max_tokens. Returns only the continuation."""
        rng = random.Random(seed)
        context = tokenize(prompt)
        out: list[str] = []
        for _ in range(max_tokens):
            dist = self.next_distribution(context + out, temperature)
            words, probs = zip(*sorted(dist.items()))  # sorted => reproducible under a seed
            nxt = rng.choices(words, weights=probs)[0]
            if nxt == EOS:
                break
            out.append(nxt)
        return detokenize(out)
