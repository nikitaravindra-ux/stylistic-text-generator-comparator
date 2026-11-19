These are single samples at one seed on a 197-line corpus, so treat the observations as qualitative, not statistical.

**N-gram (n=3).** *Good:* every word is in-domain, so it sounds like commentary, and short stretches are grammatical
because they are copied from training lines ("hanging on by their fingernails", "honours even"). It is fast, needs no
GPU and is fully inspectable. *Bad:* it only recombines fragments it has seen. Output often jumps mid-sentence between
unrelated clauses and even between sports ("...the second one rattles in, and that's four runs to the bowlers walk off to
the visitors lead by two..." mixes basketball and cricket). With a context of only two words it has no idea what the sentence is about,
and when the prompt's last words were never seen it silently backs off to a shorter context and ignores the prompt's
meaning. Longer sentences are where this falls apart.

**Base distilgpt2.** *Good:* fluent, grammatical English with sensible local structure, and it does pick up the prompt's
topic words ("The crowd ... chanting"). *Bad:* it has no commentary style at all. It drifts into web-article, podcast or
news prose ("a new series of podcasts", "one of the world's best players since 2010") and falls into repetition loops
("the life of the bowler... the life of the bowler..."). It also never ends a line: it runs to the token limit.

**distilgpt2 + LoRA (0.49% of weights trained, held-out perplexity 109 -> 28).** *Good:* the style clearly moved. Outputs
are short, exclamatory, present-tense, in the voice of a live broadcast ("The goalkeeper is on his toes, and the visitors
are at full speed!"), and the model learned the corpus convention of ending a commentary line, so it stops by itself.
It stays on prompt better than the n-gram and writes new sentences rather than only splicing training text. *Bad:* it
is fluent but not always factual: "the batsman is hit by a single delivery", "a wonderful header" after a tennis or
basketball prompt, "relegation zone" after "Fast break,". With only ~4k words of training data the adapter learned the
register and vocabulary, not the rules of each sport, and it sometimes blends sports much like the n-gram does. It is
also far heavier to run than the n-gram.

**Takeaway.** The base model has the language ability but the wrong style; the n-gram has the right style but little
language ability. The LoRA adapter combines the two: a tiny, hand-written set of trainable weights is enough to move a
pretrained model into the commentary register. Its remaining errors are about world knowledge and sport-specific
coherence, which a larger or per-sport corpus would help with.
