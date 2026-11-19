# Stylistic Text Generator Comparator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/fdv4g8r4n4/badge.svg)](https://ask.thili.ai/b/fdv4g8r4n4)

> **This is an AI assignment for the thiliAI course track project on *Language Models from scratch*.**
> Author: **Nikita Ravindra, USN 1MS24CI079**.

## What it does

It generates sports commentary with two language models trained on the **same custom corpus** and shows their output
side by side, so you can judge the style qualitatively:

| model | how it is built |
|---|---|
| **N-gram** | written from scratch: counts, add-k smoothing, backoff to a lower order on unseen context, temperature sampling, perplexity. `n` is configurable (default 3). |
| **distilgpt2 + LoRA** | the pretrained `distilgpt2` (82M params) with **hand-written LoRA adapters** (no `peft`), fine-tuned on the corpus. The sampling loop (temperature / top-k) is also hand-written; `model.generate` is not used. |

The comparison shows three columns: **N-gram**, **base distilgpt2** (adapters switched off) and **distilgpt2 + LoRA**.
Base vs adapted is what proves the fine-tune changed the style.

## The six milestones

1. **Corpus** - `corpus/raw/` holds original, deliberately noisy transcripts (timestamps, speaker tags, `[crowd noise]`, ad reads, HTML entities, stray whitespace; football, cricket, basketball, tennis; all fictional players and teams, no copyrighted text). `app/corpus.py` cleans them into `corpus/clean.txt`.
2. **N-gram model** - `app/ngram.py`.
3. **LoRA injection** - `app/lora.py` (`LoRALinear`, `inject_lora`, freezing, save/load) and `app/models.py`.
4. **Fine-tuning** - `app/finetune.py`.
5. **Generation** - `app/generate.py`.
6. **Comparison interface** - `app/compare.py`, the `compare` CLI command and `outputs/comparison.md`.

`app/core.py` has one function per milestone (`milestone_1` ... `milestone_6`) that orchestrates the modules above.

## Setup

Python 3.12 and git. Apple Silicon uses the `mps` device automatically, otherwise CPU (override with `STG_DEVICE=cpu`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # only needed for the optional chat helper in app/llm.py; never commit .env
python -m app.cli hello
pytest                      # offline, no model download
```

The first real run downloads `distilgpt2` (~350 MB) from Hugging Face once.

## CLI

All commands: `python -m app.cli <command>`.

| command | what it does |
|---|---|
| `hello` | check the wiring |
| `corpus` | clean `corpus/raw/*.txt` into `corpus/clean.txt`, print lines / tokens / vocab size |
| `ngram [-n 3] [--prompt TEXT] [--samples 5] [--temperature 0.8]` | build the n-gram model, print held-out perplexity and samples |
| `train [--epochs 8] [--lr 1e-3] [--rank 8] [--alpha 16] [--batch-size 8] [--block-size 64]` | inject LoRA into distilgpt2, fine-tune, save the adapter to `outputs/adapter/`, print the loss curve |
| `generate --which {ngram,base,lora} --prompt TEXT` | generate from one model (uses the saved adapter) |
| `compare [--prompt TEXT ...]` | side-by-side N-gram vs base vs LoRA in the terminal, and write `outputs/comparison.md` |
| `all` | corpus -> n-gram -> train -> compare, end to end |

`--model-name tiny` swaps in a random 2-layer GPT-2 with a byte tokenizer so any command runs offline in seconds (this is what the tests use).
The trained adapter is committed (`outputs/adapter/lora_adapter.pt`, 1.6 MB), so `compare` and `generate` work without retraining:

```bash
python -m app.cli compare --prompt "The crowd" --prompt "Match point,"
```

## Corpus

4 raw files, 277 raw lines -> **197 clean commentary lines, 4,328 tokens, vocab size 1,050** (word-level tokens, 22.0 per line).
Every 10th line is held out for validation / perplexity; both models train on the rest.

## LoRA configuration

- Base model `distilgpt2`, loaded in float32; **all base weights frozen**, only adapter weights train (asserted in code and tests).
- **Rank r = 8, alpha = 16** (scale alpha / r = 2), dropout 0.05, `B` initialised to zero so the adapted model starts identical to the base.
- **Target modules: `c_attn`, `c_proj`** (attention qkv, attention output, MLP output; 18 layers wrapped across the 6 blocks).
- **Trainable parameters: 405,504 of 82,318,080 = 0.493%.**
- GPT-2 uses `transformers.pytorch_utils.Conv1D`, whose weight is `(in, out)` and computes `x @ W + b`, the transpose of `nn.Linear`. `LoRALinear` reads the sizes by layer type and a test checks `x @ (W + dW) + b` equals the adapter forward.
- Training: AdamW, lr 0.001, 8 epochs, batch 8, block size 64, grad-clip 1.0, seed 0, on MPS in about a minute.

### Training loss

Held-out loss: **4.691 (base) -> 3.331** after training. Train loss per step (8 epochs, 72 steps):

```
train loss
  4.98 | *                                                               
  4.77 |                                                                 
  4.57 |  **                                                             
  4.36 |    *                                                            
  4.15 |     ****                                                        
  3.95 |         * *** *     *                                           
  3.74 |          *     ****     *    *                                  
  3.54 |              *     *  **  ***  ** ******    *   *     *         
  3.33 |                      *   *    *  *      **** *** ****  * * * * *
  3.12 |                                                      *  * * * * 
       +----------------------------------------------------------------
        step 1                                                        step 72
```

## Sample outputs (real, from `python -m app.cli all`)

Temperature 0.8, top-k 40, seed 0, up to 40 new tokens.

#### 1. Prompt: `The crowd`

| model | output |
|---|---|
| N-gram | The crowd rises to salute two warriors of the floor, with honours even at the net, with honours even at the other end! |
| distilgpt2 (base) | The crowd, who wanted to see the movie, started singing songs, the crowd was chanting, and they started singing about it.   “The audience sang it in a large way,“ |
| distilgpt2 + LoRA | The crowd cheers for their winner! |

#### 2. Prompt: `He shoots and`

| model | output |
|---|---|
| N-gram | He shoots and the second one rattles in, and that's four runs to the bowlers walk off to the visitors lead by two with ten seconds to play. |
| distilgpt2 (base) | He shoots and shoots when he's caught by your nose,” he says.     "Just look at the guy that was so upset he just got up in a car he saw somebody jump |
| distilgpt2 + LoRA | He shoots and a wonderful header over the spot! |

#### 3. Prompt: `What a delivery from the bowler,`

| model | output |
|---|---|
| N-gram | What a delivery from the bowler, and the home favourite on the spinner, and the visitors are hanging on by their fingernails. |
| distilgpt2 (base) | What a delivery from the bowler, a delivery from the bowler... is the life of the bowler... the life of the bowler... is the life of the bowler... the life of the bowler... is the life |
| distilgpt2 + LoRA | What a delivery from the bowler, a delivery from the left stump, is off the ground, the batsman is hit by a single delivery! |

#### 4. Prompt: `Break point,`

| model | output |
|---|---|
| N-gram | Break point, whatever it is, the drums are pounding, and the home side hold on for a winner! |
| distilgpt2 (base) | Break point, a new method to make the use of "lots of the data" to be available to anyone. It is a new method to make the use of "lots of the data" to be |
| distilgpt2 + LoRA | Break point, a wonderful pass, and the ball is bouncing back to the top of the line for the game! |

#### 5. Prompt: `The goalkeeper`

| model | output |
|---|---|
| N-gram | The goalkeeper is jogging up the pitch. |
| distilgpt2 (base) | The goalkeeper, who scored the first goal in the game at the age of 21, has been described as one of the world's best players since 2010. |
| distilgpt2 + LoRA | The goalkeeper is on his toes, and the visitors are at full speed! |

#### 6. Prompt: `Fast break,`

| model | output |
|---|---|
| N-gram | Fast break, two sets down, five, four, and the visitors are hanging on by their fingernails. |
| distilgpt2 (base) | Fast break, a new series of events that might prove to be a boon to my heart.  This is the third installment in my series of podcasts. This is the fifth in a series of podcasts designed to |
| distilgpt2 + LoRA | Fast break, a wonderful header, and the hosts are on the verge of the relegation zone. |

### Held-out perplexity

| N-gram (n=3) | word-level perplexity | 192.2 |
| distilgpt2 (base) | token-level perplexity | 109.0 |
| distilgpt2 + LoRA | token-level perplexity | 28.0 |

N-gram (word-level) and LLM (sub-word) perplexities are not comparable with each other; base vs LoRA is (109 -> 28).

## What the comparison showed

- **N-gram:** right vocabulary, but it only splices fragments it has seen. It loses the thread mid-sentence and mixes sports.
- **Base distilgpt2:** fluent, but the wrong style: web / podcast prose, repetition loops, never ends a line.
- **LoRA:** the style clearly moved: short, exclamatory, present-tense broadcast lines that end on their own, and it stays closer to the prompt. It is still shaky on facts and sport-specific logic ("a wonderful header" in a tennis or basketball context) because the corpus is only ~4k words.

The full write-up is in [`outputs/comparison.md`](outputs/comparison.md) (source: `docs/discussion.md`).

## Project layout

```
app/
  core.py       one function per milestone (orchestration)
  corpus.py     cleaning pipeline, tokenizer, train/val split
  ngram.py      n-gram model
  lora.py       hand-written LoRA (LoRALinear, inject, freeze, save/load)
  models.py     distilgpt2 loader, tiny offline test model, byte tokenizer, device pick
  finetune.py   training loop, eval loss, ASCII loss curve
  generate.py   temperature / top-k sampling loop
  compare.py    side-by-side view and markdown report
  cli.py        corpus / ngram / train / generate / compare / all
  llm.py, storage.py   starter helpers (optional chat helper reads .env)
corpus/raw/     noisy raw transcripts          corpus/clean.txt   cleaned corpus (NOT in data/, which is gitignored)
scripts/make_raw_corpus.py   writes the raw transcripts
outputs/        adapter/, comparison.md, train_log.json, loss_curve.txt, corpus_stats.json
docs/discussion.md   qualitative discussion included in the report
tests/          one test per milestone + smoke tests (all offline, no downloads)
```

## Tests and CI

`pytest` runs offline in a few seconds: the LoRA / training / generation tests use a 2-layer random GPT-2 (`n_embd=64`) with a byte-level tokenizer, so CI never downloads weights. The real `distilgpt2` run lives behind the CLI.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** The `LLM_*` settings are for the optional chat helper in `app/llm.py`; the LoRA fine-tune is fully local.
