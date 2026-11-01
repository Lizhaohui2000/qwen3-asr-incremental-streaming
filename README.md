# Bounded-context Qwen3-ASR streaming with cache reuse

Experimental 2-second streaming backend for Qwen3-ASR-0.6B. It bounds audio
context to 30 seconds, reuses completed encoder windows and decoder KV within
each epoch, and revises a short text suffix. It uses the Hugging Face model
backend; it is not wired into the official vLLM demo.

## Compared paths

| Path | Audio context | Reuses encoder/decoder state |
|---|---:|---|
| Official prefix replay | Grows with session | No |
| Fixed rolling window | 30 s | No; re-decodes the window |
| Incremental cache | 30 s | Yes; completed 8 s encoder blocks and epoch-local decoder KV |

The 8-second block follows [`n_window_infer=800`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/config.json)
(800 10-ms Mel frames). It has not been selected by a block-size sweep. The
active partial encoder block is recomputed; log-Mel features are recomputed over
the bounded epoch on each update.

## Relation to existing streaming issue

[Qwen3-ASR issue #199](https://github.com/QwenLM/Qwen3-ASR/issues/199) discusses
unbounded audio replay. [Hangry-Labs](https://github.com/Hangry-Labs/Qwen3-ASR-STT)
addresses it with repeated inference over a bounded rolling window. This repo
tests reuse of completed encoder windows and decoder KV on top of a bounded
context; it is a separate Hugging Face backend, not a benchmark of Hangry-Labs.

## Install

Install PyTorch for the target CUDA version, then:

```bash
pip install -r requirements.txt
```

The vLLM comparison paths additionally require the matching vLLM build:

```bash
pip install -r requirements-vllm-benchmark.txt
```

## Run

The benchmark runner expects a private JSONL manifest with `id`, `audio`, and
`reference` fields. Keep the dataset and per-utterance outputs private.

```bash
python scripts/evaluate_streaming_modes.py --mode official --model /path/to/Qwen3-ASR-0.6B --manifest /path/to/manifest.jsonl --out /tmp/official.json
python scripts/evaluate_streaming_modes.py --mode rolling --model /path/to/Qwen3-ASR-0.6B --manifest /path/to/manifest.jsonl --out /tmp/rolling.json
python scripts/evaluate_streaming_modes.py --mode incremental --model /path/to/Qwen3-ASR-0.6B --manifest /path/to/manifest.jsonl --out /tmp/incremental.json
```

To check long-session rollover, select several clips and join them into one
continuous stream with short silent gaps:

```bash
python scripts/evaluate_streaming_modes.py --mode incremental --model /path/to/Qwen3-ASR-0.6B --manifest /path/to/manifest.jsonl --ids 000,001,002,003 --continuous --gap-sec 0.2 --out /tmp/incremental-long.json
```

All three modes use automatic language detection, greedy decoding, 2-second
updates, 512 maximum new tokens, and the same 16-kHz inputs. The rolling and
incremental paths use a 30-second window. JSON output contains references and
hypotheses; do not publish it if the underlying evaluation set is private.

## Current status

On the private 50-clip eval set, corpus CER was 3.709% for official prefix
replay, 4.004% for 30-second rolling re-decode, and 3.772% for incremental
cache reuse. Against rolling re-decode, cache reuse had 0.483 s vs 0.845 s
update P50 and 0.232 percentage points lower CER. The sampled GPU peak was 2.2
vs 12.8 GiB, but this memory comparison is confounded by the different
Transformers and vLLM backends. Against official prefix replay, it was slower
(0.483 s vs 0.274 s P50) and had 0.063 percentage points higher CER. The
119-second rollover check and full limitations are in
[`reports/eval50.md`](reports/eval50.md).

These are experimental results, not statistical equivalence claims. The
incremental implementation accesses Qwen model internals and should be tested
against pinned Transformers and Qwen-ASR versions.
