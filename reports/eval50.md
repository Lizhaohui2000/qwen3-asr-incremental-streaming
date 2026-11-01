# eval50 results

## Setup

- Model: Qwen3-ASR-0.6B, local checkpoint
- Device: RTX 3090, p9203 GPU 0
- Runtime: PyTorch `2.9.1+cu128`, `qwen-asr 0.0.6`, `transformers 4.57.3`, `vllm 0.14.0`, BF16
- Input: private 50-clip streaming evaluation set; 1,465.66 seconds total, 16 kHz
- Updates: 2 seconds; automatic language; greedy decoding; 512 max new tokens
- Bounded paths: 30-second audio context; 2-second overlap for the incremental epoch rollover
- Scores: CER with whitespace and selected punctuation removed; raw audio, references,
  and per-clip hypotheses are intentionally not included here.

## Matched per-clip evaluation

| Path | Corpus CER | Update P50 / P95 | RTF | GPU peak | Host process-tree peak RSS / USS |
|---|---:|---:|---:|---:|---:|
| Official prefix replay | **3.709%** (176/4,745) | 0.274 / 0.349 s | **0.141** | 12,811 MiB | 4,421 / 3,555 MiB |
| 30-second rolling re-decode | 4.004% (190/4,745) | 0.845 / 1.461 s | 0.433 | 12,811 MiB | 4,559 / 3,694 MiB |
| Encoder-block + decoder-KV reuse | 3.772% (179/4,745) | 0.483 / 0.601 s | 0.248 | **2,223 MiB** | **2,287 / 2,281 MiB** |

All three paths completed 747 updates with no update exceeding two seconds.
Against rolling re-decode, the incremental path reduced update P50 by 43% and
RTF by 43%, with 0.232 percentage points lower corpus CER. Against official
prefix replay, it used much less measured GPU memory but was slower (P50 0.483
vs 0.270 s) and had 0.063 percentage points higher CER. These are descriptive
results on this eval set, not evidence that the incremental method improves
recognition accuracy.

Each clip starts a fresh stream, so this table measures per-clip recognition and
runtime, not unbounded-session growth. The official path's host peak was sampled
on a repeat run; runtime values in the table use that repeat run.

## Long-session check

The first four eval50 clips were joined with 0.2-second silent gaps and processed
as one 119.15-second stream, with the same 2-second update and 512-token limit.

| Path | Corpus CER | Update P50 / P95 | First-half → second-half P50 | RTF | Max audio context | GPU peak | Host RSS / USS peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official prefix replay | 3.235% (12/371) | 0.318 / 0.392 s | 0.299 → 0.339 s | **0.161** | 118 s | 15,715 MiB | 4,390 / 3,530 MiB |
| 30-second rolling re-decode | 9.704% (36/371) | 1.418 / 1.589 s | 1.284 → 1.491 s | 0.665 | 30 s | 12,811 MiB | 4,321 / 3,463 MiB |
| Encoder-block + decoder-KV reuse | **2.965%** (11/371) | 0.502 / 0.595 s | 0.494 → 0.506 s | 0.255 | 30 s | **2,223 MiB** | **2,313 / 2,308 MiB** |

The cache path stayed within its 30-second context through epoch rollover and
kept update latency stable across the two halves. The rolling comparator's CER
rose sharply during transcript stitching; it is our local rolling-window
baseline, not a direct run of the Hangry-Labs repository. This four-clip result
is small and should be treated as a rollover check, not a general accuracy
claim. This run does not establish 10-minute stability.

## Caveats

- This is a local engineering benchmark, not a public benchmark or statistical
  equivalence claim.
- The incremental cache tolerance was selected during earlier development on
  eval50; verify it on held-out audio before making a general accuracy claim.
- The 8-second encoder block follows `n_window_infer=800`; no block-size sweep
  has been run.
- Runtime versions and model backend differ between vLLM and Transformers. The
  three modes use the same model checkpoint, audio, update cadence, language
  mode, and token limit, but backend overhead is part of the measured result.
- VRAM was sampled with `nvidia-smi`; host RSS/USS sums the evaluator process and
  child workers at 0.5-second intervals. Reported peaks are sampled peaks.
- The 120-second rolling-window result is sensitive to transcript-boundary
  stitching. Do not use it as a direct estimate of Hangry-Labs' implementation.
