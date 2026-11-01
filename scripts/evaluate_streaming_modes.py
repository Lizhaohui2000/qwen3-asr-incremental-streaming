"""Matched eval50 runner for official replay, fixed-window replay, and KV reuse."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace() and ch not in '，。！？、；：“”‘’"\'')


def edit_distance(ref: str, hyp: str) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


class GPUMemorySampler:
    def __init__(self, gpu_index: int, period_sec: float = 0.5):
        self.gpu_index = gpu_index
        self.period_sec = period_sec
        self.samples_mib: list[int] = []
        self.host_rss_mib: list[float] = []
        self.host_uss_mib: list[float] = []
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--id", str(self.gpu_index), "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True, timeout=3
                )
                self.samples_mib.append(int(out.strip().splitlines()[0]))
            except Exception:
                pass
            try:
                import psutil
                root = psutil.Process(self.pid)
                processes = [root, *root.children(recursive=True)]
                rss = 0
                uss = 0
                for process in processes:
                    try:
                        mem = process.memory_full_info()
                        rss += mem.rss
                        uss += getattr(mem, "uss", mem.rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                self.host_rss_mib.append(rss / (1024**2))
                self.host_uss_mib.append(uss / (1024**2))
            except Exception:
                pass
            self._stop.wait(self.period_sec)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=4)


def make_model(args):
    from qwen_asr import Qwen3ASRModel

    if args.mode == "incremental":
        import torch
        return Qwen3ASRModel.from_pretrained(
            args.model, max_inference_batch_size=1, max_new_tokens=args.max_new_tokens,
            torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
    return Qwen3ASRModel.LLM(
        model=args.model, gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enforce_eager=True, max_inference_batch_size=1, max_new_tokens=args.max_new_tokens,
    )


def warmup(asr, args) -> float:
    started = time.perf_counter()
    silence = np.zeros(32_000, dtype=np.float32)
    if args.mode == "incremental":
        from qwen_incremental_kv_stream import IncrementalKVStreamingASR
        IncrementalKVStreamingASR(asr).warmup(2.0)
    else:
        asr.transcribe(audio=(silence, 16_000), language=None)
    return time.perf_counter() - started


def run_clip(asr, audio: np.ndarray, args) -> tuple[str, list[float], float]:
    chunk_samples = max(1, int(round(args.input_chunk_sec * 16_000)))
    latencies: list[float] = []
    max_window_sec = 0.0

    if args.mode == "official":
        state = asr.init_streaming_state(language=None, chunk_size_sec=args.update_sec)
        for offset in range(0, len(audio), chunk_samples):
            previous_chunk_id = state.chunk_id
            start = time.perf_counter()
            asr.streaming_transcribe(audio[offset : offset + chunk_samples], state)
            elapsed = time.perf_counter() - start
            if state.chunk_id > previous_chunk_id:
                latencies.append(elapsed)
            max_window_sec = max(max_window_sec, len(state.audio_accum) / 16_000)
        previous_chunk_id = state.chunk_id
        start = time.perf_counter()
        asr.finish_streaming_transcribe(state)
        elapsed = time.perf_counter() - start
        if state.chunk_id > previous_chunk_id:
            latencies.append(elapsed)
        return state.text or "", latencies, max_window_sec

    if args.mode == "rolling":
        from qwen_fixed_window_stream import FixedWindowStreamingASR
        stream = FixedWindowStreamingASR(
            model=asr, language=None, window_seconds=args.context_sec,
            update_seconds=args.update_sec,
        )
    else:
        from qwen_incremental_kv_stream import IncrementalKVStreamingASR
        stream = IncrementalKVStreamingASR(
            asr, language=None, window_seconds=args.context_sec,
            update_seconds=args.update_sec, overlap_seconds=args.overlap_sec,
            rollback_tokens=args.rollback_tokens,
            feature_mean_threshold=args.feature_mean_threshold,
            feature_changed_fraction_threshold=args.feature_changed_fraction_threshold,
            max_new_tokens=args.max_new_tokens,
        )

    for offset in range(0, len(audio), chunk_samples):
        for update in stream.feed(audio[offset : offset + chunk_samples]):
            latencies.append(update.decode_wall_sec)
            max_window_sec = max(max_window_sec, update.window_duration_sec)
    final = stream.finish()
    if final.decode_wall_sec > 0:
        latencies.append(final.decode_wall_sec)
    max_window_sec = max(max_window_sec, final.window_duration_sec)
    return final.text or "", latencies, max_window_sec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("official", "rolling", "incremental"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--ids", default="all")
    parser.add_argument("--continuous", action="store_true", help="Join selected clips into one long stream")
    parser.add_argument("--gap-sec", type=float, default=0.2, help="Silence inserted between clips in continuous mode")
    parser.add_argument("--context-sec", type=float, default=30.0)
    parser.add_argument("--overlap-sec", type=float, default=2.0)
    parser.add_argument("--update-sec", type=float, default=2.0)
    parser.add_argument("--input-chunk-sec", type=float, default=0.5)
    parser.add_argument("--rollback-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--feature-mean-threshold", type=float, default=0.10)
    parser.add_argument("--feature-changed-fraction-threshold", type=float, default=0.90)
    args = parser.parse_args()
    if abs(args.update_sec - 2.0) > 1e-6:
        raise ValueError("This matched profile fixes the ASR update interval at 2 seconds")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids.lower() != "all":
        wanted = set(args.ids.split(","))
        rows = [row for row in rows if str(row["id"]) in wanted]
    if not rows:
        raise ValueError("No manifest rows selected")

    asr = make_model(args)
    warmup_sec = warmup(asr, args)
    records = []
    all_latencies: list[float] = []
    total_audio_sec = 0.0
    total_wall_sec = 0.0
    max_window_sec = 0.0
    with GPUMemorySampler(args.gpu_index) as sampler:
        if args.continuous:
            audio_parts = []
            references = []
            gap = np.zeros(max(0, int(round(args.gap_sec * 16_000))), dtype=np.float32)
            for row in rows:
                audio, rate = sf.read(Path(row["audio"]), dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if rate != 16_000:
                    raise ValueError(f"Expected 16 kHz audio for {row['id']}, got {rate}")
                if audio_parts and len(gap):
                    audio_parts.append(gap)
                audio_parts.append(audio)
                references.append(str(row["reference"]))
            audio = np.concatenate(audio_parts)
            reference = " ".join(references)
            started = time.perf_counter()
            hypothesis, latencies, window_sec = run_clip(asr, audio, args)
            wall = time.perf_counter() - started
            errors = edit_distance(normalize(reference), normalize(hypothesis))
            ref_chars = len(normalize(reference))
            records.append({
                "id": "continuous-" + "-".join(str(row["id"]) for row in rows),
                "errors": errors, "reference_chars": ref_chars,
                "cer": errors / max(1, ref_chars), "duration_sec": len(audio) / 16_000,
                "wall_sec": wall, "update_latencies_sec": latencies,
                "max_audio_window_sec": window_sec, "hypothesis": hypothesis,
                "reference": reference,
            })
            all_latencies.extend(latencies)
            total_audio_sec = len(audio) / 16_000
            total_wall_sec = wall
            max_window_sec = window_sec
            print(json.dumps({"mode": args.mode, "id": records[-1]["id"],
                              "cer": records[-1]["cer"], "updates": len(latencies),
                              "wall_sec": wall}, ensure_ascii=False), flush=True)
        else:
            for row in rows:
                audio_path = Path(row["audio"])
                audio, rate = sf.read(audio_path, dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if rate != 16_000:
                    raise ValueError(f"Expected 16 kHz audio for {row['id']}, got {rate}")
                started = time.perf_counter()
                hypothesis, latencies, window_sec = run_clip(asr, audio, args)
                wall = time.perf_counter() - started
                errors = edit_distance(normalize(row["reference"]), normalize(hypothesis))
                ref_chars = len(normalize(row["reference"]))
                records.append({
                    "id": str(row["id"]), "errors": errors, "reference_chars": ref_chars,
                    "cer": errors / max(1, ref_chars), "duration_sec": len(audio) / rate,
                    "wall_sec": wall, "update_latencies_sec": latencies,
                    "max_audio_window_sec": window_sec, "hypothesis": hypothesis,
                    "reference": row["reference"],
                })
                all_latencies.extend(latencies)
                total_audio_sec += len(audio) / rate
                total_wall_sec += wall
                max_window_sec = max(max_window_sec, window_sec)
                print(json.dumps({"mode": args.mode, "id": row["id"], "cer": records[-1]["cer"],
                                  "updates": len(latencies), "wall_sec": wall}, ensure_ascii=False), flush=True)

    total_errors = sum(row["errors"] for row in records)
    total_ref_chars = sum(row["reference_chars"] for row in records)
    midpoint = len(all_latencies) // 2
    summary = {
        "mode": args.mode, "model": args.model, "items": len(records),
        "context_sec": args.context_sec, "overlap_sec": args.overlap_sec,
        "update_sec": args.update_sec, "input_chunk_sec": args.input_chunk_sec,
        "max_new_tokens": args.max_new_tokens, "language": "auto",
        "duration_sec": total_audio_sec, "wall_sec_excluding_load_and_warmup": total_wall_sec,
        "rtf": total_wall_sec / max(total_audio_sec, 1e-9),
        "errors": total_errors, "reference_chars": total_ref_chars,
        "corpus_cer": total_errors / max(total_ref_chars, 1),
        "macro_cer": statistics.mean(row["cer"] for row in records),
        "updates": len(all_latencies),
        "update_p50_sec": statistics.median(all_latencies) if all_latencies else 0.0,
        "update_p95_sec": float(np.percentile(all_latencies, 95)) if all_latencies else 0.0,
        "update_max_sec": max(all_latencies, default=0.0),
        "updates_over_2sec": sum(value > 2.0 for value in all_latencies),
        "first_half_update_p50_sec": statistics.median(all_latencies[:midpoint]) if args.continuous and midpoint else None,
        "second_half_update_p50_sec": statistics.median(all_latencies[midpoint:]) if args.continuous and midpoint else None,
        "max_audio_window_sec": max_window_sec, "warmup_sec": warmup_sec,
        "gpu_sampled_peak_mib": max(sampler.samples_mib, default=0),
        "host_process_tree_rss_peak_mib": max(sampler.host_rss_mib, default=0.0),
        "host_process_tree_uss_peak_mib": max(sampler.host_uss_mib, default=0.0),
        "records": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
