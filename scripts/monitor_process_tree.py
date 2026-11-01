"""Sample host RSS/USS for one ASR process and its child workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--period-sec", type=float, default=0.5)
    args = parser.parse_args()
    rss_samples = []
    uss_samples = []
    while True:
        try:
            root = psutil.Process(args.pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            break
        rss = uss = 0
        for process in processes:
            try:
                memory = process.memory_full_info()
                rss += memory.rss
                uss += getattr(memory, "uss", memory.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        rss_samples.append(rss / (1024**2))
        uss_samples.append(uss / (1024**2))
        time.sleep(args.period_sec)
    output = {
        "pid": args.pid,
        "sampling_period_sec": args.period_sec,
        "samples": len(rss_samples),
        "process_tree_rss_peak_mib": max(rss_samples, default=0.0),
        "process_tree_uss_peak_mib": max(uss_samples, default=0.0),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
