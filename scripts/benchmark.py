#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — Benchmark: embedding throughput and latency.

Closed-loop benchmark against /v1/embeddings: C concurrent workers each send
requests of B texts x ~L words until N total requests complete. Reports
requests/sec, embeddings/sec, and latency percentiles.

    python3 scripts/benchmark.py --concurrency 8 --batch-size 8 \
        --num-requests 200 --words 64
"""

import argparse
import json
import random
import statistics
import threading
import time

import requests

MODEL = "Qwen/Qwen3-Embedding-8B"
WORDS = (
    "market neighborhood trade karrot neuron trainium embedding vector index "
    "retrieval search query document semantic latency throughput benchmark "
    "compile kernel tensor parallel pooling normalize token sequence batch"
).split()


def make_text(words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="texts per request")
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--words", type=int, default=64, help="words per text")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(42)
    payloads = [
        [make_text(args.words, rng) for _ in range(args.batch_size)]
        for _ in range(args.num_requests + args.warmup)
    ]

    lock = threading.Lock()
    next_idx = 0
    latencies: list[float] = []
    errors = 0
    t_start = None

    def worker():
        nonlocal next_idx, errors, t_start
        session = requests.Session()
        while True:
            with lock:
                idx = next_idx
                if idx >= len(payloads):
                    return
                next_idx += 1
            t0 = time.perf_counter()
            try:
                r = session.post(
                    f"{args.base_url}/v1/embeddings",
                    json={"model": MODEL, "input": payloads[idx]},
                    timeout=600,
                )
                r.raise_for_status()
            except Exception:  # noqa: BLE001
                with lock:
                    errors += 1
                continue
            dt = time.perf_counter() - t0
            with lock:
                if idx >= args.warmup:
                    if t_start is None:
                        t_start = t0
                    latencies.append(dt)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t_wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - (t_start or t_wall0)

    n = len(latencies)
    lat_sorted = sorted(latencies)
    result = {
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "words_per_text": args.words,
        "completed_requests": n,
        "errors": errors,
        "wall_seconds": round(wall, 2),
        "requests_per_sec": round(n / wall, 2),
        "embeddings_per_sec": round(n * args.batch_size / wall, 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 1),
            "p50": round(lat_sorted[n // 2] * 1000, 1),
            "p95": round(lat_sorted[int(n * 0.95)] * 1000, 1),
            "p99": round(lat_sorted[min(int(n * 0.99), n - 1)] * 1000, 1),
        },
    }
    print(json.dumps(result, indent=2))
    with open(
        f"results/bench_c{args.concurrency}_b{args.batch_size}_w{args.words}.json",
        "w",
    ) as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
