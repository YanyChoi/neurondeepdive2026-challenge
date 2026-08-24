#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 — Smoke test the /v1/embeddings server.

Checks, in order:
  1. /health returns 200
  2. /v1/embeddings returns one 4096-dim vector per input
  3. vectors are L2-normalized (dot product == cosine similarity)
  4. semantic sanity: a paraphrase pair scores higher than an unrelated pair
     (with the official Qwen3 query-instruction template on the query side)

Usage: python3 scripts/smoke_test.py [--base-url http://localhost:8000]
"""

import argparse
import json
import math
import sys
import urllib.request

MODEL = "Qwen/Qwen3-Embedding-8B"
INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)


def post_embeddings(base_url: str, inputs: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        f"{base_url}/v1/embeddings",
        data=json.dumps({"model": MODEL, "input": inputs}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    # 1. health
    with urllib.request.urlopen(f"{args.base_url}/health", timeout=30) as r:
        assert r.status == 200, r.status
    print("[1/4] /health OK")

    # 2. shape
    query = INSTRUCT + "What is the capital of France?"
    docs = [
        "Paris is the capital and most populous city of France.",
        "The mitochondria is the powerhouse of the cell.",
    ]
    vecs = post_embeddings(args.base_url, [query] + docs)
    dims = {len(v) for v in vecs}
    assert dims == {4096}, f"expected 4096-dim embeddings, got {dims}"
    print(f"[2/4] {len(vecs)} embeddings, dim=4096 OK")

    # 3. L2 norm ~ 1
    norms = [math.sqrt(dot(v, v)) for v in vecs]
    assert all(abs(n - 1.0) < 1e-2 for n in norms), norms
    print(f"[3/4] L2-normalized OK (norms={[round(n, 4) for n in norms]})")

    # 4. semantic ordering
    sim_rel = dot(vecs[0], vecs[1])
    sim_unrel = dot(vecs[0], vecs[2])
    print(f"[4/4] cos(query, Paris doc)={sim_rel:.4f}  "
          f"cos(query, mitochondria doc)={sim_unrel:.4f}")
    assert sim_rel > sim_unrel + 0.1, (
        "relevant document did not clearly outrank the irrelevant one"
    )
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
