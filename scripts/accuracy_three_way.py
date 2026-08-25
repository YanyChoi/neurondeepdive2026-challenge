#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 4 (L2) — Three-Way Comparison, per the onboarding guide.

The guide validates a generative model by comparing teacher-forced logits
three ways (Neuron vs HF FP32 vs HF BF16). The embedding-model analog
compares the final embedding vectors:

    target   = Neuron embeddings   (the live /v1/embeddings server)
    expected = HF FP32 embeddings  (gold standard, CPU)
    baseline = HF BF16 embeddings  (expected numerical-noise floor, CPU)

Interpretation (same as assert_close_three_way): Neuron error ~ BF16 error
means the port is numerically sound; Neuron error >> BF16 error means a bug.

    ~/mteb-venv/bin/python scripts/accuracy_three_way.py

Needs ~50GB RAM headroom for the FP32 pass (run on the instance).
"""

import argparse
import json
import sys

import numpy as np
import requests

MODEL = "Qwen/Qwen3-Embedding-8B"

PROMPTS = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "Retrieval-augmented generation grounds answers in retrieved documents.",
    "def add(a, b):\n    return a + b",
    "당근마켓은 동네 이웃 간의 중고거래 플랫폼입니다.",
    "Photosynthesis converts sunlight, water, and CO2 into glucose.",
    "The quarterly earnings report exceeded analyst expectations by 12%.",
    "Trainium2 exposes a 24MB SBUF per NeuronCore to NKI kernels.",
]


def neuron_embeddings(base_url: str) -> np.ndarray:
    r = requests.post(
        f"{base_url}/v1/embeddings",
        json={"model": MODEL, "input": PROMPTS},
        timeout=600,
    )
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return np.asarray([d["embedding"] for d in data], dtype=np.float32)


def hf_embeddings(dtype: str) -> np.ndarray:
    """Last-token pool + L2 normalize on CPU, the checkpoint's own recipe
    (modules.json: Transformer -> LAST Pooling -> Normalize)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch_dtype)
    model.eval()

    out = []
    with torch.no_grad():
        for text in PROMPTS:
            batch = tok(text, return_tensors="pt")
            hidden = model(**batch).last_hidden_state  # [1, T, H]
            vec = hidden[0, -1].float()  # LAST token pool
            vec = vec / vec.norm()  # L2 normalize
            out.append(vec.numpy())
    del model
    return np.asarray(out, dtype=np.float32)


def report(name: str, target: np.ndarray, expected: np.ndarray) -> dict:
    cos = (target * expected).sum(axis=1) / (
        np.linalg.norm(target, axis=1) * np.linalg.norm(expected, axis=1)
    )
    return {
        "pair": name,
        "max_abs_diff": float(np.abs(target - expected).max()),
        "mean_abs_diff": float(np.abs(target - expected).mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    print("[1/3] Neuron embeddings (server)...")
    neuron = neuron_embeddings(args.base_url)
    print("[2/3] HF BF16 embeddings (CPU)...")
    bf16 = hf_embeddings("bfloat16")
    print("[3/3] HF FP32 embeddings (CPU, gold standard)...")
    fp32 = hf_embeddings("float32")

    rows = [
        report("neuron_vs_fp32", neuron, fp32),   # target vs expected
        report("bf16_vs_fp32", bf16, fp32),       # baseline vs expected
        report("neuron_vs_bf16", neuron, bf16),
    ]
    for r in rows:
        print(
            f"{r['pair']:<16} max|Δ|={r['max_abs_diff']:.5f} "
            f"mean|Δ|={r['mean_abs_diff']:.6f} min_cos={r['min_cosine']:.6f}"
        )

    # Pass criterion (assert_close_three_way semantics): the Neuron error may
    # not exceed the BF16 numerical-noise floor by more than a small margin.
    neuron_err = rows[0]["max_abs_diff"]
    bf16_err = rows[1]["max_abs_diff"]
    ok = neuron_err <= bf16_err * 2.0 + 1e-4 and rows[0]["min_cosine"] > 0.999
    print(
        f"\n{'PASS' if ok else 'FAIL'}: neuron_err={neuron_err:.5f} vs "
        f"bf16_err={bf16_err:.5f} (allowed <= 2x + 1e-4), "
        f"min cos(neuron, fp32)={rows[0]['min_cosine']:.6f}"
    )

    import os

    os.makedirs("results", exist_ok=True)
    with open("results/three_way_summary.json", "w") as f:
        json.dump({"pass": ok, "rows": rows}, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
