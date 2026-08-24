#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 4 — Accuracy: MTEB STS12 / NFCorpus / SciFact against the server.

Runs the MTEB harness with a model wrapper that calls the running
/v1/embeddings endpoint, applying the official Qwen3 per-task query
instruction (``Instruct: {task}\\nQuery: {text}``; documents are embedded
bare). Compares against the published MTEB scores for Qwen3-Embedding-8B and
the Neuron BF16 reference from the official model recipe.

    pip install "mteb>=1.25" numpy requests
    python3 scripts/eval_mteb.py [--base-url http://localhost:8000] \
        [--tasks STS12 NFCorpus SciFact]
"""

import argparse
import json
import sys
import time

import numpy as np
import requests

MODEL = "Qwen/Qwen3-Embedding-8B"
BATCH = 32

# Official per-task query instructions (Qwen3-Embedding evaluation setup).
TASK_INSTRUCTIONS = {
    "STS12": "Retrieve semantically similar text.",
    "NFCorpus": "Given a question, retrieve relevant documents that best "
    "answer the question",
    "SciFact": "Given a scientific claim, retrieve documents that support or "
    "refute the claim",
}

# main score, published MTEB score, Neuron BF16 score from the official recipe
REFERENCE = {
    "STS12": ("cosine_spearman", 0.8614, 0.8639),
    "NFCorpus": ("ndcg_at_10", 0.4145, 0.4143),
    "SciFact": ("ndcg_at_10", 0.7846, 0.7839),
}


class RemoteQwen3Embedding:
    """MTEB model wrapper over the OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i : i + BATCH]
            for attempt in range(5):
                try:
                    r = self.session.post(
                        f"{self.base_url}/v1/embeddings",
                        json={"model": MODEL, "input": chunk},
                        timeout=600,
                    )
                    r.raise_for_status()
                    data = r.json()["data"]
                    out.extend(
                        d["embedding"] for d in sorted(data, key=lambda d: d["index"])
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    print(f"  retry {attempt + 1} after error: {e}", file=sys.stderr)
                    time.sleep(5)
            done = min(i + BATCH, len(texts))
            if done % (BATCH * 20) < BATCH or done == len(texts):
                print(f"  embedded {done}/{len(texts)}", file=sys.stderr)
        return np.asarray(out, dtype=np.float32)

    def encode(self, sentences, task_name=None, prompt_type=None, **kwargs):
        # Queries (and STS sentences, which MTEB passes without prompt_type or
        # as queries) get the instruction; corpus documents are embedded bare.
        ptype = getattr(prompt_type, "value", prompt_type)
        inst = TASK_INSTRUCTIONS.get(task_name)
        if inst and ptype != "passage" and ptype != "document":
            sentences = [f"Instruct: {inst}\nQuery: {s}" for s in sentences]
        return self._embed(list(sentences))


def extract_main_score(task_result, metric: str) -> float:
    scores = task_result.scores  # {split: [{... metric ...}]}
    split = "test" if "test" in scores else next(iter(scores))
    entry = scores[split][0]
    return float(entry.get("main_score", entry.get(metric)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--tasks", nargs="+", default=["STS12", "NFCorpus", "SciFact"])
    ap.add_argument("--output", default="results/mteb")
    args = ap.parse_args()

    import os

    import mteb

    os.makedirs("results", exist_ok=True)
    model = RemoteQwen3Embedding(args.base_url)
    rows = []
    for task_name in args.tasks:
        print(f"=== {task_name} ===")
        tasks = mteb.get_tasks(tasks=[task_name])
        evaluation = mteb.MTEB(tasks=tasks)
        t0 = time.time()
        results = evaluation.run(
            model, output_folder=args.output, overwrite_results=True
        )
        metric, published, neuron_ref = REFERENCE[task_name]
        score = extract_main_score(results[0], metric)
        rows.append((task_name, metric, score, published, neuron_ref))
        print(f"{task_name}: {metric}={score:.4f}  ({time.time() - t0:.0f}s)")

    print("\nRESULTS")
    print(f"{'task':<10}{'metric':<18}{'ours':>8}{'published':>11}{'neuron ref':>12}{'Δ vs pub':>10}")
    ok = True
    for task, metric, score, pub, ref in rows:
        delta = score - pub
        ok &= abs(delta) < 0.005
        print(f"{task:<10}{metric:<18}{score:>8.4f}{pub:>11.4f}{ref:>12.4f}{delta:>+10.4f}")
    print("\nPASS (all within 0.005 of published)" if ok else "\nWARN: score gap > 0.005")

    with open("results/mteb_summary.json", "w") as f:
        json.dump(
            [
                {"task": t, "metric": m, "score": s, "published": p, "neuron_ref": r}
                for t, m, s, p, r in rows
            ],
            f,
            indent=2,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
