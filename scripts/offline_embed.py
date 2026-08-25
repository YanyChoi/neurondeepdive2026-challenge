#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 (offline path) — batch embedding with LLM.embed().

Same model and config as the online server, reached in-process. Use this for
bulk index building; per the tutorial both paths produce identical vectors.

    python3 scripts/offline_embed.py
"""

import os

os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1200"
os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
# trn2.3xlarge has no EFA; affinity is a CPU perf optimization only.
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

from vllm import LLM  # noqa: E402


def main() -> None:
    llm = LLM(
        model="Qwen/Qwen3-Embedding-8B",
        runner="pooling",
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=int(os.environ.get("TP", "4")),
        # Short embedding requests share no prefix; single-shot prefill
        # (max_num_batched_tokens == max_model_len) requires APC off.
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096]
            }
        },
    )

    docs = [
        "Retrieval-augmented generation grounds answers in retrieved documents.",
        "def add(a, b):\n    return a + b",
        "당근마켓은 동네 이웃 간의 중고거래 플랫폼입니다.",
        "Photosynthesis converts sunlight, water, and CO2 into glucose.",
    ]
    for doc, out in zip(docs, llm.embed(docs)):
        vec = out.outputs.embedding
        print(f"dim={len(vec)}  first3={[round(x, 4) for x in vec[:3]]}  | {doc[:40]}")


if __name__ == "__main__":
    main()
