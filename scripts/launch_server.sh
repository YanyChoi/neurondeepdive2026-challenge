#!/usr/bin/env bash
# Stage 3 — Compile & smoke test: launch the online embedding server.
# Flags follow the official Qwen3-Embedding recipe, sized for trn2.3xlarge (TP=4).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"

# Qwen3-Embedding compilation/execution timeouts (per the official tutorial).
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
# trn2.3xlarge has no EFA; affinity is a CPU perf optimization only.
export NEURON_SKIP_EFA_AFFINITY=1

# --runner pooling is REQUIRED: the checkpoint declares
# architectures=["Qwen3ForCausalLM"], so without it vLLM loads it as a
# generative model. Prefix caching off: short single-shot embedding requests
# don't share prefixes, and segmented prefill would only add overhead.
exec vllm serve "$MODEL" \
    --runner pooling \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    --tensor-parallel-size "$TP" \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096]
        }
    }'
