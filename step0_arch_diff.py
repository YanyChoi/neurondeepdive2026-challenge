#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 0 — Architecture diff: Llama-3.1-8B vs Qwen/Qwen3-Embedding-8B.

Compares the two HF config.json files field by field and prints the structural
differences that drive the porting work in src/qwen3_embedding/ (which llama3/
template code must change, and which code carries over untouched).

Usage:
    python3 step0_arch_diff.py            # uses the bundled configs in configs/
    python3 step0_arch_diff.py --fetch    # fetches both live from huggingface.co
                                          # (Llama is gated: needs HF_TOKEN)
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LLAMA_ID = "meta-llama/Llama-3.1-8B"
QWEN_ID = "Qwen/Qwen3-Embedding-8B"

# config key -> which part of the port it affects
IMPLICATIONS = {
    "num_hidden_layers": "config.py defaults / KV spec (36 vs 32 layers)",
    "intermediate_size": "config.py — MLP weight shapes (12288 vs 14336)",
    "vocab_size": "config.py — embed_tokens shape (151665 vs 128256)",
    "max_position_embeddings": "config.py — max context (40960 vs 131072)",
    "rms_norm_eps": "config.py — every RMSNorm (1e-6 vs 1e-5)",
    "rope_theta": "model.py RotaryEmbedding — base frequency (1e6 vs 5e5)",
    "rope_scaling": "model.py — DELETE the Llama-3.1 piecewise rope scaling: "
    "Qwen3 uses standard rotate_half RoPE",
    "model_type": "registry — but NOTE architectures is still Qwen3ForCausalLM",
    "bos_token_id": "tokenizer only, no model code change",
    "eos_token_id": "tokenizer only, no model code change",
    "transformers_version": "(irrelevant)",
    "head_dim": "config.py — explicit in Qwen3 (128, same derived value)",
    "sliding_window": "not used (use_sliding_window=false) — no attention change",
    "use_sliding_window": "not used — no attention change",
    "max_window_layers": "not used — no attention change",
    "attention_bias": "same (false) — QKV/O proj stay bias-free like Llama",
}

# Structural facts that config.json alone cannot show. Sources: the two
# checkpoints' weight indices (model.safetensors.index.json) and Qwen's
# sentence-transformers metadata (modules.json).
STRUCTURE_NOTES = """
STRUCTURAL DIFF (beyond config.json)
------------------------------------
1. Per-head QK-norm — THE model-code diff.
   Qwen3 checkpoints carry per-layer weights Llama does not have:
       layers.N.self_attn.q_norm.weight   [head_dim=128]
       layers.N.self_attn.k_norm.weight   [head_dim=128]
   RMSNorm over head_dim, applied per head BEFORE RoPE.
   -> model.py: add q_norm/k_norm modules + apply in forward_prefill;
      decode megakernel flag rmsnorm_QK_pre_rope_enabled=True.
   -> load_weights: add the two mappings (replicated, NOT TP-sharded).

2. No LM head. Qwen3-Embedding-8B ships no lm_head.weight; its weight keys
   also drop the "model." prefix ("embed_tokens.weight", not
   "model.embed_tokens.weight").
   -> model_embedding.py: drop lm_head entirely; adjust checkpoint-side
      mapping keys.

3. Embedding output stage. The checkpoint is a sentence-transformers model:
   modules.json = Transformer -> Pooling(lasttoken) -> Normalize.
   -> forward returns the flattened [T, H] post-norm hidden states; vLLM's
      pooling runner does LAST-token gather + L2 normalize (DispatchPooler).
   -> serve with --runner pooling. Prefill-only: no decode graph at all.

4. architectures in config.json is still ["Qwen3ForCausalLM"].
   -> Stage 2 registers our factory under that exact string; the pooling
      runner (--runner pooling), not the arch name, selects the embedding path.

WHAT CARRIES OVER FROM THE llama3/ TEMPLATE UNCHANGED
-----------------------------------------------------
- TP head sharding & GQA replication (both are 8-KV-head GQA)
- SP all-gather / reduce-scatter around attention and MLP
- SiLU gate/up/down MLP structure (only dims change)
- Standard RMSNorm layers (input/post-attention/final), only eps changes
- KV-cache block layout, weight-loader utilities, sampler (unused for pooling)
"""


def load_config(local: Path, repo_id: str, fetch: bool) -> dict:
    if fetch:
        # Same path as the official arch_diff_analysis.py: AutoConfig honors
        # HF auth (HF_TOKEN / HUGGING_FACE_HUB_TOKEN env, or the token stored
        # by `hf auth login`), which the gated Llama repo requires.
        try:
            from transformers import AutoConfig

            return AutoConfig.from_pretrained(repo_id).to_dict()
        except ImportError:
            url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
            req = urllib.request.Request(url)
            import os

            tok = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req) as r:
                return json.load(r)
    return json.loads(local.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch configs from HF hub")
    args = ap.parse_args()

    llama = load_config(HERE / "configs/llama-3.1-8b.config.json", LLAMA_ID, args.fetch)
    qwen = load_config(
        HERE / "configs/qwen3-embedding-8b.config.json", QWEN_ID, args.fetch
    )

    # Derived field Llama leaves implicit
    llama.setdefault(
        "head_dim", llama["hidden_size"] // llama["num_attention_heads"]
    )

    keys = sorted(set(llama) | set(qwen))
    same, diff = [], []
    for k in keys:
        lv, qv = llama.get(k, "—"), qwen.get(k, "—")
        (same if lv == qv else diff).append((k, lv, qv))

    w = max(len(k) for k, *_ in diff) + 2
    print(f"CONFIG DIFF  {LLAMA_ID}  vs  {QWEN_ID}")
    print("=" * 100)
    print(f"{'field':<{w}}{'Llama-3.1-8B':<38}{'Qwen3-Embedding-8B'}")
    print("-" * 100)
    for k, lv, qv in diff:
        print(f"{k:<{w}}{json.dumps(lv):<38}{json.dumps(qv)}")
        note = IMPLICATIONS.get(k)
        if note:
            print(f"{'':<{w}}-> {note}")
    print("-" * 100)
    print(f"identical fields ({len(same)}): {', '.join(k for k, *_ in same)}")
    print(STRUCTURE_NOTES)
    print_module_structures(llama, qwen)
    return 0


def print_module_structures(llama_cfg: dict, qwen_cfg: dict) -> None:
    """Part 2 of the official onboarding Step 0 script: instantiate both
    models on the meta device (no weights, no memory) and print the module
    tree, so the structural diff (q_norm/k_norm, lm_head, ...) is observed
    rather than asserted.

    Unlike the official script we build the configs from the bundled JSONs
    instead of ``AutoConfig.from_pretrained`` — Llama-3.1-8B is gated on HF,
    and this works offline. Skipped gracefully if transformers/torch are not
    installed.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        print(f"\n[structure dump skipped — pip install transformers torch] ({e})")
        return

    print("\n" + "=" * 80)
    print("  2. 모델 구조 출력 (meta device, weight 로드 없음)")
    print("=" * 80)

    for name, cfg_dict in ((LLAMA_ID, llama_cfg), (QWEN_ID, qwen_cfg)):
        print(f"\n### {name} 구조 ###\n")
        try:
            cfg_dict = {k: v for k, v in cfg_dict.items() if k != "architectures"}
            config = AutoConfig.for_model(cfg_dict.pop("model_type"), **cfg_dict)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
            lines = str(model).split("\n")
            for line in lines[:60]:
                print(f"  {line}")
            if len(lines) > 60:
                print(f"  ... ({len(lines) - 60} more lines)")
            del model
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    sys.exit(main())
