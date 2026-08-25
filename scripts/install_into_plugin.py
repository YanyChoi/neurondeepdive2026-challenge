#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 — Register: install the challenge model into the vllm-neuron plugin.

The plugin resolves architectures through vLLM's ModelRegistry, but
NeuronWorker.__init__ re-registers every entry of vllm_neuron.model.registry
in each worker process, overwriting anything registered earlier. So the
reliable way to make OUR implementation the one that serves
"Qwen3ForCausalLM" is the one the onboarding guide describes: put the model
directory inside the installed plugin and point registry.py at it.

This script:
  1. copies src/qwen3_embedding/ -> <site-packages>/vllm_neuron/model/qwen3_embedding/
  2. rewrites registry.py's `from .qwen3 import Qwen3ForCausalLM` to import
     from .qwen3_embedding instead (idempotent; --uninstall restores it).

Run INSIDE the serving environment (DLC container or vllm-neuron venv):
    python3 scripts/install_into_plugin.py
    python3 scripts/install_into_plugin.py --uninstall
"""

import argparse
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "model" / "qwen3_embedding"
UPSTREAM_IMPORT = "from .qwen3 import Qwen3ForCausalLM"
CHALLENGE_IMPORT = (
    "from .qwen3_embedding import Qwen3ForCausalLM  # NDD-D2-CHALLENGE override"
)


def plugin_model_dir() -> Path:
    import vllm_neuron.model as m

    return Path(m.__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()

    model_dir = plugin_model_dir()
    registry = model_dir / "registry.py"
    target = model_dir / "qwen3_embedding"
    text = registry.read_text()

    if args.uninstall:
        if CHALLENGE_IMPORT in text:
            registry.write_text(text.replace(CHALLENGE_IMPORT, UPSTREAM_IMPORT))
        if target.exists():
            shutil.rmtree(target)
        print(f"[uninstall] restored {registry} and removed {target}")
        return 0

    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SRC, target)
    print(f"[install] copied {SRC} -> {target}")

    if CHALLENGE_IMPORT in text:
        print(f"[install] {registry} already patched")
    elif UPSTREAM_IMPORT in text:
        registry.write_text(text.replace(UPSTREAM_IMPORT, CHALLENGE_IMPORT))
        print(f"[install] patched {registry}: Qwen3ForCausalLM -> challenge impl")
    else:
        print(
            f"ERROR: expected import line not found in {registry}; "
            "plugin version mismatch?",
            file=sys.stderr,
        )
        return 1

    # sanity: import through the plugin exactly like the worker will
    from importlib import reload

    import vllm_neuron.model.registry as r

    reload(r)
    cls = dict(r.get_models())["Qwen3ForCausalLM"]
    assert cls.__module__.startswith("vllm_neuron.model.qwen3_embedding"), cls
    print(f"[install] verified: Qwen3ForCausalLM -> {cls.__module__}.{cls.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
