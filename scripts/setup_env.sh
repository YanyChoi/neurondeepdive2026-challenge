#!/usr/bin/env bash
# Environment setup on a trn2.3xlarge. Two supported paths:
#
#   A) Official vLLM Inference NeuronX DLC (what the challenge specifies):
#        ./scripts/setup_env.sh dlc <image-uri>
#      Image list: https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx
#      (ECR login for 763104351884.dkr.ecr.<region>.amazonaws.com required.)
#
#   B) Source install of the plugin (release-0.24), when the DLC is not
#      reachable from the instance:
#        ./scripts/setup_env.sh venv
set -euo pipefail

MODE="${1:-venv}"
BRANCH="release-0.24.0.1.1.0"

if [ "$MODE" = "dlc" ]; then
    IMAGE="${2:?usage: setup_env.sh dlc <image-uri>}"
    docker pull "$IMAGE"
    # Host network for port 8000; all Neuron devices; repo mounted at /challenge.
    exec docker run -it --name qwen3-embedding \
        --network host \
        $(ls /dev/neuron* | sed 's/^/--device /') \
        -v "$(cd "$(dirname "$0")/.." && pwd)":/challenge \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        -e VLLM_NEURON_COMPILATION_TIMEOUT=1200 \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
        "$IMAGE" bash
fi

# ── Source path ──────────────────────────────────────────────────────────────
VENV="$HOME/vllm-neuron-venv"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip

git -C "$HOME" clone --depth 1 --branch "$BRANCH" \
    https://github.com/vllm-project/vllm-neuron.git 2>/dev/null || true
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    -e "$HOME/vllm-neuron"

# Neuron compiler stack, pinned to the versions the official DLC for this
# release ships (deep-learning-containers vllm/inference/0.24.0.1.1.0).
# Do NOT install torch-neuronx: this release uses libtorch-neuronx-lite
# (pulled in above), and torch-neuronx would downgrade torch/torch-xla to 2.9.
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    "neuronx-cc==2.27.5334.0+f702b353" \
    "nki==0.6.0+31049202112.g85070674"

# islpy 2026.2.1 (Aug 2026) changed pybind signatures and crashes the
# neuronx-cc 2.27 Simplifier with:
#   [NCC_ISMP902] Simplifier error: is_subset(): incompatible function arguments
# Pin the version the DLC was built against.
pip install "islpy==2026.1"

python -c "import vllm; from vllm.platforms import current_platform; print('platform:', current_platform.device_name)"
echo "OK — activate with: source $VENV/bin/activate"
