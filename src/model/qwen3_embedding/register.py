# SPDX-License-Identifier: Apache-2.0
"""Stage 2 — Register (모델 등록), exactly as the onboarding guide shows.

The string key must match the ``architectures`` field of the model's HF
config.json — for Qwen3-Embedding-8B that is ``"Qwen3ForCausalLM"`` (the
embedding checkpoints reuse the generative architecture name; the pooling
runner selects the embedding path at load time).

NOTE (why scripts/install_into_plugin.py also exists): calling this at import
time is not sufficient on its own with the Neuron plugin, because
``NeuronWorker.__init__`` re-registers every entry of
``vllm_neuron.model.registry`` in each worker process, overwriting earlier
registrations of the same key. The install script therefore places this model
package inside the installed plugin and points that registry at it, so the
worker's own re-registration installs *this* factory.
"""

from vllm import ModelRegistry

from .factory import Qwen3EmbeddingForCausalLM


def register() -> None:
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",  # ← HF config.json의 "architectures" 필드와 일치
        Qwen3EmbeddingForCausalLM,
    )
