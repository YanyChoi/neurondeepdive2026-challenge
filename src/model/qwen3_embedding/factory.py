# SPDX-License-Identifier: Apache-2.0
"""Factory for the NDD Day-2 Challenge Qwen3-Embedding onboarding (Stage 2).

Registered under the architecture string ``"Qwen3ForCausalLM"`` because that is
what ``Qwen/Qwen3-Embedding-8B``'s config.json declares (the embedding
checkpoints reuse the generative architecture name). The pooling runner
(``--runner pooling``) is what actually routes to the embedding model.
"""

import logging

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

logger = logging.getLogger(__name__)


class Qwen3EmbeddingForCausalLM(nn.Module):
    """Factory that validates config and selects the challenge implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config.runner_type == "pooling":
            # A native sentence-transformers embedding checkpoint (has
            # modules.json, e.g. Qwen3-Embedding-8B) gets the hand-written
            # pooling model; a generative checkpoint run with --convert embed
            # goes through the plugin's generic pooling adapter instead.
            from vllm.transformers_utils.config import get_pooling_config

            is_native_st_embedding = (
                get_pooling_config(model_config.model, model_config.revision)
                is not None
            )
            if is_native_st_embedding:
                from .model_embedding import Qwen3ForEmbedding as Model

                logger.info(
                    "[NDD-D2-CHALLENGE] Using challenge Qwen3ForEmbedding "
                    "implementation (pooling runner, native ST checkpoint)"
                )
            else:
                from .model import Qwen3ForCausalLM as Model

                logger.info(
                    "[NDD-D2-CHALLENGE] Pooling runner on a generative "
                    "checkpoint — building challenge backbone for --convert"
                )
        else:
            from .model import Qwen3ForCausalLM as Model

            logger.info(
                "[NDD-D2-CHALLENGE] Using challenge Qwen3ForCausalLM "
                "(generative runner)"
            )

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Qwen3-Embedding ships BF16 only; reject quantized configs up front."""
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for "
                "Qwen3-Embedding-8B. Only BF16 (quantization=None or 'bf16') "
                "is implemented."
            )
