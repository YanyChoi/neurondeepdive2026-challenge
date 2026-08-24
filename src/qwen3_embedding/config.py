# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Embedding Config (NDD Day-2 Challenge)
============================================

Ported from the vllm-neuron ``llama3/config.py`` template (Stage 1 of the
model-onboarding guide). Defaults below are the exact values from
``Qwen/Qwen3-Embedding-8B``'s config.json — see ``step0_arch_diff.py`` for the
full field-by-field diff against Llama-3.1-8B.

Key differences from the Llama template (found in Step 0):
  - QK-norm applied per-head (q_norm / k_norm on head_dim)   -> handled in model.py
  - Standard RoPE, rope_theta = 1_000_000, no rope_scaling   (Llama3: 500k + scaling)
  - rms_norm_eps = 1e-6                                      (Llama3: 1e-5)
  - vocab_size = 151665, intermediate_size = 12288, 36 layers
  - Embedding checkpoint: no lm_head at all                  -> model_embedding.py
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3EmbeddingConfig:
    # <-- MODEL-SPECIFIC: Qwen/Qwen3-Embedding-8B architecture parameters
    vocab_size: int = 151665
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)


# The model/embedding modules were written against the name ``Qwen3Config``
# in the upstream template; keep an alias so the port stays a minimal diff.
Qwen3Config = Qwen3EmbeddingConfig
