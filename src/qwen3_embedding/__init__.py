# SPDX-License-Identifier: Apache-2.0
"""NDD Day-2 Challenge: Qwen3-Embedding-8B onboarding for vLLM Neuron."""

from .factory import Qwen3EmbeddingForCausalLM

# Alias so a patched vllm_neuron registry can import the expected symbol name.
Qwen3ForCausalLM = Qwen3EmbeddingForCausalLM

__all__ = ["Qwen3EmbeddingForCausalLM", "Qwen3ForCausalLM"]
