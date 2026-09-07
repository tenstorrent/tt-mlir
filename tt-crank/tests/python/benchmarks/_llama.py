# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared decoder-LM helpers for the LLM decode benchmark.

Provides the model loader, the on-device sampling wrapper, and the StaticCache
initialization used by the generate-loop benchmark.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StaticCache

_DTYPE = torch.bfloat16

# Fixed benchmark prompt. The prompt is held constant (typo included) so the
# generated token stream is reproducible run-to-run.
BENCHMARK_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)


def load_model(model_id: str, *, num_layers: int | None = None) -> torch.nn.Module:
    """Load a causal-LM in eval mode, skipping the test if unavailable.

    `num_layers` truncates the decoder stack (smoke-run knob): fewer layers
    compile and run faster while still exercising the full per-layer code path.
    Like tt-xla, the layer count is set on the config *before* instantiation so
    HF only builds and loads the kept layers, rather than allocating the full
    stack and weights and slicing the tail off afterward.
    """
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as e:
        pytest.skip(f"could not load config for {model_id!r}: {e}")
    # Multimodal configs nest the decoder count under text_config; plain causal
    # LMs expose it at the root. get_text_config() returns whichever applies.
    layer_cfg = (
        config.get_text_config() if hasattr(config, "get_text_config") else config
    )
    if num_layers is not None:
        layer_cfg.num_hidden_layers = num_layers
    # Force every layer to full attention so all models exercise one attention
    # path (some configs declare per-layer sliding-window variants). Sized to the
    # (possibly truncated) layer count, since setting num_hidden_layers does not
    # recompute a pre-built layer_types list.
    if getattr(layer_cfg, "layer_types", None):
        layer_cfg.layer_types = ["full_attention"] * layer_cfg.num_hidden_layers
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, dtype=_DTYPE
        ).eval()
    except Exception as e:
        pytest.skip(f"could not load {model_id!r}: {e}")
    return model


def load_tokenizer(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        pytest.skip(f"could not load tokenizer for {model_id!r}: {e}")


class LLMSamplingWrapper(torch.nn.Module):
    """Keep token selection and cache-position increment on device.

    Between decode steps only the next token crosses to host, never logits or
    positions, so the per-step host fence stays cheap. With
    ``return_logits=True`` the forward also returns full logits (used by the
    PCC check only - transferring [B, seq, vocab] every step would dominate
    perf timings).
    """

    def __init__(self, model: torch.nn.Module, return_logits: bool = False):
        super().__init__()
        self.model = model
        self.return_logits = return_logits

    def forward(self, input_ids, past_key_values, cache_position):
        position_ids = cache_position.unsqueeze(0)
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        # Last position only: prefill takes the final prompt token; decode no-op.
        next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        next_cache_position = cache_position[-1:] + 1
        if self.return_logits:
            return next_token, next_cache_position, out.logits
        return next_token, next_cache_position


def init_static_cache(
    config,
    *,
    batch_size: int,
    max_cache_len: int,
    device: torch.device | str,
    dtype: torch.dtype = _DTYPE,
) -> StaticCache:
    """Pre-allocate a StaticCache directly on ``device``.

    head_dim and num_key_value_heads come from the model config (with the
    standard fallbacks) so the cache matches the model's attention shape.
    """
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    return cache
