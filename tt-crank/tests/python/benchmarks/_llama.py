"""Shared decoder-LM helpers for the LLM decode benchmark.

Provides the model loader, the on-device sampling wrapper, and the StaticCache
initialization used by the generate-loop benchmark.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

_DTYPE = torch.bfloat16

# Fixed benchmark prompt. The prompt is held constant (typo included) so the
# generated token stream is reproducible run-to-run.
BENCHMARK_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)


def load_model(model_id: str) -> torch.nn.Module:
    """Load a causal-LM in eval mode, skipping the test if unavailable."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_DTYPE).eval()
    except Exception as e:
        pytest.skip(f"could not load {model_id!r}: {e}")
    # Force every layer to full attention so all models exercise one attention
    # path (some configs declare per-layer sliding-window variants).
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
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
