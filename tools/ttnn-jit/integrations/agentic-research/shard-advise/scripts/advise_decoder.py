# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""`ttnn-advise capture` target for an agentic-research OptimizedDecoder.

The advisor traces by EXECUTING the ttnn function, so it needs the decoder built
with real weights and the decode inputs on device. This file exposes the two
symbols `ttnn-advise capture` expects:

  * `decode`            -- the ttnn function to trace (one decode step)
  * `make_inputs(device)` -- returns the positional args to call `decode` with

`make_inputs` reuses the experiment's OWN construction helpers
(`OptimizedDecoder.from_state_dict`, `prepare_decode_inputs`, `prepare_page_table`,
etc.) -- it is a thin bridge, not new logic. Edit the "experiment wiring" block
to match the model under test, then:

    source .agents/skills/shard-advise/scripts/bootstrap.sh
    ttnn-advise capture .agents/skills/shard-advise/scripts/advise_decoder.py:decode \
        --out /tmp/shard-advice 2>/dev/null

Read /tmp/shard-advice/report.json.
"""
import os
import sys

import torch
import ttnn

# ============================ EDIT: experiment wiring ========================
# Point these at the experiment's model package and describe one decode step.
MODEL_DIR = os.environ.get(
    "SHARD_ADVISE_MODEL_DIR",
    "experiment-13/code/llama31-8b/models/autoports/llama31-8b/tt",
)
LAYER_IDX = int(os.environ.get("SHARD_ADVISE_LAYER", "0"))
MAX_BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))
MAX_SEQ_LEN = int(os.environ.get("SHARD_ADVISE_SEQ", "128"))
PAGE_BLOCK = int(os.environ.get("SHARD_ADVISE_PAGE_BLOCK", "64"))


def _build(device):
    """Build (decoder, decode_kwargs, hidden) using the experiment's helpers.

    Return: (decoder, kwargs_without_hidden, hidden_tensor).
    Adjust the calls below to the model's actual signatures -- the names match
    the llama31-8b optimized_decoder.py; other models differ slightly.
    """
    sys.path.insert(0, MODEL_DIR)
    from optimized_decoder import OptimizedDecoder  # experiment's module

    # -- weights: prefer the experiment's real state dict; dummy is fine for a
    #    layout probe (shapes/dtypes drive the advice, not values).
    hf_config = OptimizedDecoder.load_hf_config()  # or hand-build a config stub
    state_dict = OptimizedDecoder.load_layer_state_dict(LAYER_IDX)  # or dummy

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        max_batch_size=MAX_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        page_block_size=PAGE_BLOCK,
    )

    # -- decode inputs via the experiment's own prep helpers.
    hidden = OptimizedDecoder.prepare_decode_inputs(
        torch.randn(1, 1, MAX_BATCH, hf_config.hidden_size, dtype=torch.bfloat16),
        device,
    )
    cos, sin = OptimizedDecoder.prepare_decode_rope(device, MAX_SEQ_LEN)  # rot_mats
    page_table = OptimizedDecoder.prepare_page_table(
        OptimizedDecoder.build_contiguous_page_table(MAX_BATCH, MAX_SEQ_LEN, PAGE_BLOCK),
        device,
    )
    current_pos = ttnn.from_torch(
        torch.zeros(MAX_BATCH, dtype=torch.int32),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kwargs = dict(current_pos=current_pos, rot_mats=(cos, sin), page_table=page_table)
    return decoder, kwargs, hidden
# ========================== END experiment wiring ===========================


_DECODER = None
_KWARGS = None


def decode(hidden):
    # One decode step through the model's real ttnn path. The tracer records the
    # ttnn ops; the advisor's optimizer re-derives the L1 layout.
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
