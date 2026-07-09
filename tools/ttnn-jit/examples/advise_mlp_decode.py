# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""`ttnn-advise capture` target: agentic-research MLP-decode block.

Mirrors the ttnn op sequence + Llama-3.1-8B shapes of the agentic-research
optimized_decoder.py `_mlp_decode` (gate/up SiLU-gated + down projection), with
the MLP weights quantized to bfp4 as the expert decode does.

The advisor ignores program_config / memory_config / compute_kernel_config: the
tracer records only the ops + shapes + dtypes, and the greedy optimizer
re-derives the L1 layout -- so you can compare its recommendation against the
layout the model hand-wrote.

Run (needs a built tt-mlir with OpModel + ttnn_jit):

    source env/activate
    export LIBRARY_PATH="$(pwd)/.local/libnsl-shim:$LIBRARY_PATH"
    export PYTHONPATH="$TT_METAL_HOME/tools:$PYTHONPATH"
    export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"

    ttnn-advise capture tools/ttnn-jit/examples/advise_mlp_decode.py:mlp_decode \
        --out ./mlp_advice 2>/dev/null

Then read ./mlp_advice/report.json (ops, layouts, reshards, L1 spill).
"""
import torch
import ttnn

H, I = 4096, 14336  # hidden, intermediate


def mlp_decode(x, w_gate, w_up, w_down):
    gate = ttnn.linear(x, w_gate)
    up = ttnn.linear(x, w_up)
    hidden = ttnn.multiply(ttnn.silu(gate), up)
    return ttnn.linear(hidden, w_down)


def make_inputs(device):
    def w(shape, dtype):
        return ttnn.from_torch(
            torch.randn(*shape, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = w((1, 1, 1, H), ttnn.bfloat16)          # decode: a single token
    w_gate = w((1, 1, H, I), ttnn.bfloat4_b)    # bfp4 MLP weights (as expert decode)
    w_up = w((1, 1, H, I), ttnn.bfloat4_b)
    w_down = w((1, 1, I, H), ttnn.bfloat4_b)
    return x, w_gate, w_up, w_down
