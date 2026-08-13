# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers.

Kept free of `ttmlir` / `runner` imports: `test/d2m-jit/sim/test_sim.py` imports
`assert_pcc` from here and must stay importable with no tt-metal build
(SIMULATOR_SPEC.md §2). `runner.py` re-exports both PCC helpers so there is one
implementation behind both import paths.
"""

import torch
import math


def compute_pcc(golden, actual) -> float:
    """Pearson correlation coefficient between two tensors.

    Casts to f32 first so reduced-precision inputs (bf16/fp16 device outputs)
    correlate at f32 precision rather than in the tile dtype.
    """
    combined = torch.stack([golden.flatten().float(), actual.flatten().float()])
    return torch.corrcoef(combined)[0, 1].item()


def assert_pcc(golden, actual, threshold=0.99):
    pcc = compute_pcc(golden, actual)
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold}\ngolden:\n{golden}\nactual:\n{actual}"


def arange_tile(*shape, tile_size=[32, 32], dtype=None):
    assert len(shape) >= 2
    assert shape[-2] % tile_size[-2] == 0
    assert shape[-1] % tile_size[-1] == 0
    tiled_shape = list(shape)
    tiled_shape[-2] //= tile_size[-2]
    tiled_shape[-1] //= tile_size[-1]
    tensor = torch.arange(math.prod(tiled_shape), dtype=dtype).reshape(tiled_shape)
    tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    tensor = tensor.repeat([1] * len(tiled_shape) + tile_size)
    return tensor.transpose(-2, -3).reshape(shape)
