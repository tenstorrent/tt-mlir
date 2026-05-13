# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import math


def assert_pcc(golden, actual, threshold=0.99):
    combined = torch.stack([golden.flatten(), actual.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
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
    tensor = tensor.repeat([1]*len(tiled_shape) + tile_size)
    return tensor.transpose(-2, -3).reshape(shape)
