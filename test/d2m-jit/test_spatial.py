# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import d2m_jit as d2m


@d2m.kernel
def k_spatial_add(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], x + x)


@d2m.kernel
def k_spatial_mul(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], x * x)


def test_spatial_two_regions_two_outputs():
    t = torch.randn(32, 32, dtype=torch.float32)
    in_layout = d2m.Layout(
        shape=t.shape, dtype=d2m.float32, block_shape=[1, 1], mem_space="dram"
    )
    out_layout = d2m.Layout(shape=t.shape, dtype=d2m.float32, block_shape=[1, 1])

    inp = d2m.to_layout(t, in_layout)
    out_add = d2m.empty(out_layout)
    out_mul = d2m.empty(out_layout)

    d2m.spatial(
        inputs=[inp],
        outputs=[out_add, out_mul],
        grid_ranges=[((0, 0), (0, 0)), ((1, 0), (1, 0))],
        region_builders=[
            lambda: k_spatial_add(inp, out_add, grid=(1, 1)),
            lambda: k_spatial_mul(inp, out_mul, grid=(1, 1)),
        ],
    )

    add_result, mul_result = d2m.to_host(out_add, out_mul)
    add_diff = (t + t - add_result).abs().max().item()
    mul_diff = (t * t - mul_result).abs().max().item()
    assert add_diff < 0.01, f"spatial add max diff {add_diff}"
    assert mul_diff < 0.05, f"spatial mul max diff {mul_diff}"
