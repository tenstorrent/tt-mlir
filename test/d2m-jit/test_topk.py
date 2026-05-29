# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import d2m_jit as d2m


@d2m.kernel
def k_topk_values_1x2048(in_t, out_t):
    row = remote_load(in_t, [0, 0])
    values = topk_values(row, 32, True, False)
    remote_store(out_t, [0, 0], values)


def _input_layout():
    return d2m.Layout(
        shape=(1, 2048),
        dtype=d2m.float32,
        block_shape=[1, 64],
        grid_shape=[1, 1],
    )


def _output_layout():
    return d2m.Layout(
        shape=(1, 32),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )


def _host_input(case_name):
    values = torch.arange(2048, dtype=torch.float32)
    if case_name == "ascending":
        return values.reshape(1, 2048)
    if case_name == "descending":
        return values.flip(0).reshape(1, 2048)
    if case_name == "tile_strided":
        return values.reshape(64, 32).transpose(0, 1).reshape(1, 2048)
    if case_name == "permuted":
        generator = torch.Generator().manual_seed(0)
        return (
            torch.randperm(2048, generator=generator).to(torch.float32).reshape(1, 2048)
        )
    if case_name == "duplicate_plateaus":
        return torch.div(values, 16, rounding_mode="floor").reshape(1, 2048)
    if case_name == "signed_fractional":
        return ((values.remainder(257) - 128) * 0.5).reshape(1, 2048)
    raise AssertionError(f"unknown topk input case: {case_name}")


@pytest.mark.parametrize(
    "case_name",
    [
        "ascending",
        "descending",
        "tile_strided",
        "permuted",
        "duplicate_plateaus",
        "signed_fractional",
    ],
)
def test_topk_values_1x2048_sorted_results(case_name):
    input_layout = _input_layout()
    output_layout = _output_layout()
    host_input = _host_input(case_name)
    out_d = d2m.empty(output_layout)
    k_topk_values_1x2048(d2m.to_layout(host_input, input_layout), out_d, grid=(1, 1))

    actual = out_d.to_host()
    expected = torch.topk(host_input, k=32, dim=-1, largest=True, sorted=True).values
    assert torch.all(
        actual[:, :-1] >= actual[:, 1:]
    ), f"{case_name} produced unsorted topk values: {actual}"
    assert torch.allclose(
        actual, expected, atol=0.01
    ), f"{case_name} topk mismatch:\nactual={actual}\nexpected={expected}"
