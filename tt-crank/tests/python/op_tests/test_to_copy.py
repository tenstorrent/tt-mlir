# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Conversion dispatch, ownership, and allocation coverage for aten::_to_copy."""

import gc

import pytest
import torch

from tt_crank.torch.testing import strict_no_fallback


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.float16, torch.int32]
)
@pytest.mark.parametrize("shape", [(), (0,), (32, 32)])
def test_cpu_to_tt_round_trip(
    dtype: torch.dtype, shape: tuple[int, ...], request: pytest.FixtureRequest
) -> None:
    source = torch.full(shape, 3, dtype=dtype)
    with strict_no_fallback():
        converted = source.to("tt")
        if shape == (0,) and dtype == torch.float16:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Empty float16 readback rejects null buffers",
                    raises=RuntimeError,
                    strict=True,
                )
            )
        actual = converted.cpu()
    assert converted.device.type == "tt"
    torch.testing.assert_close(actual, source, atol=0, rtol=0)


def test_cpu_to_tt_does_not_allocate_destination() -> None:
    source = torch.ones((32, 32), dtype=torch.bfloat16)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        converted = source.to("tt")
    operations = {event.key for event in prof.key_averages()}
    assert "aten::_to_copy" in operations
    assert not {"aten::empty", "aten::empty_strided"} & operations
    torch.testing.assert_close(converted.cpu(), source, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int32])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cpu_to_tt_with_dtype_conversion(
    dtype: torch.dtype, non_blocking: bool
) -> None:
    source = torch.arange(32, dtype=torch.float32)
    expected = source.to(dtype)
    with strict_no_fallback():
        converted = source.to("tt", dtype=dtype, non_blocking=non_blocking)
    del source
    gc.collect()
    torch.testing.assert_close(converted.cpu(), expected, atol=0, rtol=0)


def test_cpu_to_tt_borrow_keeps_source_alive_and_checks_version() -> None:
    source = torch.arange(32, dtype=torch.float32)
    expected = source.clone()
    converted = source.to("tt")
    del source
    gc.collect()
    torch.testing.assert_close(converted.cpu(), expected, atol=0, rtol=0)

    source = torch.ones(32, dtype=torch.float32)
    converted = source.to("tt")
    source.add_(1)
    with pytest.raises(RuntimeError, match="modified in-place"):
        converted.cpu()


@pytest.mark.parametrize("direct", [False, True])
def test_tt_to_tt_forced_copy_is_independent(direct: bool) -> None:
    source = torch.ones((32, 32), dtype=torch.bfloat16).to("tt")
    assert source.to("tt") is source
    with strict_no_fallback():
        copied = (
            torch.ops.aten._to_copy.default(source)
            if direct
            else source.to("tt", copy=True)
        )
        source.copy_(torch.full((32, 32), 2, dtype=torch.bfloat16))
    assert copied is not source
    torch.testing.assert_close(
        copied.cpu(), torch.ones((32, 32), dtype=torch.bfloat16), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(
            "cpu",
            marks=pytest.mark.xfail(
                reason="TT-to-CPU dtype conversion recurses into CPU-to-CPU",
                raises=RuntimeError,
                strict=True,
            ),
        ),
        "tt",
    ],
)
def test_tt_dtype_conversion(target: str) -> None:
    source = torch.arange(32, dtype=torch.float32)
    with strict_no_fallback():
        converted = source.to("tt").to(target, dtype=torch.bfloat16)
    torch.testing.assert_close(converted.cpu(), source.bfloat16(), atol=0, rtol=0)


def test_cpu_tt_cpu_autograd() -> None:
    source = torch.randn((32, 32), requires_grad=True)
    gradient = torch.randn_like(source)
    with strict_no_fallback():
        converted = source.to("tt").cpu()
        converted.backward(gradient)
    torch.testing.assert_close(source.grad, gradient, atol=0, rtol=0)


def test_to_copy_preserves_layout_validation() -> None:
    source = torch.ones((32, 32), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="different layout"):
        torch.ops.aten._to_copy.default(source, device="tt", layout=torch.sparse_coo)

    source = torch.ones((2, 3, 4, 5), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="contiguous memory_format"):
        source.to("tt", memory_format=torch.channels_last)


@pytest.mark.multichip
def test_to_copy_preserves_distinct_shards(tt_pg) -> None:
    from torch.distributed.tensor import Shard, distribute_tensor

    chips = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((chips,))
    source = torch.arange(chips, dtype=torch.float32).repeat_interleave(32 * 32)
    source = source.reshape(chips * 32, 32).bfloat16()
    with strict_no_fallback():
        sharded = distribute_tensor(source.to("tt"), mesh, [Shard(0)])
        copied = sharded.to("tt", copy=True)
        sharded._local_tensor.zero_()
        actual = copied.full_tensor().cpu()
    torch.testing.assert_close(actual, source, atol=0, rtol=0)
