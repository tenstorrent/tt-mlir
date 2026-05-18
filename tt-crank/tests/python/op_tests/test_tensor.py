"""Coverage for tensor-lifecycle aten kernels in src/torch/ops/tensor.cpp:
empty.memory_format / empty_strided (factories) and _copy_from (host↔device).

These tests don't go through any TTNN kernel — empty just allocates a host
runtime tensor, and _copy_from is a pure host-side memcpy on the cpu↔tt paths
— so they're safe to run on both real silicon and ttsim regardless of the
tile-alignment / dtype restrictions that gate the elementwise tests.
"""

import gc

import pytest
import torch


# -----------------------------------------------------------------------------
# empty.memory_format
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (32, 64, 32)])
def test_empty_shape_and_device(shape: tuple[int, ...]) -> None:
    t = torch.empty(shape, device="tt", dtype=torch.bfloat16)
    assert t.device.type == "tt"
    assert tuple(t.shape) == shape
    assert t.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32])
def test_empty_supported_dtypes(dtype: torch.dtype) -> None:
    t = torch.empty((32, 32), device="tt", dtype=dtype)
    assert t.dtype == dtype


def test_empty_default_dtype_is_float() -> None:
    # make_empty_tt_tensor defaults to c10::ScalarType::Float when dtype is unset.
    t = torch.empty((32, 32), device="tt")
    assert t.dtype == torch.float32


def test_empty_unsupported_dtype_rejected() -> None:
    # to_runtime_dtype maps only bf16/f32/i32; f64 hits its default TORCH_CHECK.
    with pytest.raises(RuntimeError, match="unsupported torch dtype"):
        torch.empty((32, 32), device="tt", dtype=torch.float64)


# -----------------------------------------------------------------------------
# empty_strided
# -----------------------------------------------------------------------------


def test_empty_strided_natural_strides_ok() -> None:
    t = torch.empty_strided((32, 32), stride=(32, 1), device="tt", dtype=torch.bfloat16)
    assert t.device.type == "tt"
    assert tuple(t.shape) == (32, 32)


def test_empty_strided_non_natural_rejected() -> None:
    # (1, 32) is the column-major stride for a (32, 32) tensor; our impl only
    # accepts the natural row-major stride.
    with pytest.raises(RuntimeError, match="non-contiguous strides"):
        torch.empty_strided((32, 32), stride=(1, 32), device="tt", dtype=torch.bfloat16)


# -----------------------------------------------------------------------------
# _copy_from
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32), (64, 128)])
def test_cpu_tt_cpu_round_trip_preserves_data(shape: tuple[int, ...]) -> None:
    # Exercises both _copy_from branches (cpu→tt and tt→cpu) without involving
    # any op kernel. The runtime tensor stays host-resident, so tolerance is zero.
    src = torch.randn(shape, dtype=torch.bfloat16)
    round_trip = src.to("tt").cpu()
    torch.testing.assert_close(round_trip, src, atol=0, rtol=0)


def test_cpu_to_tt_survives_source_deletion() -> None:
    # Probes the ownership contract of _copy_from(cpu→tt): the tt-side runtime
    # tensor is built via createOwnedHostTensor — "Owned" should mean it copies
    # the source bytes at upload time. If it secretly borrows the cpu pointer,
    # dropping `src` before the readback is a use-after-free and the round-trip
    # values will be garbage (or asan will fire).
    src = torch.randn((32, 32), dtype=torch.bfloat16)
    expected = src.clone()  # independent cpu copy that survives `del src`
    tt_tensor = src.to("tt")
    del src
    gc.collect()
    round_trip = tt_tensor.cpu()
    torch.testing.assert_close(round_trip, expected, atol=0, rtol=0)


def test_tt_to_tt_copy_not_implemented() -> None:
    a = torch.empty((32, 32), device="tt", dtype=torch.bfloat16)
    b = torch.empty((32, 32), device="tt", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="tt->tt"):
        a.copy_(b)
