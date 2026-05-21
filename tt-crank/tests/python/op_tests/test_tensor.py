"""Coverage for the tensor-lifecycle aten kernels in src/torch/ops/tensor.cpp.

The tests below also exercise `aten::empty.memory_format`, `aten::resize_`, and
`torch.reshape` — but those paths are handled by PyTorch's composite-implicit
dispatch (decomposing to `empty_strided` / storage allocator / `view`) before
they reach our backend; we don't register kernels for them.
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


def test_tt_to_tt_copy_preserves_data() -> None:
    src = torch.randn((32, 32), dtype=torch.bfloat16)
    a = src.to("tt")
    b = torch.empty((32, 32), device="tt", dtype=torch.bfloat16)
    b.copy_(a)
    torch.testing.assert_close(b.cpu(), src, atol=0, rtol=0)


# -----------------------------------------------------------------------------
# resize_ / _copy_from_and_resize
# -----------------------------------------------------------------------------


def test_resize_grow_then_copy() -> None:
    src = torch.randn((64, 32), dtype=torch.bfloat16)
    dst = torch.empty((0,), device="tt", dtype=torch.bfloat16)
    dst.resize_(src.shape)
    dst.copy_(src.to("tt"))
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


def test_resize_same_numel_no_realloc() -> None:
    # Same-numel resize keeps the existing storage; verify shape updates and
    # data survives (because TensorStorage isn't swapped).
    src = torch.arange(0, 64, dtype=torch.int32).reshape(8, 8)
    t = src.to("tt")
    t.resize_((4, 16))
    assert tuple(t.shape) == (4, 16)
    torch.testing.assert_close(t.cpu(), src.reshape(4, 16), atol=0, rtol=0)


# -----------------------------------------------------------------------------
# set_.source_Tensor
# -----------------------------------------------------------------------------


def test_set_source_tensor_shares_storage() -> None:
    src = torch.randn((32, 32), dtype=torch.bfloat16).to("tt")
    dst = torch.empty((4, 4), device="tt", dtype=torch.bfloat16)
    dst.set_(src)
    # Same data and same shape after the rebind.
    assert tuple(dst.shape) == tuple(src.shape)
    torch.testing.assert_close(dst.cpu(), src.cpu(), atol=0, rtol=0)
    # Same storage — mutating src's bytes should surface in dst.
    other = torch.randn((32, 32), dtype=torch.bfloat16)
    src.copy_(other.to("tt"))
    torch.testing.assert_close(dst.cpu(), other, atol=0, rtol=0)


# -----------------------------------------------------------------------------
# view / as_strided
#
# These ops MATERIALIZE on tt — they allocate a fresh contiguous copy rather
# than aliasing the source storage. The xfail at the bottom pins down that
# intentional break so a future switch to true aliasing semantics is a
# deliberate decision.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_shape,new_shape",
    [
        ((32, 32), (1024,)),
        ((64, 128), (128, 64)),
        ((32, 64, 32), (32, 2048)),
    ],
)
def test_view_roundtrip(src_shape: tuple[int, ...], new_shape: tuple[int, ...]) -> None:
    src = torch.randn(src_shape, dtype=torch.bfloat16)
    got = src.to("tt").view(new_shape).cpu()
    expected = src.view(new_shape)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_reshape_roundtrip() -> None:
    # torch.reshape decomposes via composite-implicit dispatch to aten::view
    # (we don't register a separate _reshape_alias kernel — it was probed to
    # be dead code).
    src = torch.randn((32, 32), dtype=torch.bfloat16)
    got = torch.reshape(src.to("tt"), (1024,)).cpu()
    torch.testing.assert_close(got, src.reshape(1024), atol=0, rtol=0)


def test_as_strided_correctness() -> None:
    src = torch.arange(0, 64, dtype=torch.int32).reshape(8, 8)
    # Take every other row: shape (4, 8), stride (16, 1).
    got = src.to("tt").as_strided((4, 8), (16, 1)).cpu()
    expected = src.as_strided((4, 8), (16, 1))
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_view_numel_mismatch_raises() -> None:
    src = torch.randn((32, 32), dtype=torch.bfloat16).to("tt")
    with pytest.raises(RuntimeError):
        src.view((33, 32))


@pytest.mark.xfail(reason="tt-backend view materializes — aliasing is intentionally not preserved", strict=True)
def test_view_aliasing_is_broken_by_design() -> None:
    # On CPU/CUDA: y = x.view(-1); y[0] = 5 mutates x[0, 0]. On tt, view
    # returns a fresh contiguous copy, so x is untouched. This xfail pins the
    # intentional semantic break so a future switch to true aliasing trips it.
    x = torch.zeros((32, 32), dtype=torch.bfloat16).to("tt")
    y = x.view(1024)
    y[0] = 5
    assert x.cpu()[0, 0].item() == 5.0


# -----------------------------------------------------------------------------
# _local_scalar_dense (backs Tensor.item())
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,value",
    [
        (torch.bfloat16, 1.5),
        (torch.float32, 3.25),
        (torch.int32, 42),
    ],
)
def test_item_roundtrip(dtype: torch.dtype, value: float | int) -> None:
    src = torch.tensor([value], dtype=dtype)
    got = src.to("tt").item()
    assert got == pytest.approx(value)


def test_item_on_zero_dim_tensor() -> None:
    # 0-d tensors are the canonical .item() case.
    src = torch.tensor(7, dtype=torch.int32)
    got = src.to("tt").item()
    assert got == 7
