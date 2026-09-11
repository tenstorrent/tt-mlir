"""
Dtype-resolution tests for native tt kernels.

For every (op, dtype_pair) we check:
  1. The result's dtype matches PyTorch's promotion contract (compared
     against a CPU reference).
  2. Values are close to CPU (within the storage precision; tt stores
     unsupported wide dtypes as their narrower hardware alias so f64 cases
     are accurate only to f32).
  3. The op stays on the native path — wrapped in `strict_no_fallback`, so a
     regression that silently routes a promotion case through CPU fails the
     test instead of looking correct.

Adding a new binary op? Add it to `BINARY_OPS` below; the full dtype matrix
applies automatically. If a specific (op, dtype) combo isn't supported yet,
add it to that op's `xfail` set.

Adding a new dtype pair? Append to `DTYPE_PAIRS`. The promoted dtype is read
from `torch.result_type` on CPU — we don't bake expectations into the test.
"""

from __future__ import annotations

import pytest
import torch

from tt_kurbla.torch.testing import DeviceType, ExecutionMode, assert_close_cpu_vs_tt, strict_no_fallback


# Run every binary-op test in both modes so any eager-vs-compile dtype
# divergence shows up in the regular matrix.
_MODES = [ExecutionMode.EAGER, ExecutionMode.COMPILE]


# Tensor-tensor dtype combinations. Covers same-dtype, supported-but-mixed
# (bf16+f32), wide-narrow same-physical (f32+f64), and integer combos.
DTYPE_PAIRS: list[tuple[torch.dtype, torch.dtype]] = [
    (torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.bfloat16, torch.float64),
    (torch.int32, torch.int32),
    (torch.int64, torch.int64),
    (torch.int32, torch.int64),
    (torch.int64, torch.int32),
]


# Each entry: name → (op_callable, set of (a_dtype, b_dtype) combos that are
# expected to fail/skip for this op until support lands).
BINARY_OPS: dict[str, tuple] = {
    "add": (torch.add, set()),
}


def _make_tensor(dtype: torch.dtype, shape: tuple[int, ...] = (32,)) -> torch.Tensor:
    """Random tensor of the requested dtype. Float dtypes get `randn`-style
    values; int dtypes get small positives so promotion-overflow doesn't kick
    in (the f32-aliased i64 storage can clip past int32 max)."""
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype)
    return torch.randint(0, 64, shape, dtype=dtype)


def _tolerance(a_dtype: torch.dtype, b_dtype: torch.dtype) -> dict[str, float]:
    """Loosen the comparison when storage precision is coarser than the
    promoted dtype suggests (f64 → f32 alias, i64 → i32 alias)."""
    if torch.float64 in (a_dtype, b_dtype) or torch.bfloat16 in (a_dtype, b_dtype):
        return {"atol": 1e-2, "rtol": 1e-2}
    return {}


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("op_name", list(BINARY_OPS))
@pytest.mark.parametrize("a_dtype,b_dtype", DTYPE_PAIRS)
def test_binary_op_dtype(
    mode: ExecutionMode,
    op_name: str,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
) -> None:
    op, xfail_set = BINARY_OPS[op_name]
    if (a_dtype, b_dtype) in xfail_set:
        pytest.xfail(f"{op_name}({a_dtype}, {b_dtype}) not yet supported on tt")

    a = _make_tensor(a_dtype)
    b = _make_tensor(b_dtype)
    assert_close_cpu_vs_tt(op, a, b, mode=mode, **_tolerance(a_dtype, b_dtype))


# Tensor + Python scalar: result dtype follows the tensor (PyTorch's
# wrapped-scalar rule), not the scalar's natural Python type.
@pytest.mark.parametrize("mode", _MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("op_name", list(BINARY_OPS))
@pytest.mark.parametrize(
    "a_dtype,scalar",
    [
        (torch.bfloat16, 3.14),
        (torch.float32, 3.14),
        (torch.float32, 7),  # int scalar against float tensor
        (torch.int64, 3),
        (torch.int32, 3),
    ],
)
def test_binary_op_scalar(
    mode: ExecutionMode,
    op_name: str,
    a_dtype: torch.dtype,
    scalar,
) -> None:
    op, _ = BINARY_OPS[op_name]
    a = _make_tensor(a_dtype)
    assert_close_cpu_vs_tt(lambda x: op(x, scalar), a, mode=mode, **_tolerance(a_dtype, a_dtype))


# Cross-device: one tt operand, one CPU operand. Eager only - torch.compile
# rejects mixed-device inputs at FakeTensor trace time, so the compile path
# never sees this case.
@pytest.mark.parametrize("op_name", list(BINARY_OPS))
@pytest.mark.parametrize("a_dtype,b_dtype", [(torch.float32, torch.float32), (torch.float32, torch.float64)])
def test_binary_op_mixed_device(op_name: str, a_dtype: torch.dtype, b_dtype: torch.dtype) -> None:
    op, _ = BINARY_OPS[op_name]
    a = _make_tensor(a_dtype)
    b = _make_tensor(b_dtype)
    expected_dtype = op(a, b).dtype

    a_tt = a.to("tt")  # b stays on CPU
    with strict_no_fallback():
        out_tt = op(a_tt, b)
    assert out_tt.dtype == expected_dtype, (
        f"{op_name}({a_dtype}@tt, {b_dtype}@cpu): tt dtype {out_tt.dtype} != expected {expected_dtype}"
    )


# `tensor.to(other_dtype)` on tt: values survive the cross-dtype conversion
# and the output reports the target dtype. Eager only - this isn't an op
# torch.compile traces into a graph.
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.int64, torch.float32),
        (torch.int32, torch.int64),
    ],
)
def test_to_dtype_on_tt(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> None:
    a = _make_tensor(src_dtype)
    expected = a.to(dst_dtype)

    a_tt = a.to("tt")
    out_tt = a_tt.to(dst_dtype)

    assert out_tt.dtype == dst_dtype, f"to({dst_dtype}): tt dtype {out_tt.dtype} != expected {dst_dtype}"
    torch.testing.assert_close(out_tt.cpu(), expected, **_tolerance(src_dtype, dst_dtype))
