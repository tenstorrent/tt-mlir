# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for scalar-lift and trig math ops."""

import pytest
import torch

from tt_crank.torch.testing import (
    assert_close_cpu_vs_tt,
    get_supported_dtypes,
    strict_no_fallback,
)


@pytest.mark.parametrize("dtype", get_supported_dtypes())
@pytest.mark.parametrize("exp", [2.0, 0.5, -1.0])
def test_pow_tensor_scalar(exp: float, dtype: torch.dtype) -> None:
    a = torch.rand((32, 64), dtype=dtype).add(0.1)  # keep positive for fractional exp
    assert_close_cpu_vs_tt(lambda x: x.pow(exp), a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dtype", get_supported_dtypes())
@pytest.mark.parametrize("base", [2.0, 0.5])
def test_pow_scalar(base: float, dtype: torch.dtype) -> None:
    # Positive base only, a negative base with a non-integer exponent is
    # mathematically undefined.
    a = torch.randn((32, 64), dtype=dtype)
    assert_close_cpu_vs_tt(lambda x: torch.pow(base, x), a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dtype", get_supported_dtypes())
@pytest.mark.parametrize("base", [-2.0, -0.5])
def test_pow_scalar_negative_base(base: float, dtype: torch.dtype) -> None:
    # A negative base is only defined for integer exponents, so the exponents come
    # from randint here rather than the randn of test_pow_scalar.
    a = torch.randint(-3, 4, (32, 64)).to(dtype)
    assert_close_cpu_vs_tt(lambda x: torch.pow(base, x), a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dtype", get_supported_dtypes())
@pytest.mark.parametrize("scalar", [2.0, -1.5, 0.5])
def test_add_scalar(scalar: float, dtype: torch.dtype) -> None:
    a = torch.randn((32, 64), dtype=dtype)
    assert_close_cpu_vs_tt(lambda x: x + scalar, a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dtype", get_supported_dtypes())
@pytest.mark.parametrize("scalar", [2.0, -0.5, 3.0])
def test_mul_scalar(scalar: float, dtype: torch.dtype) -> None:
    a = torch.randn((32, 64), dtype=dtype)
    assert_close_cpu_vs_tt(lambda x: x * scalar, a, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", get_supported_dtypes())
def test_div_tensor(dtype: torch.dtype) -> None:
    a = torch.randn((32, 64), dtype=dtype)
    b = torch.rand((32, 64), dtype=dtype).add(0.1)
    assert_close_cpu_vs_tt(lambda x, y: x / y, a, b, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("scalar", [2.0, 4.0, 0.5])
def test_div_scalar(scalar: float) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x / scalar, a, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_div_integer_true_division(dtype: torch.dtype) -> None:
    # int / int is true division: the result is the default float dtype
    # (float32), not integer division. Divisor is nonzero.
    a = torch.randint(-20, 20, (32, 64), dtype=dtype)
    b = torch.randint(1, 8, (32, 64), dtype=dtype)
    assert_close_cpu_vs_tt(torch.div, a, b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_cos(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.cos, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_sin(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.sin, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 32)])
def test_neg(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.neg, a)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_log(shape: tuple) -> None:
    # Strictly positive input: log is undefined at and below zero.
    a = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    assert_close_cpu_vs_tt(torch.log, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_exp(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.exp, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_log1p(shape: tuple) -> None:
    # log1p is defined for x > -1, and is the accurate form near zero — which is
    # where it earns its place over log(1 + x), so the range straddles it.
    a = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    assert_close_cpu_vs_tt(torch.log1p, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_sqrt(shape: tuple) -> None:
    # Non-negative input: sqrt is undefined below zero. The offset also keeps the
    # values off zero, where sqrt's derivative blows up and bf16 loses precision.
    a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
    assert_close_cpu_vs_tt(torch.sqrt, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_tanh(shape: tuple) -> None:
    # [-2, 2] covers tanh's whole interesting range: the near-linear region
    # around zero and the onset of saturation towards ±1.
    a = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2
    assert_close_cpu_vs_tt(torch.tanh, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_reciprocal(shape: tuple) -> None:
    # Reciprocal is mathematically undefined at zero, keep inputs in
    # [0.5, 1.5], safely away from the singularity.
    a = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    assert_close_cpu_vs_tt(torch.reciprocal, a, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "min,max",
    [(-0.5, 0.5), (0.0, None), (None, 1.0)],
    ids=["both", "min_only", "max_only"],
)
@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_clamp(shape: tuple, min: float | None, max: float | None) -> None:
    # Clamp only selects between the input and a bound — no arithmetic error to
    # tolerate, so the default (exact-ish) tolerances apply.
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.clamp(x, min=min, max=max), a)


@pytest.mark.parametrize("min", [-0.5, 0.0, 1.0])
@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_clamp_min(shape: tuple, min: float) -> None:
    # clamp_min is clamp with no upper bound: selection only, no arithmetic
    # error to tolerate.
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.clamp_min(x, min), a)


@pytest.mark.parametrize("max", [-0.5, 0.0, 1.0])
@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_clamp_max(shape: tuple, max: float) -> None:
    # clamp_max is clamp with no lower bound: selection only, no arithmetic
    # error to tolerate.
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.clamp_max(x, max), a)


@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_cumsum(shape: tuple, dim: int) -> None:
    # Scaled down so the running total stays in a range bf16 can still resolve
    # to the tolerance below: bf16 has 8 mantissa bits, and cumsum's error
    # accumulates along the scanned dim.
    a = torch.randn(shape, dtype=torch.bfloat16) * 0.1
    assert_close_cpu_vs_tt(lambda x: torch.cumsum(x, dim=dim), a, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_silu(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.nn.functional.silu, a, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_gelu(shape: tuple, approximate: str) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    # We emit the accurate gelu (approximate="none"); a "tanh" request is served
    # by that same accurate op. strict_no_fallback asserts gelu runs on the
    # native tt kernel rather than silently falling back to CPU.
    gelu = lambda x: torch.nn.functional.gelu(x, approximate=approximate)  # noqa: E731
    with strict_no_fallback():
        assert_close_cpu_vs_tt(gelu, a, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_softmax(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.softmax(x, dim=dim), a, atol=0.05, rtol=0.05)


def test_softmax_attention_shape() -> None:
    # Attention scores: [B, H, S, S]
    a = torch.randn((1, 8, 128, 128), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.softmax(x, dim=-1), a, atol=0.05, rtol=0.05)
