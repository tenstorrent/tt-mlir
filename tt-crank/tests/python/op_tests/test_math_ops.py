"""Tests for scalar-lift and trig math ops: pow, add/mul/div scalars, cos, sin, neg, log, silu, softmax."""

import pytest
import torch

from tt_kurbla.torch.testing import (
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
