# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tt_crank.torch.testing import ExecutionMode, assert_close_cpu_vs_tt

_MODES = [ExecutionMode.EAGER, ExecutionMode.COMPILE]
_MODE_IDS = [m.value for m in _MODES]


@pytest.mark.parametrize("m,n", [(32, 64), (64, 32), (32, 32)])
def test_t(m: int, n: int) -> None:
    a = torch.randn((m, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.t, a)


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 64, 32), (64, 128, 64), (32, 32, 32)],
)
def test_mm(m: int, k: int, n: int) -> None:
    a = torch.randn((m, k), dtype=torch.bfloat16)
    b = torch.randn((k, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.mm, a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 64, 32), (64, 128, 64), (32, 32, 32)],
)
@pytest.mark.parametrize(
    "beta,alpha",
    [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5), (0.0, 1.0)],
    ids=["default", "beta2", "alpha05", "beta0"],
)
def test_addmm(m: int, k: int, n: int, beta: float, alpha: float) -> None:
    bias = torch.randn((n,), dtype=torch.bfloat16)
    mat1 = torch.randn((m, k), dtype=torch.bfloat16)
    mat2 = torch.randn((k, n), dtype=torch.bfloat16)
    fn = lambda b, m1, m2: torch.addmm(b, m1, m2, beta=beta, alpha=alpha)
    # beta=2 scales the bias before the BF16 add, doubling the rounding budget;
    # atol reflects the observed worst-case for beta=2 at K=128.
    assert_close_cpu_vs_tt(fn, bias, mat1, mat2, atol=0.22, rtol=0.1)


def test_linear_backward_lowered_for_compile() -> None:
    from tt_crank.torch import _compile

    op = torch.ops.aten.linear_backward.default
    assert op in _compile._LOWERINGS
    assert op not in _compile._TT_DECOMPOSITIONS


@pytest.mark.parametrize(
    "in_shape,out_features",
    [((64, 32), 16), ((4, 64, 32), 16), ((2, 8, 32), 64)],
    ids=["2d", "3d", "3d_widen"],
)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
def test_linear(in_shape, out_features: int, bias: bool, mode: ExecutionMode) -> None:
    x = torch.randn(in_shape, dtype=torch.bfloat16)
    w = torch.randn((out_features, in_shape[-1]), dtype=torch.bfloat16)
    args = (
        (x, w, torch.randn((out_features,), dtype=torch.bfloat16)) if bias else (x, w)
    )
    assert_close_cpu_vs_tt(
        torch.nn.functional.linear, *args, atol=0.2, rtol=0.1, mode=mode
    )


def _linear_grads(dev: str, x_cpu, w_cpu, b_cpu, g_cpu, need_x, need_w, need_b):
    x = x_cpu.to(dev).detach().requires_grad_(need_x)
    w = w_cpu.to(dev).detach().requires_grad_(need_w)
    args = [x, w]
    if b_cpu is not None:
        args.append(b_cpu.to(dev).detach().requires_grad_(need_b))
    out = torch.nn.functional.linear(*args)
    wanted = tuple(t for t, need in zip(args, (need_x, need_w, need_b)) if need)
    return [g.cpu() for g in torch.autograd.grad(out, wanted, g_cpu.to(dev))]


@pytest.mark.parametrize(
    "need_x,need_w,need_b",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
    ],
    ids=["all", "x", "w", "b", "x_w"],
)
@pytest.mark.parametrize("in_shape", [(64, 32), (4, 64, 32)], ids=["2d", "3d"])
def test_linear_backward(in_shape, need_x: bool, need_w: bool, need_b: bool) -> None:
    x = torch.randn(in_shape, dtype=torch.bfloat16)
    w = torch.randn((16, in_shape[-1]), dtype=torch.bfloat16)
    b = torch.randn((16,), dtype=torch.bfloat16)
    g = torch.randn((*in_shape[:-1], 16), dtype=torch.bfloat16)

    tt = _linear_grads("tt", x, w, b, g, need_x, need_w, need_b)
    cpu = _linear_grads("cpu", x, w, b, g, need_x, need_w, need_b)
    assert len(tt) == sum((need_x, need_w, need_b))
    for got, ref in zip(tt, cpu):
        assert got.shape == ref.shape
        torch.testing.assert_close(got, ref, atol=0.2, rtol=0.2)


def test_linear_backward_no_bias() -> None:
    x = torch.randn((64, 32), dtype=torch.bfloat16)
    w = torch.randn((16, 32), dtype=torch.bfloat16)
    g = torch.randn((64, 16), dtype=torch.bfloat16)
    tt = _linear_grads("tt", x, w, None, g, True, True, False)
    cpu = _linear_grads("cpu", x, w, None, g, True, True, False)
    for got, ref in zip(tt, cpu):
        torch.testing.assert_close(got, ref, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
@pytest.mark.parametrize(
    "fn,a_shape,b_shape",
    [
        (lambda a, b: a @ b.t(), (64, 32), (16, 32)),
        (lambda a, b: a.t() @ b, (32, 64), (32, 16)),
        (lambda a, b: a.t() @ b.t(), (32, 64), (16, 32)),
        (lambda a, b: a @ b.transpose(-1, -2), (4, 8, 32), (4, 16, 32)),
        (lambda a, b: a @ b.t(), (2, 64, 32), (16, 32)),
        (lambda a, b: a @ b.permute(1, 0, 2, 3), (2, 4, 8, 16), (4, 2, 16, 8)),
    ],
    ids=["b_t", "a_t", "both_t", "batched_t", "3d_by_2d_t", "leading_permute"],
)
def test_matmul_transposed_operands(fn, a_shape, b_shape, mode: ExecutionMode) -> None:
    a = torch.randn(a_shape, dtype=torch.bfloat16)
    b = torch.randn(b_shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(fn, a, b, atol=0.2, rtol=0.1, mode=mode)


@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
def test_matmul_shared_transpose(mode: ExecutionMode) -> None:
    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    fn = lambda x, y: (x @ y.t()) + y.t()
    assert_close_cpu_vs_tt(fn, a, b, atol=0.2, rtol=0.1, mode=mode)
