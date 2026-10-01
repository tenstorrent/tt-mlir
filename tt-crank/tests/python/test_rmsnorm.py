# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from tt_crank.torch.testing import strict_no_fallback

_FUSED_GRAD_FN = "FusedRmsNormBackward0"
_DT = torch.bfloat16
_PCC = 0.999


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _check(got: torch.Tensor | None, ref: torch.Tensor | None, name: str) -> None:
    if ref is None:
        assert got is None, f"unexpected {name}"
        return
    assert got is not None, f"no {name}"
    assert got.shape == ref.shape, f"{name} shape"
    assert _pcc(got.cpu(), ref) >= _PCC, f"{name} mismatch"
    atol = max(0.05, 0.01 * ref.abs().max().item())
    assert torch.allclose(
        got.cpu().float(), ref.float(), atol=atol, rtol=0.05
    ), f"{name} magnitude"


# input shape, has weight, (input requires grad, weight requires grad), eps
_FUSED_CASES = {
    "base": ((2, 32, 64), True, (True, True), 1e-6),
    "no-weight": ((2, 32, 64), False, (True, False), 1e-6),
    "4d": ((1, 2, 32, 64), True, (True, True), 1e-6),
    "2d": ((32, 64), True, (True, True), 1e-6),
    "eps-default": ((2, 32, 64), True, (True, True), None),
    "frozen-weight": ((2, 32, 64), True, (True, False), 1e-6),
    "weight-grad-only": ((2, 32, 64), True, (False, True), 1e-6),
    "c-48": ((2, 32, 48), True, (True, True), 1e-6),
    "eps-large": ((2, 32, 64), True, (True, True), 1.0),
    "eps-default-small-input": ((2, 32, 64), True, (True, True), None),
    "1d": ((64,), True, (True, True), 1e-6),
    "s17-c40": ((3, 17, 40), True, (True, True), 1e-6),
}
_INPUT_SCALE = {"eps-default-small-input": 3e-4}


@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_rmsnorm_eager_fused(case: str) -> None:
    shape, has_weight, (x_grad, w_grad), eps = _FUSED_CASES[case]
    x = torch.randn(*shape, dtype=_DT)
    if case in _INPUT_SCALE:
        x = (x.float() * _INPUT_SCALE[case]).to(_DT)
    w = torch.randn(shape[-1], dtype=_DT) if has_weight else None
    ref_x = x.clone().requires_grad_(x_grad)
    ref_w = w.clone().requires_grad_(w_grad) if has_weight else None
    ref_out = F.rms_norm(ref_x, (shape[-1],), ref_w, eps)
    ref_out.sum().backward()

    tt_x = x.to("tt").requires_grad_(x_grad)
    tt_w = w.to("tt").requires_grad_(w_grad) if has_weight else None
    with strict_no_fallback():
        out = F.rms_norm(tt_x, (shape[-1],), tt_w, eps)
        assert out.grad_fn.name() == _FUSED_GRAD_FN, f"decomposed: {out.grad_fn.name()}"
        out.sum().backward()

    _check(out.detach(), ref_out.detach(), "output")
    _check(tt_x.grad, ref_x.grad, "grad_input")
    if has_weight:
        _check(tt_w.grad, ref_w.grad, "grad_weight")


# input dtype, input shape, normalized_shape, weight dtype (None = no weight),
# (input requires grad, weight requires grad), eps
_DECOMPOSE_CASES = {
    "fp32": (torch.float32, (2, 32, 64), (64,), torch.float32, (True, True), 1e-6),
    "two-dims": (_DT, (2, 32, 64), (32, 64), _DT, (True, True), 1e-6),
    "fp32-weight": (_DT, (2, 32, 64), (64,), torch.float32, (True, True), 1e-6),
    "fp32-no-weight": (torch.float32, (2, 32, 64), (64,), None, (True, False), 1e-6),
    "fp32-weight-grad-only": (
        torch.float32,
        (2, 32, 64),
        (64,),
        torch.float32,
        (False, True),
        1e-6,
    ),
    "fp32-eps-default": (
        torch.float32,
        (2, 32, 64),
        (64,),
        torch.float32,
        (True, True),
        None,
    ),
}


def _decompose_inputs(case: str):
    x_dtype, shape, normalized_shape, w_dtype, grads, eps = _DECOMPOSE_CASES[case]
    x = torch.randn(*shape, dtype=x_dtype)
    w = torch.randn(*normalized_shape, dtype=w_dtype) if w_dtype is not None else None
    return x, w, normalized_shape, grads, eps


def _leaf(t: torch.Tensor | None, requires_grad: bool, device: str | None = None):
    if t is None:
        return None
    t = t.to(device) if device else t.clone()
    return t.requires_grad_(requires_grad)


@pytest.mark.parametrize("case", list(_DECOMPOSE_CASES), ids=list(_DECOMPOSE_CASES))
def test_rmsnorm_eager_decomposes(case: str) -> None:
    x, w, normalized_shape, (x_grad, w_grad), eps = _decompose_inputs(case)
    ref_x, ref_w = _leaf(x, x_grad), _leaf(w, w_grad)
    ref_out = F.rms_norm(ref_x, normalized_shape, ref_w, eps)
    ref_out.sum().backward()

    tt_x, tt_w = _leaf(x, x_grad, "tt"), _leaf(w, w_grad, "tt")
    out = F.rms_norm(tt_x, normalized_shape, tt_w, eps)
    assert out.grad_fn.name() != _FUSED_GRAD_FN, "expected crank's decomposition"
    out.sum().backward()

    _check(out.detach(), ref_out.detach(), "output")
    _check(tt_x.grad, ref_x.grad, "grad_input")
    if w is not None:
        _check(tt_w.grad, ref_w.grad, "grad_weight")


class _OpLog(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_rmsnorm_no_grad() -> None:
    """Proves the aten-level route only; eager cannot observe whether the kernel built the composite."""
    x = torch.randn(2, 32, 64, dtype=_DT)
    w = torch.randn(64, dtype=_DT)
    ref = F.rms_norm(x, (64,), w, 1e-6)

    with strict_no_fallback(), torch.no_grad(), _OpLog() as log:
        out = F.rms_norm(x.to("tt"), (64,), w.to("tt"), 1e-6)

    assert "aten._fused_rms_norm.default" in log.ops, f"not fused: {log.ops}"
    assert "aten.pow.Tensor_Scalar" not in log.ops, f"decomposed: {log.ops}"
    _check(out, ref, "output")


@pytest.mark.parametrize("case", list(_DECOMPOSE_CASES), ids=list(_DECOMPOSE_CASES))
def test_rmsnorm_inference_mode_decomposes(case: str) -> None:
    x, w, normalized_shape, _, eps = _decompose_inputs(case)
    ref = F.rms_norm(x, normalized_shape, w, eps)

    with torch.inference_mode():
        out = F.rms_norm(x.to("tt"), normalized_shape, _leaf(w, False, "tt"), eps)

    _check(out, ref, "output")


@pytest.mark.parametrize("case", list(_DECOMPOSE_CASES), ids=list(_DECOMPOSE_CASES))
def test_rmsnorm_direct_fused_op_decomposes(case: str) -> None:
    x, w, normalized_shape, (x_grad, w_grad), eps = _decompose_inputs(case)
    ref_x, ref_w = _leaf(x, x_grad), _leaf(w, w_grad)
    ref_out, _ = torch.ops.aten._fused_rms_norm(
        ref_x, list(normalized_shape), ref_w, eps
    )
    ref_out.sum().backward()

    tt_x, tt_w = _leaf(x, x_grad, "tt"), _leaf(w, w_grad, "tt")
    out, _ = torch.ops.aten._fused_rms_norm(tt_x, list(normalized_shape), tt_w, eps)
    assert (
        out.grad_fn.name() == _FUSED_GRAD_FN
    ), f"expected the fused node, got {out.grad_fn.name()}"
    out.sum().backward()

    _check(out.detach(), ref_out.detach(), "output")
    _check(tt_x.grad, ref_x.grad, "grad_input")
    if w is not None:
        _check(tt_w.grad, ref_w.grad, "grad_weight")


# input dtype, weight dtype
_UNSUPPORTED_DTYPE_CASES = {
    "input-fp16": (torch.float16, torch.float16),
    "weight-fp16": (_DT, torch.float16),
}


@pytest.mark.parametrize("inference", [False, True], ids=["grad", "inference-mode"])
@pytest.mark.parametrize(
    "case", list(_UNSUPPORTED_DTYPE_CASES), ids=list(_UNSUPPORTED_DTYPE_CASES)
)
def test_rmsnorm_unsupported_dtype_raises(case: str, inference: bool) -> None:
    x_dtype, w_dtype = _UNSUPPORTED_DTYPE_CASES[case]
    x = torch.randn(2, 32, 64, dtype=x_dtype).to("tt")
    w = torch.randn(64, dtype=w_dtype).to("tt")
    with pytest.raises(NotImplementedError, match="use bfloat16 or float32"):
        with torch.inference_mode() if inference else contextlib.nullcontext():
            F.rms_norm(x, (64,), w, 1e-6)


# input shape, normalized_shape, weight shape (None = no weight)
_INVALID_CASES = {
    "shape-mismatch-2d": ((2, 32, 64), (99, 64), None),
    "shape-mismatch-1d": ((2, 32, 64), (65,), None),
    "weight-shape": ((2, 32, 64), (64,), (1, 64)),
    "empty-shape": ((2, 32, 64), (), None),
    "input-rank": ((64,), (2, 64), None),
}


@pytest.mark.parametrize("inference", [False, True], ids=["grad", "inference-mode"])
@pytest.mark.parametrize("case", list(_INVALID_CASES), ids=list(_INVALID_CASES))
def test_rmsnorm_invalid_inputs(case: str, inference: bool) -> None:
    shape, normalized_shape, w_shape = _INVALID_CASES[case]
    x = torch.randn(*shape, dtype=_DT)
    w = torch.randn(*w_shape, dtype=_DT) if w_shape else None
    with pytest.raises(Exception) as cpu_err:
        F.rms_norm(x, normalized_shape, w, 1e-6)
    with pytest.raises(cpu_err.type):
        with torch.inference_mode() if inference else contextlib.nullcontext():
            F.rms_norm(
                x.to("tt"),
                normalized_shape,
                w.to("tt") if w is not None else None,
                1e-6,
            )


@pytest.mark.multichip
@pytest.mark.parametrize("has_weight", [True, False], ids=["weight", "no-weight"])
@pytest.mark.parametrize("dim", [0, 1], ids=["batch", "seq"])
def test_rmsnorm_multi_chip(tt_pg, dim: int, has_weight: bool) -> None:
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    n = torch.tt.num_chips()
    shape = [2, 32, 64]
    shape[dim] *= n
    x = torch.randn(*shape, dtype=_DT)
    w = torch.randn(64, dtype=_DT) if has_weight else None
    ref_x = x.clone().requires_grad_()
    ref_w = w.clone().requires_grad_() if has_weight else None
    ref_out = F.rms_norm(ref_x, (64,), ref_w, 1e-6)
    ref_out.sum().backward()

    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    tt_x = distribute_tensor(x.to("tt"), mesh, [Shard(dim)]).requires_grad_()
    tt_w = (
        distribute_tensor(w.to("tt"), mesh, [Replicate()]).requires_grad_()
        if has_weight
        else None
    )
    with strict_no_fallback():
        out = F.rms_norm(tt_x, (64,), tt_w, 1e-6)
        assert out.grad_fn.name() == _FUSED_GRAD_FN, f"decomposed: {out.grad_fn.name()}"
        out.sum().backward()

    assert out.placements == (Shard(dim),), f"output: {out.placements}"
    assert tt_x.grad.placements == (Shard(dim),), f"grad_input: {tt_x.grad.placements}"
    _check(out.detach().full_tensor(), ref_out.detach(), "output")
    _check(tt_x.grad.full_tensor(), ref_x.grad, "grad_input")
    if has_weight:
        _check(tt_w.grad.full_tensor(), ref_w.grad, "grad_weight")


# fp32 stat: only device sqrt/reciprocal error; one bf16 rounding would be ~3.9e-3.
_RSTD_RTOL = 1e-4


def _check_rstd(got: torch.Tensor, ref: torch.Tensor) -> None:
    assert got.dtype == torch.float32, f"rstd dtype {got.dtype}"
    assert (
        got.shape == ref.shape
    ), f"rstd shape {tuple(got.shape)} != {tuple(ref.shape)}"
    rel_err = ((got.cpu() - ref).abs() / ref.abs()).max().item()
    assert (
        rel_err <= _RSTD_RTOL
    ), f"rstd max relative error {rel_err:.3e} > {_RSTD_RTOL:.3e}"


@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_rmsnorm_rstd_precision(case: str) -> None:
    shape, has_weight, _, eps = _FUSED_CASES[case]
    x = torch.randn(*shape, dtype=_DT)
    if case in _INPUT_SCALE:
        x = (x.float() * _INPUT_SCALE[case]).to(_DT)
    w = torch.randn(shape[-1], dtype=_DT) if has_weight else None
    _, ref_rstd = torch.ops.aten._fused_rms_norm(x, [shape[-1]], w, eps)

    with strict_no_fallback():
        _, rstd = torch.ops.aten._fused_rms_norm(
            x.to("tt"), [shape[-1]], w.to("tt") if has_weight else None, eps
        )

    _check_rstd(rstd, ref_rstd)


def _stress_input(dist: str, shape: tuple[int, ...]) -> torch.Tensor:
    x = torch.randn(*shape)
    rows = x.view(-1, shape[-1])
    if dist == "large":
        x *= 1e3
    elif dist == "tiny":
        x *= 1e-4
    elif dist == "zero-rows":
        rows[::2] = 0
    elif dist == "outlier":
        rows[:, 3] = 300.0
    elif dist == "constant-magnitude":
        # random signs keep grad_weight from being constant
        rows.copy_(torch.rand(rows.shape[0], 1) * rows.sign())
    elif dist == "mixed-scales":
        rows *= torch.logspace(-4, 4, rows.shape[0]).unsqueeze(1)
    return x.to(_DT)


_STRESS_SHAPES = [(4, 32, 64), (2, 7, 100), (1, 128, 4096)]
_STRESS_DISTS = [
    "randn",
    "large",
    "tiny",
    "zero-rows",
    "outlier",
    "constant-magnitude",
    "mixed-scales",
]


@pytest.mark.parametrize("dist", _STRESS_DISTS)
@pytest.mark.parametrize("shape", _STRESS_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rmsnorm_stress(shape: tuple[int, ...], dist: str) -> None:
    x = _stress_input(dist, shape)
    w = torch.randn(shape[-1], dtype=_DT)
    ref_x, ref_w = x.clone().requires_grad_(), w.clone().requires_grad_()
    ref_out, ref_rstd = torch.ops.aten._fused_rms_norm(ref_x, [shape[-1]], ref_w, None)
    ref_out.sum().backward()

    tt_x, tt_w = x.to("tt").requires_grad_(), w.to("tt").requires_grad_()
    with strict_no_fallback():
        out, rstd = torch.ops.aten._fused_rms_norm(tt_x, [shape[-1]], tt_w, None)
        out.sum().backward()

    _check_rstd(rstd, ref_rstd)
    assert torch.allclose(
        out.detach().cpu().float(), ref_out.detach().float(), rtol=2.0**-7, atol=1e-6
    ), "output off by more than two bf16 steps"
    _check(tt_x.grad, ref_x.grad, "grad_input")
    _check(tt_w.grad, ref_w.grad, "grad_weight")
