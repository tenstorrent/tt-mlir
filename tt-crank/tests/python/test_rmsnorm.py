# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from tt_crank.torch import _artifacts
from tt_crank.torch._artifacts import collect_artifacts
from tt_crank.torch._compile import CompileOption
from tt_crank.torch.testing import post_aot_fx_hook, strict_no_fallback

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


_SIM = os.environ.get("TT_CRANK_USE_SIMULATOR") == "1"
_OPT1_ON_SIM = pytest.mark.xfail(
    _SIM, reason="OPT_LEVEL 1 aborts under ttsim", run=False
)
_OPT_LEVELS = [0, pytest.param(1, marks=_OPT1_ON_SIM)]

# input dtype, normalized_shape, weight dtype
_COMPILE_OUTSIDE_TTML_CASES = {
    "fp32": (torch.float32, (64,), torch.float32),
    "fp32-two-dims": (torch.float32, (32, 64), torch.float32),
    "bf16-two-dims": (_DT, (32, 64), _DT),
    "bf16-fp32-weight": (_DT, (64,), torch.float32),
}


@pytest.mark.parametrize("opt_level", _OPT_LEVELS)
@pytest.mark.parametrize(
    "case", list(_COMPILE_OUTSIDE_TTML_CASES), ids=list(_COMPILE_OUTSIDE_TTML_CASES)
)
def test_rmsnorm_compile_direct_fused_op_outside_ttml(
    case: str, opt_level: int
) -> None:
    x_dtype, normalized_shape, w_dtype = _COMPILE_OUTSIDE_TTML_CASES[case]
    x = torch.randn(2, 32, 64, dtype=x_dtype)
    w = torch.randn(*normalized_shape, dtype=w_dtype)
    ref = F.rms_norm(x, normalized_shape, w, 1e-6)

    ops: set[str] = set()

    def record(gm):
        ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")

    fn = torch.compile(
        lambda a, b: torch.ops.aten._fused_rms_norm(a, list(normalized_shape), b, 1e-6)[
            0
        ],
        backend="tt",
        fullgraph=True,
        options={CompileOption.OPT_LEVEL: opt_level},
    )
    with post_aot_fx_hook(record):
        out = fn(x.to("tt"), w.to("tt"))
    torch._dynamo.reset()

    assert any("_fused_rms_norm" in op for op in ops), sorted(ops)
    if x_dtype == torch.float32:
        # tt's fp32 math is accurate to about one bf16 rounding; a promoted bf16 ttml kernel measures ~1e-2.
        d = (out.cpu() - ref).abs()
        rel_err = (d / ref.abs().clamp_min(1e-3)).max().item()
        assert (
            rel_err <= 2.0**-8 * 1.1
        ), f"fp32 off by {rel_err:.3e}, more than one bf16 rounding (promoted to bf16 ttml?)"
    else:
        _check(out, ref, "output")


@pytest.fixture
def artifacts_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)


def _compile_run(fn, opt_level: int, *args, backward: bool):
    """Compile and run fn; return (result, post-aot op names, captured TTIR of every graph)."""
    ops: set[str] = set()

    def record(gm):
        ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")

    compiled = torch.compile(
        fn, backend="tt", fullgraph=True, options={CompileOption.OPT_LEVEL: opt_level}
    )
    with post_aot_fx_hook(record), collect_artifacts("rmsnorm") as collection:
        result = compiled(*args)
        if backward:
            result.backward()
    torch._dynamo.reset()
    return result, ops, [a.compile_result.ttir for a in collection.artifacts]


def _fused_inputs(case: str):
    shape, has_weight, _, _ = _FUSED_CASES[case]
    x = torch.randn(*shape, dtype=_DT)
    if case in _INPUT_SCALE:
        x = (x.float() * _INPUT_SCALE[case]).to(_DT)
    w = torch.randn(shape[-1], dtype=_DT) if has_weight else None
    return x, w


@pytest.mark.parametrize("opt_level", _OPT_LEVELS)
@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_rmsnorm_compile_fused(case: str, opt_level: int, artifacts_tmp) -> None:
    shape, _, (x_grad, w_grad), eps = _FUSED_CASES[case]
    x, w = _fused_inputs(case)
    ref_x, ref_w = _leaf(x, x_grad), _leaf(w, w_grad)
    ref_out = F.rms_norm(ref_x, (shape[-1],), ref_w, eps)
    ref_out.sum().backward()

    tt_x, tt_w = _leaf(x, x_grad, "tt"), _leaf(w, w_grad, "tt")
    outs: list[torch.Tensor] = []

    def fn(a, b):
        outs.append(F.rms_norm(a, (shape[-1],), b, eps))
        return outs[-1].sum()

    _, ops, ttirs = _compile_run(fn, opt_level, tt_x, tt_w, backward=True)

    assert "aten._fused_rms_norm.default" in ops, sorted(ops)
    assert "aten._fused_rms_norm_backward.default" in ops, sorted(ops)
    assert any("rmsnorm_fw" in t for t in ttirs), "no rmsnorm_fw composite"
    assert any("rmsnorm_bw" in t for t in ttirs), "no rmsnorm_bw composite"
    _check(outs[-1].detach(), ref_out.detach(), "output")
    _check(tt_x.grad, ref_x.grad, "grad_input")
    if w is not None:
        _check(tt_w.grad, ref_w.grad, "grad_weight")


# Compile keeps rms in bf16 so ttml can promote it. The inlined decomposition rounds once (2**-8); the ttml
# kernel measures up to 5.3e-3 when C is not a multiple of 32, so OPT 1 allows two roundings.
_RSTD_RTOL_COMPILE = {0: 2.0**-8 * 1.05, 1: 2.0**-7}


@pytest.mark.parametrize("opt_level", _OPT_LEVELS)
@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_rmsnorm_compile_rstd_precision(
    case: str, opt_level: int, artifacts_tmp
) -> None:
    shape, _, _, eps = _FUSED_CASES[case]
    x, w = _fused_inputs(case)
    _, ref_rstd = torch.ops.aten._fused_rms_norm(x, [shape[-1]], w, eps)

    def fn(a, b):
        return torch.ops.aten._fused_rms_norm(a, [shape[-1]], b, eps)

    (_, rstd), _, _ = _compile_run(
        fn, opt_level, x.to("tt"), _leaf(w, False, "tt"), backward=False
    )

    assert rstd.dtype == torch.float32, f"rstd dtype {rstd.dtype}"
    assert rstd.shape == ref_rstd.shape, f"rstd shape {tuple(rstd.shape)}"
    rel_err = ((rstd.cpu() - ref_rstd).abs() / ref_rstd.abs()).max().item()
    rtol = _RSTD_RTOL_COMPILE[opt_level]
    assert rel_err <= rtol, f"rstd max relative error {rel_err:.3e} > {rtol:.3e}"
