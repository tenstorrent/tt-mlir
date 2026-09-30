# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from tt_crank.torch.testing import strict_no_fallback

# autograd node `_fused_rms_norm` records; the composite fallback records the decomposition's nodes instead
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
    # C not tile-aligned: checks the padded last tile is handled
    "c-48": ((2, 32, 48), True, (True, True), 1e-6),
    # eps comparable to mean(x^2), so a dropped or wrong eps moves the output past tolerance
    "eps-large": ((2, 32, 64), True, (True, True), 1.0),
    "eps-default-small-input": ((2, 32, 64), True, (True, True), None),
}
_INPUT_SCALE = {"eps-default-small-input": 3e-4}


@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_rmsnorm_eager_fused(case: str) -> None:
    """bf16 rms_norm over the last dim takes the fused op pair natively and matches CPU."""
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


# input dtype, input shape, normalized_shape, weight dtype
_DECOMPOSE_CASES = {
    "fp32": (torch.float32, (2, 32, 64), (64,), torch.float32),
    "two-dims": (_DT, (2, 32, 64), (32, 64), _DT),
    "fp32-weight": (_DT, (2, 32, 64), (64,), torch.float32),
}


@pytest.mark.parametrize("case", list(_DECOMPOSE_CASES), ids=list(_DECOMPOSE_CASES))
def test_rmsnorm_eager_decomposes(case: str) -> None:
    """Calls outside ttml's reach take torch's composite decomposition, with correct grads."""
    x_dtype, shape, normalized_shape, w_dtype = _DECOMPOSE_CASES[case]
    x = torch.randn(*shape, dtype=x_dtype)
    w = torch.randn(*normalized_shape, dtype=w_dtype)
    ref_x, ref_w = x.clone().requires_grad_(), w.clone().requires_grad_()
    ref_out = F.rms_norm(ref_x, normalized_shape, ref_w, 1e-6)
    ref_out.sum().backward()

    tt_x, tt_w = x.to("tt").requires_grad_(), w.to("tt").requires_grad_()
    out = F.rms_norm(tt_x, normalized_shape, tt_w, 1e-6)
    assert out.grad_fn.name() != _FUSED_GRAD_FN, "expected the composite decomposition"
    out.sum().backward()

    _check(out.detach(), ref_out.detach(), "output")
    _check(tt_x.grad, ref_x.grad, "grad_input")
    _check(tt_w.grad, ref_w.grad, "grad_weight")


class _OpLog(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_rmsnorm_no_grad() -> None:
    """Under no_grad no autograd node exists to check, so the dispatched ops show the fused path ran."""
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
    """inference_mode skips the autograd override, so the forward kernel must decompose unsupported calls itself."""
    x_dtype, shape, normalized_shape, w_dtype = _DECOMPOSE_CASES[case]
    x = torch.randn(*shape, dtype=x_dtype)
    w = torch.randn(*normalized_shape, dtype=w_dtype)
    ref = F.rms_norm(x, normalized_shape, w, 1e-6)

    with torch.inference_mode():
        out = F.rms_norm(x.to("tt"), normalized_shape, w.to("tt"), 1e-6)

    _check(out, ref, "output")


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

    # No gather: output and input gradient stay sharded on the input's dim.
    assert out.placements == (Shard(dim),), f"output: {out.placements}"
    assert tt_x.grad.placements == (Shard(dim),), f"grad_input: {tt_x.grad.placements}"
    _check(out.detach().full_tensor(), ref_out.detach(), "output")
    _check(tt_x.grad.full_tensor(), ref_x.grad, "grad_input")
    if has_weight:
        _check(tt_w.grad.full_tensor(), ref_w.grad, "grad_weight")
