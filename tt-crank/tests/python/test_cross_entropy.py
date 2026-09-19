# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tt_crank.torch import _artifacts
from tt_crank.torch._artifacts import collect_artifacts
from tt_crank.torch._compile import CompileOption
from tt_crank.torch.testing import post_aot_fx_hook

# tt-mlir resolves the cross_entropy_fw/bw composites by optimization level: 0 inlines crank's plain-TTIR
# decomposition, 1 promotes to the ttml kernels.
_OPT = {
    "compile": {CompileOption.OPT_LEVEL: 1},
    "compile-opt0": {CompileOption.OPT_LEVEL: 0},
}
# Any OPT_LEVEL 1 compile aborts the process under ttsim, so those cases cannot even run there.
_SIM = os.environ.get("TT_CRANK_USE_SIMULATOR") == "1"
_OPT1_ON_SIM = pytest.mark.xfail(
    _SIM, reason="OPT_LEVEL 1 aborts under ttsim", run=False
)
_MODES = [pytest.param("compile", marks=_OPT1_ON_SIM), "compile-opt0"]
_IGNORE = -100
_FUSED = ("tt_crank.cross_entropy_fw", "tt_crank.cross_entropy_bw")
_ATEN = ("aten.nll_loss_forward", "aten.nll_loss_backward", "aten._log_softmax")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _inputs(
    rows: int, classes: int, ignore: bool, dtype=torch.bfloat16, scale: float = 3.0
):
    logits = torch.randn(rows, classes, dtype=dtype) * scale
    target = torch.randint(classes, (rows,))
    if ignore:
        target[::3] = _IGNORE
    return logits, target


def _run(mode: str, reduction: str, logits, target, grad=None):
    """Compiled loss + backward on tt; returns (loss, grad_logits, post-aot op names)."""

    def loss(x, t):
        return F.cross_entropy(x, t, ignore_index=_IGNORE, reduction=reduction)

    x = logits.to("tt").requires_grad_(True)
    ops: set[str] = set()
    with post_aot_fx_hook(
        lambda gm: ops.update(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
    ):
        out = torch.compile(loss, backend="tt", fullgraph=True, options=_OPT[mode])(
            x, target.to("tt")
        )
        out.backward(grad.to("tt") if grad is not None else None)
    torch._dynamo.reset()
    return out.detach().cpu(), x.grad.cpu(), ops


def _reference(reduction: str, logits, target, grad=None):
    x = logits.detach().float().requires_grad_(True)
    out = F.cross_entropy(x, target, ignore_index=_IGNORE, reduction=reduction)
    out.backward(grad.float() if grad is not None else None)
    return out.detach(), x.grad


def _fused(ops: set[str]) -> bool:
    return all(any(f in op for op in ops) for f in _FUSED) and not any(
        a in op for op in ops for a in _ATEN
    )


# rows, classes, reduction, ignore some rows
_FUSED_CASES = {
    "mean-ignore": (64, 128, "mean", True),
    "mean": (64, 128, "mean", False),
    "sum-ignore": (64, 128, "sum", True),
    # off the tile grid on both axes
    "rows-40-classes-100": (40, 100, "mean", True),
    "wide": (32, 8192, "mean", True),
}


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_cross_entropy_fused(case: str, mode: str) -> None:
    """bf16 mean/sum cross entropy fuses onto the ttml pair and matches an f32 CPU reference."""
    rows, classes, reduction, ignore = _FUSED_CASES[case]
    logits, target = _inputs(rows, classes, ignore)
    ref_loss, ref_grad = _reference(reduction, logits, target)
    loss, grad, ops = _run(mode, reduction, logits, target)
    assert _fused(ops), sorted(ops)
    torch.testing.assert_close(loss.float(), ref_loss, atol=0.05, rtol=0.02)
    assert grad.shape == ref_grad.shape
    assert _pcc(grad, ref_grad) >= 0.99


_FALLBACK_CASES = {
    # the backward kernel takes one grad for all rows
    "none": (
        "none",
        torch.bfloat16,
        lambda rows: torch.randn(rows, dtype=torch.bfloat16),
    ),
    # the kernels are bf16 only
    "f32": ("mean", torch.float32, lambda rows: None),
}


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case", list(_FALLBACK_CASES), ids=list(_FALLBACK_CASES))
def test_cross_entropy_aten_fallback(case: str, mode: str) -> None:
    """Outside the kernels' reach the aten ops stay and lower on their own, with correct grads."""
    reduction, dtype, make_grad = _FALLBACK_CASES[case]
    logits, target = _inputs(64, 128, True, dtype)
    grad = make_grad(64)
    ref_loss, ref_grad = _reference(reduction, logits, target, grad)
    loss, got, ops = _run(mode, reduction, logits, target, grad)
    assert not [op for op in ops if any(f in op for f in _FUSED)], sorted(ops)
    assert any("aten.nll_loss_forward" in op for op in ops), sorted(ops)
    torch.testing.assert_close(loss.float(), ref_loss, atol=0.05, rtol=0.02)
    assert _pcc(got, ref_grad) >= 0.99


@_OPT1_ON_SIM
def test_cross_entropy_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One composite per graph, promoted to the ttml kernel at OPT 1."""
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)
    logits, target = _inputs(64, 128, True)
    ref_loss, ref_grad = _reference("mean", logits, target)
    torch._dynamo.reset()
    with collect_artifacts("cross_entropy"):
        loss, grad, _ = _run("compile", "mean", logits, target)
    torch.testing.assert_close(loss.float(), ref_loss, atol=0.05, rtol=0.02)
    assert _pcc(grad, ref_grad) >= 0.99

    (out_dir,) = list(tmp_path.iterdir())
    index = json.loads((out_dir / "artifacts.json").read_text())
    assert [g["graph"] for g in index["graphs"]] == [
        "graph_0_forward",
        "graph_1_backward",
    ]
    for graph, name in (
        ("graph_0_forward", "cross_entropy_fw"),
        ("graph_1_backward", "cross_entropy_bw"),
    ):
        ttir = (out_dir / f"{graph}.ttir.mlir").read_text()
        assert ttir.count(f'composite_name = "{name}"') == 1, ttir
        assert f"ttnn.{name}" in (out_dir / f"{graph}.ttnn.mlir").read_text()


# Edge cases, on the fused (bf16 mean) and the aten (f32) path alike.
@pytest.mark.parametrize("mode", _MODES)
def test_log_softmax_large_logits_stay_finite(mode: str) -> None:
    """`log(softmax(x))` underflows to -inf past a logit gap of ~87; the aten lowering shifts by the row max."""
    logits, target = _inputs(64, 128, False, torch.float32, scale=100.0)
    grad = torch.ones(64)
    ref_loss, _ = _reference("none", logits, target, grad)
    loss, _, _ = _run(mode, "none", logits, target, grad)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss.float(), ref_loss, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32], ids=["fused", "aten"]
)
def test_cross_entropy_mean_all_ignored_zero_grad(
    dtype: torch.dtype, mode: str
) -> None:
    """Every row ignored under mean: the 0/0 loss is undefined (NaN in torch, inf on device), the gradients
    must still be zero like torch's, not NaN."""
    logits, _ = _inputs(32, 64, False, dtype)
    target = torch.full((32,), _IGNORE)
    loss, got, ops = _run(mode, "mean", logits, target)
    assert _fused(ops) == (dtype is torch.bfloat16), sorted(ops)
    assert not torch.isfinite(loss).any()
    assert torch.isfinite(got).all() and (got == 0).all()


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32], ids=["fused", "aten"]
)
def test_cross_entropy_ignored_row_non_finite_logits(
    dtype: torch.dtype, mode: str
) -> None:
    """A non-finite logit in an ignored row must not reach the loss or the kept rows' gradients; torch never
    reads it. (The ignored rows' own gradients are NaN in torch too, from log_softmax backward.)"""
    logits, target = _inputs(32, 64, True, dtype)
    logits[0, :] = float("inf")
    logits[3, 5] = float("nan")
    assert target[0] == _IGNORE and target[3] == _IGNORE
    ref_loss, ref_grad = _reference("mean", logits, target)
    loss, got, ops = _run(mode, "mean", logits, target)
    assert _fused(ops) == (dtype is torch.bfloat16), sorted(ops)
    kept = target != _IGNORE
    assert torch.isfinite(loss).all() and torch.isfinite(got[kept]).all()
    torch.testing.assert_close(loss.float(), ref_loss, atol=0.05, rtol=0.02)
    assert _pcc(got[kept], ref_grad[kept]) >= 0.99


def test_nll_loss_rejects_rank_1() -> None:
    """Unbatched `[C]` logits reach `nll_loss_forward` with rank 1; the lowering says so instead of indexing."""
    with pytest.raises(Exception, match="expected \\[rows x C\\]"):
        torch.compile(lambda a, b: F.cross_entropy(a, b), backend="tt", fullgraph=True)(
            torch.randn(128).to("tt"), torch.tensor(3).to("tt")
        )
    torch._dynamo.reset()
