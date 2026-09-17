# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tt_crank.torch._compile import CompileOption
from tt_crank.torch.testing import post_aot_fx_hook

_IGNORE = -100
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


def _run(reduction: str, logits, target, grad=None):
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
        out = torch.compile(
            loss, backend="tt", fullgraph=True, options={CompileOption.OPT_LEVEL: 1}
        )(x, target.to("tt"))
        out.backward(grad.to("tt") if grad is not None else None)
    torch._dynamo.reset()
    return out.detach().cpu(), x.grad.cpu(), ops


def _reference(reduction: str, logits, target, grad=None):
    x = logits.detach().float().requires_grad_(True)
    out = F.cross_entropy(x, target, ignore_index=_IGNORE, reduction=reduction)
    out.backward(grad.float() if grad is not None else None)
    return out.detach(), x.grad


# reduction, dtype, ignore_index rows, upstream grad factory
_CASES = {
    "mean": ("mean", torch.bfloat16, True, lambda rows: None),
    "sum": ("sum", torch.bfloat16, True, lambda rows: None),
    "none": (
        "none",
        torch.bfloat16,
        True,
        lambda rows: torch.randn(rows, dtype=torch.bfloat16),
    ),
    "mean-no-ignore": ("mean", torch.bfloat16, False, lambda rows: None),
    "mean-f32": ("mean", torch.float32, True, lambda rows: None),
}


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_cross_entropy_aten(case: str) -> None:
    """`F.cross_entropy` lowers through the aten `_log_softmax` / `nll_loss_*` ops with correct grads."""
    reduction, dtype, ignore, make_grad = _CASES[case]
    logits, target = _inputs(64, 128, ignore, dtype)
    grad = make_grad(64)
    ref_loss, ref_grad = _reference(reduction, logits, target, grad)
    loss, got, ops = _run(reduction, logits, target, grad)
    assert any("aten.nll_loss_forward" in op for op in ops), sorted(ops)
    torch.testing.assert_close(loss.float(), ref_loss, atol=0.05, rtol=0.02)
    assert _pcc(got, ref_grad) >= 0.99


def test_log_softmax_large_logits_stay_finite() -> None:
    """`log(softmax(x))` underflows to -inf past a logit gap of ~87; the lowering shifts by the row max."""
    logits, target = _inputs(64, 128, False, torch.float32, scale=100.0)
    grad = torch.ones(64)
    ref_loss, _ = _reference("none", logits, target, grad)
    loss, _, _ = _run("none", logits, target, grad)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss.float(), ref_loss, atol=1e-2, rtol=1e-3)


def test_nll_loss_rejects_rank_1() -> None:
    """Unbatched `[C]` logits reach `nll_loss_forward` with rank 1; the lowering says so instead of indexing."""
    with pytest.raises(Exception, match="expected \\[rows x C\\]"):
        torch.compile(lambda a, b: F.cross_entropy(a, b), backend="tt", fullgraph=True)(
            torch.randn(128).to("tt"), torch.tensor(3).to("tt")
        )
    torch._dynamo.reset()
