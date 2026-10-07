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

# Present in the post-aot graph iff the `AutogradPrivateUse1` kernel for `aten::cross_entropy_loss` took
# the call onto the ttml pair; otherwise torch decomposes it onto the aten ops.
_FUSED_FW = "tt_crank.cross_entropy_fw"
_FUSED_BW = "tt_crank.cross_entropy_bw"
_ATEN = ("aten.nll_loss_forward", "aten.nll_loss_backward", "aten._log_softmax")
# autograd node the fw custom op records in eager; the decomposition records NllLossBackward0 instead
_FUSED_GRAD_FN = "GeneratedBackwardFor_tt_crank_cross_entropy_fw_defaultBackward"

_DT = torch.bfloat16
_ROWS, _CLASSES = 64, 128
_IGNORE = -100
_PCC = 0.99  # bf16 kernels vs an f32 CPU reference
_TOL = dict(atol=0.05, rtol=0.02)  # loss vs the f32 CPU reference

# tt-mlir resolves the cross_entropy_fw/bw composites by optimization level: 0 inlines crank's plain-TTIR
# decomposition, 1 promotes to the ttml kernels. Eager runs the custom ops' reference bodies.
_OPT = {
    "eager": None,
    "compile": {CompileOption.OPT_LEVEL: 1},
    "compile-opt0": {CompileOption.OPT_LEVEL: 0},
}
# Any OPT_LEVEL 1 compile aborts the process under ttsim, so the promotion cases cannot even run there.
_SIM = os.environ.get("TT_CRANK_USE_SIMULATOR") == "1"
_OPT1_ON_SIM = pytest.mark.xfail(
    _SIM, reason="OPT_LEVEL 1 aborts under ttsim", run=False
)
_COMPILE = pytest.param("compile", marks=_OPT1_ON_SIM)
_OPT_MODES = [_COMPILE if m == "compile" else m for m in _OPT]
_COMPILE_MODES = [_COMPILE, "compile-opt0"]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _inputs(
    rows: int = _ROWS,
    classes: int = _CLASSES,
    ignore: bool = True,
    dtype: torch.dtype = _DT,
    scale: float = 3.0,
):
    logits = torch.randn(rows, classes, dtype=dtype) * scale
    target = torch.randint(classes, (rows,))
    if ignore:
        target[::3] = _IGNORE
    return logits, target


def _loss(**kw):
    return lambda a, b: F.cross_entropy(a, b, ignore_index=_IGNORE, **kw)


def _reference(logits, target, grad=None, **kw):
    """f32 CPU loss and grad_logits."""
    x = logits.detach().float().requires_grad_(True)
    t = target.float() if target.is_floating_point() else target
    out = F.cross_entropy(x, t, ignore_index=_IGNORE, **kw)
    out.backward(grad.float() if grad is not None else None)
    return out.detach(), x.grad


def _run(mode: str, fn, *tt_args, grad=None, backward: bool = True, options=None):
    """Run `fn(*tt_args)` on tt in `mode` and its backward; return the output and the post-aot op names
    (empty under eager). `options` add to the mode's compile options."""
    ops: list[str] = []
    if mode == "eager":
        out = fn(*tt_args)
        if backward:
            out.backward(grad)
        return out, ops

    def record(gm):
        ops.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")

    torch._dynamo.reset()
    with post_aot_fx_hook(record):
        out = torch.compile(
            fn, backend="tt", fullgraph=True, options={**_OPT[mode], **(options or {})}
        )(*tt_args)
        if backward:
            out.backward(grad)
    torch._dynamo.reset()
    return out, ops


def _grad_fn_names(out: torch.Tensor) -> set[str]:
    names: set[str] = set()
    todo = [out.grad_fn]
    while todo:
        fn = todo.pop()
        if fn is not None and fn.name() not in names:
            names.add(fn.name())
            todo.extend(next_fn for next_fn, _ in fn.next_functions)
    return names


def _assert_fused(mode: str, out: torch.Tensor, ops: list[str], fused: bool = True):
    """The ttml pair ran iff `fused`: by the autograd node in eager, by the post-aot ops under compile."""
    if mode == "eager":
        names = _grad_fn_names(out)
        assert (_FUSED_GRAD_FN in names) == fused, sorted(names)
        return
    on_kernels = all(any(f in op for op in ops) for f in (_FUSED_FW, _FUSED_BW))
    on_aten = any(a in op for op in ops for a in _ATEN)
    assert on_kernels == fused and on_aten != fused, sorted(ops)


# rows, classes, reduction, ignore some rows
_FUSED_CASES = {
    "mean-ignore": (64, 128, "mean", True),
    "mean": (64, 128, "mean", False),
    "sum-ignore": (64, 128, "sum", True),
    "none-ignore": (64, 128, "none", True),
    # off the tile grid on both axes
    "rows-40-classes-100": (40, 100, "mean", True),
    "wide": (32, 8192, "mean", True),
    # one row: the mean is that row's loss and its grad is softmax - onehot
    "single-row": (1, 128, "mean", False),
}


@pytest.mark.parametrize("mode", _OPT_MODES)
@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_cross_entropy_fused(case: str, mode: str) -> None:
    """bf16 cross entropy within the kernels' reach runs on the ttml pair and matches CPU."""
    rows, classes, reduction, ignore = _FUSED_CASES[case]
    logits, target = _inputs(rows, classes, ignore)
    grad = torch.randn(rows, dtype=_DT) if reduction == "none" else None
    ref_loss, ref_grad = _reference(logits, target, grad, reduction=reduction)
    x = logits.to("tt").requires_grad_(True)
    tt_grad = grad.to("tt") if grad is not None else None
    loss, ops = _run(mode, _loss(reduction=reduction), x, target.to("tt"), grad=tt_grad)
    _assert_fused(mode, loss, ops)
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss, **_TOL)
    assert x.grad.shape == ref_grad.shape
    assert _pcc(x.grad.cpu(), ref_grad) >= _PCC


# The two spellings dynamo records besides `F.cross_entropy`: the C builtin it calls, and the nn.Module
# (inlined to `F.cross_entropy`).
_SPELLINGS = {
    "builtin": lambda a, b: torch._C._nn.cross_entropy_loss(
        a, b, None, 1, _IGNORE, 0.0
    ),
    "module": torch.nn.CrossEntropyLoss(ignore_index=_IGNORE),
    "no-kwargs": lambda a, b: F.cross_entropy(a, b),
}


@pytest.mark.parametrize("mode", _OPT_MODES)
@pytest.mark.parametrize("spelling", list(_SPELLINGS), ids=list(_SPELLINGS))
def test_cross_entropy_spellings_fuse(spelling: str, mode: str) -> None:
    logits, target = _inputs()
    ref_loss, ref_grad = _reference(logits, target)
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _run(mode, _SPELLINGS[spelling], x, target.to("tt"))
    _assert_fused(mode, loss, ops)
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss, **_TOL)
    assert _pcc(x.grad.cpu(), ref_grad) >= _PCC


# Arithmetic between a 0-d tensor and a Python scalar (`loss * 0.9`) comes back as shape [1] on tt:
# _prepare_op_args lifts the scalar to a [1] constant and the broadcast wins. torch's label-smoothing and
# probability-target decompositions do exactly that on the scalar loss, so the tangent no longer binds.
# Not a cross-entropy problem; these flip to passing once that is fixed.
_SCALAR_0D_BUG = pytest.mark.xfail(
    strict=True, reason="0-d tensor * Python scalar returns shape [1] on tt"
)

# logits dtype, target factory, cross_entropy kwargs
_DECOMPOSE_CASES = {
    # the kernels are bf16 only
    "f32": (torch.float32, None, {}),
    # label smoothing and probability targets have no nll at all; the kernels must not see them
    "label-smoothing": (_DT, None, dict(label_smoothing=0.1)),
    "prob-targets": (
        _DT,
        lambda: torch.softmax(torch.randn(_ROWS, _CLASSES), 1).to(_DT),
        {},
    ),
}
_DECOMPOSE_PARAMS = [
    case if case == "f32" else pytest.param(case, marks=_SCALAR_0D_BUG)
    for case in _DECOMPOSE_CASES
]


@pytest.mark.parametrize("mode", _COMPILE_MODES)
@pytest.mark.parametrize("case", _DECOMPOSE_PARAMS)
def test_cross_entropy_decomposes(case: str, mode: str) -> None:
    """Outside the kernels' reach the aten ops stay and lower on their own, with correct grads."""
    dtype, make_target, kw = _DECOMPOSE_CASES[case]
    logits, target = _inputs(dtype=dtype)
    if make_target is not None:
        target = make_target()
    ref_loss, ref_grad = _reference(logits, target, **kw)
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _run(mode, _loss(**kw), x, target.to("tt"))
    _assert_fused(mode, loss, ops, fused=False)
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss, **_TOL)
    assert _pcc(x.grad.cpu(), ref_grad) >= _PCC


# logits, target, cross_entropy kwargs, error. All are left to aten, whose lowerings say what they lack.
_REJECTED_CASES = {
    "class-weights": (
        lambda: _inputs(),
        dict(weight=torch.rand(_CLASSES, dtype=_DT)),
        "class weights",
    ),
    # torch decomposes N-D cross entropy onto `gather`, which has no lowering
    "rank-3": (
        lambda: (torch.randn(4, 16, 8, dtype=_DT) * 3, torch.randint(16, (4, 8))),
        {},
        "not implemented|expected \\[rows x C\\]",
    ),
    # unbatched `[C]` logits reach `nll_loss_forward` with rank 1
    "rank-1": (
        lambda: (torch.randn(128), torch.tensor(3)),
        {},
        "expected \\[rows x C\\]",
    ),
}


@pytest.mark.parametrize("mode", _COMPILE_MODES)
@pytest.mark.parametrize("case", list(_REJECTED_CASES), ids=list(_REJECTED_CASES))
def test_cross_entropy_rejected(case: str, mode: str) -> None:
    make_inputs, kw, error = _REJECTED_CASES[case]
    logits, target = make_inputs()
    tt_kw = {n: a.to("tt") if isinstance(a, torch.Tensor) else a for n, a in kw.items()}
    with pytest.raises(Exception, match=error):  # dynamo rewraps under compile
        _run(
            mode,
            lambda a, b: F.cross_entropy(a, b, **tt_kw),
            logits.to("tt"),
            target.to("tt"),
            backward=False,
        )


@_OPT1_ON_SIM
def test_cross_entropy_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One composite per graph, promoted to the ttml kernel at OPT 1."""
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)
    logits, target = _inputs()
    ref_loss, ref_grad = _reference(logits, target)
    x = logits.to("tt").requires_grad_(True)
    with collect_artifacts("cross_entropy"):
        loss, _ = _run("compile", _loss(), x, target.to("tt"))
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss, **_TOL)
    assert _pcc(x.grad.cpu(), ref_grad) >= _PCC

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


@pytest.mark.parametrize("mode", _COMPILE_MODES)
def test_cross_entropy_forward_only(mode: str) -> None:
    """No requires_grad: only the forward graph, with the fw kernel and no bw."""
    logits, target = _inputs()
    ref_loss, _ = _reference(logits, target)
    loss, ops = _run(mode, _loss(), logits.to("tt"), target.to("tt"), backward=False)
    assert any(_FUSED_FW in op for op in ops), sorted(ops)
    assert not any(_FUSED_BW in op for op in ops), sorted(ops)
    torch.testing.assert_close(loss.cpu().float(), ref_loss, **_TOL)


@pytest.mark.parametrize("mode", _COMPILE_MODES)
def test_cross_entropy_two_losses_one_graph(mode: str) -> None:
    """Two calls in one graph each get their own kernel pair; the loss is also consumed twice."""
    la, ta = _inputs(64, 128, True)
    lb, tb = _inputs(32, 128, False)

    def total(a, ta, b, tb):
        l1 = F.cross_entropy(a, ta, ignore_index=_IGNORE)
        l2 = F.cross_entropy(b, tb, reduction="sum")
        return l1 + l1 + l2

    refs = [l.detach().float().requires_grad_(True) for l in (la, lb)]
    ref_loss = total(refs[0], ta, refs[1], tb)
    ref_loss.backward()
    tts = [l.to("tt").requires_grad_(True) for l in (la, lb)]
    loss, ops = _run(mode, total, tts[0], ta.to("tt"), tts[1], tb.to("tt"))
    assert sum(_FUSED_FW in op for op in ops) == 2, ops
    assert sum(_FUSED_BW in op for op in ops) == 2, ops
    torch.testing.assert_close(
        loss.detach().cpu().float(), ref_loss.detach(), atol=0.1, rtol=0.02
    )
    for got, ref in zip(tts, refs):
        assert _pcc(got.grad.cpu(), ref.grad) >= _PCC


@pytest.mark.parametrize("mode", _OPT_MODES)
def test_cross_entropy_custom_ignore_index(mode: str) -> None:
    """An in-range ignore_index (a real class id) masks exactly those rows."""
    logits, target = _inputs(ignore=False)
    ignore = 7
    target[::5] = ignore
    ref = logits.detach().float().requires_grad_(True)
    ref_loss = F.cross_entropy(ref, target, ignore_index=ignore)
    ref_loss.backward()
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _run(
        mode,
        lambda a, b: F.cross_entropy(a, b, ignore_index=ignore),
        x,
        target.to("tt"),
    )
    _assert_fused(mode, loss, ops)
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss.detach(), **_TOL)
    assert _pcc(x.grad.cpu(), ref.grad) >= _PCC
    assert (x.grad.cpu()[target == ignore] == 0).all()


# Compile only: eager `slice_backward` on tt comes back all zeros for any loss, not a cross-entropy matter.
@pytest.mark.parametrize("mode", _COMPILE_MODES)
def test_cross_entropy_shifted_lm_logits(mode: str) -> None:
    """The LM shape: [B, S, V] logits sliced off the last position and flattened, labels shifted by one."""
    b, s, v = 2, 17, 256
    logits = torch.randn(b, s, v, dtype=_DT) * 3
    labels = torch.randint(v, (b, s))
    labels[:, -3:] = _IGNORE  # padding at the tail

    def lm_loss(x, y):
        shifted = x[:, :-1, :]
        return F.cross_entropy(
            shifted.reshape(-1, shifted.shape[-1]),
            y[:, 1:].reshape(-1),
            ignore_index=_IGNORE,
        )

    ref = logits.detach().float().requires_grad_(True)
    ref_loss = lm_loss(ref, labels)
    ref_loss.backward()
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _run(mode, lm_loss, x, labels.to("tt"))
    _assert_fused(mode, loss, ops)
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss.detach(), **_TOL)
    assert x.grad.shape == ref.grad.shape
    assert _pcc(x.grad.cpu(), ref.grad) >= _PCC
    assert (x.grad.cpu()[:, -1, :] == 0).all()


# Edge cases, on the fused (bf16) and the aten (f32) path alike.
_DTYPES = pytest.mark.parametrize("dtype", [_DT, torch.float32], ids=["fused", "aten"])


@pytest.mark.parametrize("mode", _OPT_MODES)
def test_log_softmax_large_logits_stay_finite(mode: str) -> None:
    """`log(softmax(x))` underflows to -inf past a logit gap of ~87; the aten lowering shifts by the row max."""
    logits, target = _inputs(ignore=False, dtype=torch.float32, scale=100.0)
    grad = torch.ones(_ROWS)
    ref_loss, _ = _reference(logits, target, grad, reduction="none")
    loss, _ = _run(
        mode,
        _loss(reduction="none"),
        logits.to("tt").requires_grad_(True),
        target.to("tt"),
        grad=grad.to("tt"),
        # OPT 1 leaves fp32 accumulation to ttnn (off) since #9247; f32 exp/log at this scale need it.
        options={CompileOption.FP32_DEST_ACC_EN: True},
    )
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(
        loss.detach().cpu().float(), ref_loss, atol=1e-2, rtol=1e-3
    )


@pytest.mark.parametrize("mode", _OPT_MODES)
@_DTYPES
def test_cross_entropy_mean_all_ignored_zero_grad(
    dtype: torch.dtype, mode: str
) -> None:
    """Every row ignored under mean: the 0/0 loss is undefined (NaN in torch, inf on device), the gradients
    must still be zero like torch's, not NaN."""
    logits, _ = _inputs(32, 64, False, dtype)
    target = torch.full((32,), _IGNORE)
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _run(mode, _loss(), x, target.to("tt"))
    _assert_fused(mode, loss, ops, fused=dtype is _DT)
    assert not torch.isfinite(loss.detach()).any()
    got = x.grad.cpu()
    assert torch.isfinite(got).all() and (got == 0).all()


@pytest.mark.parametrize("mode", _OPT_MODES)
@_DTYPES
def test_cross_entropy_ignored_row_non_finite_logits(
    dtype: torch.dtype, mode: str
) -> None:
    """A non-finite logit in an ignored row must not reach the loss or the kept rows' gradients; torch never
    reads it. (The ignored rows' own gradients are NaN in torch too, from log_softmax backward.)"""
    logits, target = _inputs(32, 64, True, dtype)
    logits[0, :] = float("inf")
    logits[3, 5] = float("nan")
    assert target[0] == _IGNORE and target[3] == _IGNORE
    ref_loss, ref_grad = _reference(logits, target)
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _run(mode, _loss(), x, target.to("tt"))
    _assert_fused(mode, loss, ops, fused=dtype is _DT)
    kept = target != _IGNORE
    got = x.grad.cpu()
    assert torch.isfinite(loss.detach()).all() and torch.isfinite(got[kept]).all()
    torch.testing.assert_close(loss.detach().cpu().float(), ref_loss, **_TOL)
    assert _pcc(got[kept], ref_grad[kept]) >= _PCC


# Data parallel: rows sharded over the mesh. The custom ops carry their own sharding rules (_sharding):
# Shard(0) logits and targets give Shard(0) per-row losses, whose row sum comes out Partial("sum"). For
# `mean` the loss is Partial / Partial, which DTensor resolves to the exact global sum / count, where torch's
# own nll rule gives Partial("avg"), exact only with equal kept rows per shard. Compile only until #9345: in
# eager the custom ops' reference bodies hit the CPU fallback, which materializes one chip's shard of a mesh tensor.
@pytest.mark.multichip
@pytest.mark.parametrize("mode", _COMPILE_MODES)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("shards", ["equal", "unequal"])
def test_cross_entropy_multi_chip_dp(
    tt_pg, shards: str, reduction: str, mode: str
) -> None:
    """Shard(0) logits/targets: fused per shard, loss reduced across the mesh, grads keep Shard(0)."""
    from torch.distributed.tensor import Shard, distribute_tensor

    n = torch.tt.num_chips()
    logits, target = _inputs(32 * n, 128, False)
    if shards == "equal":
        target[::4] = _IGNORE  # 8 ignored rows in every 32-row shard
    else:
        target[:24] = _IGNORE  # 24 of the first shard's 32 rows ignored, none elsewhere
    ref_loss, ref_grad = _reference(logits, target, reduction=reduction)
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    x = distribute_tensor(logits.to("tt"), mesh, [Shard(0)]).requires_grad_(True)
    t = distribute_tensor(target.to("tt"), mesh, [Shard(0)])
    out, ops = _run(mode, _loss(reduction=reduction), x, t)
    _assert_fused(mode, out, ops)
    # The Partial -> Replicate reduce of the loss happens in DTensor, outside the compiled graph.
    torch.testing.assert_close(
        out.detach().full_tensor().cpu().float(), ref_loss, **_TOL
    )
    assert x.grad.placements == (Shard(0),), x.grad.placements
    assert _pcc(x.grad.full_tensor().cpu(), ref_grad) >= _PCC


@pytest.mark.multichip
@pytest.mark.parametrize("mode", _COMPILE_MODES)
@pytest.mark.parametrize(
    "placement",
    [
        "replicate",
        # dynamo's fake run of F.cross_entropy on Shard(0) logits + Replicate targets dies inside torch's
        # DTensor nll rule ("gather(): Expected dtype int32/int64 for index, but got torch.float32"),
        # before any backend code runs. Not ours to fix.
        pytest.param(
            "mixed",
            marks=pytest.mark.xfail(
                strict=True, reason="torch DTensor nll rule with Replicate targets"
            ),
        ),
    ],
)
def test_cross_entropy_multi_chip_other_placements(
    tt_pg, placement: str, mode: str
) -> None:
    """Replicate inputs fuse with a Replicate result; Shard(0) logits with Replicate targets make DTensor
    redistribute the targets (a slice) and the grads still come out Shard(0)."""
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    n = torch.tt.num_chips()
    logits, target = _inputs(32 * n, 128, True)
    ref_loss, ref_grad = _reference(logits, target)
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    x_place = [Replicate()] if placement == "replicate" else [Shard(0)]
    x = distribute_tensor(logits.to("tt"), mesh, x_place).requires_grad_(True)
    t = distribute_tensor(target.to("tt"), mesh, [Replicate()])
    out, ops = _run(mode, _loss(), x, t)
    _assert_fused(mode, out, ops)
    torch.testing.assert_close(
        out.detach().full_tensor().cpu().float(), ref_loss, **_TOL
    )
    assert x.grad.placements == tuple(x_place), x.grad.placements
    assert _pcc(x.grad.full_tensor().cpu(), ref_grad) >= _PCC
