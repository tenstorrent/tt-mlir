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


# Data parallel: rows sharded over the mesh. The rewrite runs on DTensor inputs, so the custom ops carry their
# own sharding rules (_sharding): Shard(0) logits and targets give Shard(0) per-row losses, and the row sums in
# TTCrossEntropy come out Partial("sum"). For `mean` the loss is Partial / Partial, which DTensor resolves to
# the exact global sum / count; the aligned ignore pattern here keeps the case comparable with torch's own
# nll rule (Partial("avg"), exact only with equal kept rows per shard). The unequal case is tested below.
@pytest.mark.multichip
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_cross_entropy_multi_chip_dp(tt_pg, reduction: str, mode: str) -> None:
    """Shard(0) logits/targets: fused per shard, loss reduced across the mesh, grads keep Shard(0)."""
    from torch.distributed.tensor import Shard, distribute_tensor

    n = torch.tt.num_chips()
    logits, target = _inputs(32 * n, 128, False)
    target[::4] = _IGNORE  # 8 ignored rows in every 32-row shard
    ref_loss, ref_grad = _reference(reduction, logits, target)
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    x = distribute_tensor(logits.to("tt"), mesh, [Shard(0)]).requires_grad_(True)
    t = distribute_tensor(target.to("tt"), mesh, [Shard(0)])

    def loss(a, b):
        return F.cross_entropy(a, b, ignore_index=_IGNORE, reduction=reduction)

    ops: set[str] = set()
    torch._dynamo.reset()
    with post_aot_fx_hook(
        lambda gm: ops.update(
            str(nd.target) for nd in gm.graph.nodes if nd.op == "call_function"
        )
    ):
        out = torch.compile(loss, backend="tt", fullgraph=True, options=_OPT[mode])(
            x, t
        )
        out.backward()
    torch._dynamo.reset()
    assert _fused(ops), sorted(ops)
    # The Partial -> Replicate reduce of the loss happens in DTensor, outside the compiled graph.
    torch.testing.assert_close(
        out.full_tensor().cpu().float(), ref_loss, atol=0.05, rtol=0.02
    )
    assert x.grad.placements == (Shard(0),), x.grad.placements
    assert _pcc(x.grad.full_tensor().cpu(), ref_grad) >= 0.99


def test_nll_loss_rejects_rank_1() -> None:
    """Unbatched `[C]` logits reach `nll_loss_forward` with rank 1; the lowering says so instead of indexing."""
    with pytest.raises(Exception, match="expected \\[rows x C\\]"):
        torch.compile(lambda a, b: F.cross_entropy(a, b), backend="tt", fullgraph=True)(
            torch.randn(128).to("tt"), torch.tensor(3).to("tt")
        )
    torch._dynamo.reset()


# --- the dynamo rewrite: what it matches, what it leaves to aten, and the shapes it meets in practice ---

# Arithmetic between a 0-d tensor and a Python scalar (`loss * 0.9`) comes back as shape [1] on tt:
# _prepare_op_args lifts the scalar to a [1] constant and the broadcast wins. torch's label-smoothing and
# probability-target decompositions do exactly that on the scalar loss, so the tangent no longer binds.
# Not a cross-entropy problem; these flip to passing once that is fixed.
_SCALAR_0D_BUG = pytest.mark.xfail(
    strict=True, reason="0-d tensor * Python scalar returns shape [1] on tt"
)


def _trace(mode: str, fn, *args, backward: bool = True):
    """Compile `fn(*args)` on tt, run backward when asked; returns (out, post-aot op names as a list)."""
    ops: list[str] = []
    torch._dynamo.reset()
    with post_aot_fx_hook(
        lambda gm: ops.extend(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
    ):
        out = torch.compile(fn, backend="tt", fullgraph=True, options=_OPT[mode])(*args)
        if backward:
            out.backward()
    torch._dynamo.reset()
    return out, ops


def _tt(logits, target):
    return logits.to("tt").requires_grad_(True), target.to("tt")


_SPELLINGS = {
    "builtin": lambda a, b: torch._C._nn.cross_entropy_loss(
        a, b, None, 1, _IGNORE, 0.0
    ),
    "module": torch.nn.CrossEntropyLoss(ignore_index=_IGNORE),
    "no-kwargs": lambda a, b: F.cross_entropy(a, b),
}


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("spelling", list(_SPELLINGS), ids=list(_SPELLINGS))
def test_cross_entropy_spellings_fuse(spelling: str, mode: str) -> None:
    """The C builtin, the nn.Module (dynamo inlines it to F.cross_entropy) and the all-defaults call fuse too."""
    logits, target = _inputs(64, 128, True)
    ref_loss, ref_grad = _reference("mean", logits, target)
    x, t = _tt(logits, target)
    loss, ops = _trace(mode, _SPELLINGS[spelling], x, t)
    assert _fused(set(ops)), sorted(ops)
    torch.testing.assert_close(loss.cpu().float(), ref_loss, atol=0.05, rtol=0.02)
    assert _pcc(x.grad.cpu(), ref_grad) >= 0.99


@_SCALAR_0D_BUG
@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_label_smoothing_stays_aten(mode: str) -> None:
    """label_smoothing is outside the kernels; torch decomposes it onto log_softmax and the aten lowerings run."""
    logits, target = _inputs(64, 128, True)
    ref = logits.detach().float().requires_grad_(True)
    ref_loss = F.cross_entropy(ref, target, ignore_index=_IGNORE, label_smoothing=0.1)
    ref_loss.backward()
    x, t = _tt(logits, target)
    loss, ops = _trace(
        mode,
        lambda a, b: F.cross_entropy(a, b, ignore_index=_IGNORE, label_smoothing=0.1),
        x,
        t,
    )
    assert not [op for op in ops if any(f in op for f in _FUSED)], sorted(ops)
    torch.testing.assert_close(
        loss.cpu().float(), ref_loss.detach(), atol=0.05, rtol=0.02
    )
    assert _pcc(x.grad.cpu(), ref.grad) >= 0.99


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_class_weights_rejected(mode: str) -> None:
    """Class weights skip the rewrite and the aten nll lowering says it does not take them."""
    logits, target = _inputs(64, 128, True)
    x, t = _tt(logits, target)
    weight = torch.rand(128, dtype=torch.bfloat16).to("tt")
    with pytest.raises(Exception, match="class weights"):
        _trace(
            mode,
            lambda a, b, w: F.cross_entropy(a, b, weight=w, ignore_index=_IGNORE),
            x,
            t,
            weight,
        )


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_rank_3_rejected(mode: str) -> None:
    """[N, C, L] logits skip the rewrite (2-D only); torch decomposes N-D cross entropy onto `gather`, which
    has no lowering, so it fails with a clear NotImplementedError rather than wrong IR."""
    logits = (torch.randn(4, 16, 8, dtype=torch.bfloat16) * 3).to("tt")
    target = torch.randint(16, (4, 8)).to("tt")
    with pytest.raises(Exception, match="not implemented|expected \\[rows x C\\]"):
        _trace(mode, lambda a, b: F.cross_entropy(a, b), logits, target, backward=False)


@_SCALAR_0D_BUG
@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_probability_targets_stay_aten(mode: str) -> None:
    """Float targets are class probabilities, not indices: no nll at all, the rewrite must not touch them."""
    logits, _ = _inputs(64, 128, False)
    probs = torch.softmax(torch.randn(64, 128), dim=1)
    ref = logits.detach().float().requires_grad_(True)
    ref_loss = F.cross_entropy(ref, probs)
    ref_loss.backward()
    x = logits.to("tt").requires_grad_(True)
    loss, ops = _trace(
        mode, lambda a, b: F.cross_entropy(a, b), x, probs.bfloat16().to("tt")
    )
    assert not [op for op in ops if any(f in op for f in _FUSED)], sorted(ops)
    torch.testing.assert_close(
        loss.cpu().float(), ref_loss.detach(), atol=0.05, rtol=0.02
    )
    assert _pcc(x.grad.cpu(), ref.grad) >= 0.99


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_forward_only(mode: str) -> None:
    """No requires_grad: only the forward graph, with the fw kernel and no bw."""
    logits, target = _inputs(64, 128, True)
    ref_loss, _ = _reference("mean", logits, target)
    loss, ops = _trace(
        mode,
        lambda a, b: F.cross_entropy(a, b, ignore_index=_IGNORE),
        logits.to("tt"),
        target.to("tt"),
        backward=False,
    )
    assert any(_FUSED[0] in op for op in ops), sorted(ops)
    assert not any(_FUSED[1] in op for op in ops), sorted(ops)
    torch.testing.assert_close(loss.cpu().float(), ref_loss, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_two_losses_one_graph(mode: str) -> None:
    """Two calls in one graph each get their own kernel pair; the loss is also consumed twice."""
    la, ta = _inputs(64, 128, True)
    lb, tb = _inputs(32, 128, False)

    def total(a, ta, b, tb):
        l1 = F.cross_entropy(a, ta, ignore_index=_IGNORE)
        l2 = F.cross_entropy(b, tb, reduction="sum")
        return l1 + l1 + l2

    ra, rb = (l.detach().float().requires_grad_(True) for l in (la, lb))
    ref_loss = total(ra, ta, rb, tb)
    ref_loss.backward()
    xa, xta = _tt(la, ta)
    xb, xtb = _tt(lb, tb)
    loss, ops = _trace(mode, total, xa, xta, xb, xtb)
    assert sum(_FUSED[0] in op for op in ops) == 2, ops
    assert sum(_FUSED[1] in op for op in ops) == 2, ops
    torch.testing.assert_close(
        loss.cpu().float(), ref_loss.detach(), atol=0.1, rtol=0.02
    )
    assert _pcc(xa.grad.cpu(), ra.grad) >= 0.99
    assert _pcc(xb.grad.cpu(), rb.grad) >= 0.99


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_custom_ignore_index(mode: str) -> None:
    """An in-range ignore_index (a real class id) masks exactly those rows."""
    logits, target = _inputs(64, 128, False)
    ignore = 7
    target[::5] = ignore
    ref = logits.detach().float().requires_grad_(True)
    ref_loss = F.cross_entropy(ref, target, ignore_index=ignore)
    ref_loss.backward()
    x, t = _tt(logits, target)
    loss, ops = _trace(
        mode, lambda a, b: F.cross_entropy(a, b, ignore_index=ignore), x, t
    )
    assert _fused(set(ops)), sorted(ops)
    torch.testing.assert_close(
        loss.cpu().float(), ref_loss.detach(), atol=0.05, rtol=0.02
    )
    assert _pcc(x.grad.cpu(), ref.grad) >= 0.99
    assert (x.grad.cpu()[target == ignore] == 0).all()


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_single_row(mode: str) -> None:
    """One row: the mean is that row's loss and its grad is softmax - onehot."""
    logits, target = _inputs(1, 128, False)
    ref_loss, ref_grad = _reference("mean", logits, target)
    x, t = _tt(logits, target)
    loss, ops = _trace(mode, lambda a, b: F.cross_entropy(a, b), x, t)
    assert _fused(set(ops)), sorted(ops)
    torch.testing.assert_close(loss.cpu().float(), ref_loss, atol=0.05, rtol=0.02)
    assert _pcc(x.grad.cpu(), ref_grad) >= 0.99


@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_shifted_lm_logits(mode: str) -> None:
    """The LM shape: [B, S, V] logits sliced off the last position and flattened, labels shifted by one."""
    b, s, v = 2, 17, 256
    logits = torch.randn(b, s, v, dtype=torch.bfloat16) * 3
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
    x, y = _tt(logits, labels)
    loss, ops = _trace(mode, lm_loss, x, y)
    assert _fused(set(ops)), sorted(ops)
    torch.testing.assert_close(
        loss.cpu().float(), ref_loss.detach(), atol=0.05, rtol=0.02
    )
    assert x.grad.shape == ref.grad.shape
    assert _pcc(x.grad.cpu(), ref.grad) >= 0.99
    assert (x.grad.cpu()[:, -1, :] == 0).all()


@pytest.mark.multichip
@pytest.mark.parametrize("mode", _MODES)
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
    ref_loss, ref_grad = _reference("mean", logits, target)
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    x_place = [Replicate()] if placement == "replicate" else [Shard(0)]
    x = distribute_tensor(logits.to("tt"), mesh, x_place).requires_grad_(True)
    t = distribute_tensor(target.to("tt"), mesh, [Replicate()])
    out, ops = _trace(
        mode, lambda a, b: F.cross_entropy(a, b, ignore_index=_IGNORE), x, t
    )
    assert _fused(set(ops)), sorted(ops)
    torch.testing.assert_close(
        out.full_tensor().cpu().float(), ref_loss, atol=0.05, rtol=0.02
    )
    assert x.grad.placements == tuple(x_place), x.grad.placements
    assert _pcc(x.grad.full_tensor().cpu(), ref_grad) >= 0.99


@pytest.mark.multichip
@pytest.mark.parametrize("mode", _MODES)
def test_cross_entropy_multi_chip_mean_unequal_shards(tt_pg, mode: str) -> None:
    """Shards keeping different row counts: the fused mean is sum / count across the mesh, so it is the exact
    global mean. (torch's own nll_loss rule would give an average of per-shard means here.)"""
    from torch.distributed.tensor import Shard, distribute_tensor

    n = torch.tt.num_chips()
    logits, target = _inputs(32 * n, 128, False)
    target[:24] = _IGNORE  # 24 of the first shard's 32 rows ignored, none elsewhere
    ref_loss, ref_grad = _reference("mean", logits, target)
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    x = distribute_tensor(logits.to("tt"), mesh, [Shard(0)]).requires_grad_(True)
    t = distribute_tensor(target.to("tt"), mesh, [Shard(0)])
    out, ops = _trace(
        mode, lambda a, b: F.cross_entropy(a, b, ignore_index=_IGNORE), x, t
    )
    assert _fused(set(ops)), sorted(ops)
    torch.testing.assert_close(
        out.full_tensor().cpu().float(), ref_loss, atol=0.05, rtol=0.02
    )
    assert _pcc(x.grad.full_tensor().cpu(), ref_grad) >= 0.99
