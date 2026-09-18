# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tt_crank.torch import _artifacts
from tt_crank.torch._artifacts import collect_artifacts
from tt_crank.torch._compile import CompileOption
from tt_crank.torch.testing import post_aot_fx_hook, strict_no_fallback

# Present in the post-aot graph iff torch doesn't decompose the
# `torch.nn.functional.scaled_dot_product_attention` op - this is prevented by
# our dispatch registration of `_fused_sdp_choice_stub` which tells torch
# we have our own kernel for sdpa.
_SDPA_OVERRIDEABLE = "_scaled_dot_product_fused_attention_overrideable"
_SDPA_OVERRIDEABLE_BW = _SDPA_OVERRIDEABLE + "_backward"
# autograd node the fused op records in eager; MATH records the decomposition's nodes instead
_FUSED_GRAD_FN = "ScaledDotProductFusedAttentionOverrideableBackward0"

_DT = torch.bfloat16
_B, _H, _S, _E = 1, 8, 32, 64
_PCC = 0.99  # bf16 SDPA vs an fp-accumulating CPU reference


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run_sdpa(mode: str, fn, *tt_args):
    """
    Run ``fn(*tt_args)`` on tt in ``mode`` and return its output, asserting SDPA is
    handled natively. ``eager`` runs under ``strict_no_fallback`` so any op that would
    silently route through CPU raises; ``compile`` captures the post-aot graph and
    asserts SDPA stayed the atomic overrideable op (lowered to one ttir.sdpa, not
    decomposed). The output feeds the caller's numeric check against a CPU reference.
    """
    if mode == "eager":
        with strict_no_fallback():
            return fn(*tt_args)

    ops: set[str] = set()

    def record(gm):
        ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")

    with post_aot_fx_hook(record):
        out = torch.compile(fn, backend="tt", fullgraph=True)(*tt_args)
    torch._dynamo.reset()
    assert any(
        _SDPA_OVERRIDEABLE in op for op in ops
    ), f"SDPA did not stay atomic under compile; post-aot ops: {sorted(ops)}"
    return out


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("variant", ["plain", "causal", "masked"])
def test_sdpa_single_chip(variant: str, mode: str) -> None:
    """
    Single-chip SDPA across the plain, causal, and additive-mask variants. Under
    ``compile`` the op must stay the atomic overrideable op (not decomposed by
    torch.compile) so it lowers straight to ``ttir.scaled_dot_product_attention``;
    under ``eager`` it must run natively with no CPU fallback. Both modes compare the
    result against a CPU reference.
    """
    q, k, v = (torch.randn(_B, _H, _S, _E, dtype=_DT) for _ in range(3))
    is_causal = variant == "causal"
    mask = torch.randn(_B, 1, _S, _S, dtype=_DT) if variant == "masked" else None
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

    mt = mask.to("tt") if mask is not None else None

    def sdpa(a, b, c):
        return F.scaled_dot_product_attention(
            a, b, c, attn_mask=mt, is_causal=is_causal
        )

    got = _run_sdpa(mode, sdpa, q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    assert got.shape == ref.shape
    assert _pcc(got, ref) >= _PCC
    # PCC is invariant to scale/offset; allclose also catches magnitude errors.
    assert torch.allclose(got.float(), ref.float(), atol=0.05, rtol=0.05)


@pytest.mark.multichip
@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "masked"])
@pytest.mark.parametrize("parallel", ["tp", "dp"])
def test_sdpa_multi_chip(tt_pg, parallel: str, masked: bool, mode: str) -> None:
    """
    Multi-chip SDPA: the overrideable op carries a DTensor strategy, so head-parallel
    (TP, ``Shard(1)``) and batch-parallel (DP, ``Shard(0)``) Q/K/V flow straight to a
    sharded output with no gather. Both modes assert the output stays sharded on the same
    dim - a Replicate would mean an internal all-gather (which PCC-vs-CPU alone would not
    catch), and under compile atomicity alone would not rule it out. Compile additionally
    asserts the op lowered as the overrideable op. The masked variant exercises the
    strategy's mask handling: DP shards the 4-D mask's batch dim, TP replicates the
    head-broadcast mask.
    """
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    n = torch.tt.num_chips()
    dim = 1 if parallel == "tp" else 0
    if parallel == "tp" and _H % n:
        pytest.skip(f"needs num_heads ({_H}) divisible by chip count ({n})")
    batch = (
        _B if parallel == "tp" else n
    )  # DP shards batch, so it must be >= chip count
    q, k, v = (torch.randn(batch, _H, _S, _E, dtype=_DT) for _ in range(3))
    mask = torch.randn(batch, 1, _S, _S, dtype=_DT) if masked else None
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=(parallel,))
    dq, dk, dv = (distribute_tensor(t.to("tt"), mesh, [Shard(dim)]) for t in (q, k, v))

    tt_args = [dq, dk, dv]
    if mask is not None:
        # Place the mask so it needs no redistribution: shard the parallel dim when the
        # mask is full there (DP's batch), else replicate (TP's head-broadcast dim).
        placement = [Shard(dim)] if mask.shape[dim] != 1 else [Replicate()]
        tt_args.append(distribute_tensor(mask.to("tt"), mesh, placement))

    def sdpa(a, b, c, m=None):
        return F.scaled_dot_product_attention(a, b, c, attn_mask=m)

    out = _run_sdpa(mode, sdpa, *tt_args)
    # No gather (both modes): the output must stay sharded on the same dim.
    assert out.placements == (
        Shard(dim),
    ), f"expected Shard({dim}) output, got {out.placements}"
    got = out.full_tensor().cpu()
    assert got.shape == ref.shape
    assert _pcc(got, ref) >= _PCC


# tt-mlir resolves the sdpa_fw/sdpa_bw composites by optimization level: 0 inlines
# crank's plain-TTIR decomposition, 1 promotes to the ttml kernels. Eager compiles at 0.
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
# Eager MATH sdpa backward hits the unary f32/bf16 DataType mismatch that the debug runtime asserts on (#9344, was #7930).
_EAGER_DECOMPOSE = pytest.param(
    "eager",
    marks=pytest.mark.xfail(
        strict=False,
        reason="#9344: ttnn.isfinite returns f32 for bf16, debug runtime asserts (MATH backward)",
    ),
)


def _run_backward(mode: str, fn, *tt_args, strict: bool = True) -> set[str]:
    """Run `fn(*tt_args).backward()`; return the post-aot op names (empty under eager)."""
    ops: set[str] = set()
    if mode == "eager":
        with strict_no_fallback() if strict else contextlib.nullcontext():
            fn(*tt_args).backward()
        return ops

    def record(gm):
        ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")

    with post_aot_fx_hook(record):
        torch.compile(fn, backend="tt", fullgraph=True, options=_OPT[mode])(
            *tt_args
        ).backward()
    torch._dynamo.reset()
    return ops


def _check_grads(tts, refs, pcc: float = _PCC) -> None:
    for name, got, ref in zip("qkv", tts, refs):
        if ref.grad is None:
            assert got.grad is None, f"unexpected gradient for {name}"
            continue
        assert got.grad is not None, f"no gradient for {name}"
        assert got.grad.shape == ref.grad.shape
        assert _pcc(got.grad.cpu(), ref.grad) >= pcc, f"grad_{name} mismatch"


def _padded_causal_mask(seq: int, valid: int) -> torch.Tensor:
    """HF-style bool mask: causal, keys past `valid` are padding."""
    return torch.ones(seq, seq, dtype=torch.bool).tril() & (torch.arange(seq) < valid)


# (batch, heads_q, heads_kv, seq, head_dim), is_causal, scale, requires_grad per q/k/v, pcc, attn_mask factory
_G3 = (True, True, True)
_FUSED_CASES = {
    "base": ((1, 8, 8, 32, 64), True, None, _G3, _PCC, None),
    "noncausal": ((1, 8, 8, 32, 64), False, None, _G3, _PCC, None),
    "batched-longer-seq": ((2, 4, 4, 64, 64), True, None, _G3, _PCC, None),
    "gqa": ((1, 8, 2, 32, 64), True, None, _G3, _PCC, None),
    "gqa-noncausal": ((1, 8, 2, 32, 64), False, None, _G3, _PCC, None),
    "custom-scale": ((1, 8, 8, 32, 64), True, 0.5, _G3, _PCC, None),
    # head_dim not tile-aligned: Q/K are zero-padded to 64/96 for the kernel, which scales by 1/sqrt(padded D);
    # tight PCC catches a missing fold or a missing dQ/dK slice.
    "head-dim-40": ((1, 8, 8, 32, 40), True, None, _G3, 0.999, None),
    "head-dim-80-noncausal": ((1, 8, 8, 32, 80), False, None, _G3, 0.999, None),
    "grad-q-only": ((1, 8, 8, 32, 64), True, None, (True, False, False), _PCC, None),
    "grad-kv-only": ((1, 8, 8, 32, 64), True, None, (False, True, True), _PCC, None),
    # one S x S bool mask shared by all batches/heads rides the ttml `arbitrary` mask (what HF Llama passes at batch 1)
    "bool-mask-4d": (
        (1, 8, 8, 32, 64),
        False,
        None,
        _G3,
        _PCC,
        lambda: _padded_causal_mask(32, 24)[None, None],
    ),
    "bool-mask-2d-gqa": (
        (1, 8, 2, 64, 64),
        False,
        None,
        _G3,
        _PCC,
        lambda: _padded_causal_mask(64, 40),
    ),
    "bool-mask-batched": (
        (2, 4, 4, 32, 64),
        False,
        None,
        _G3,
        _PCC,
        lambda: _padded_causal_mask(32, 24)[None, None],
    ),
    # left padding: rows with no allowed key must give zeros like torch, not NaN
    "bool-mask-left-pad": (
        (1, 8, 8, 32, 64),
        False,
        None,
        _G3,
        _PCC,
        lambda: _padded_causal_mask(32, 32) & (torch.arange(32) >= 8),
    ),
}


@pytest.mark.parametrize("mode", _OPT_MODES)
@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_sdpa_backward_fused(case: str, mode: str) -> None:
    """Training within ttml's reach stays on the fused ttml pair and matches CPU."""
    (
        (batch, heads_q, heads_kv, seq, head_dim),
        is_causal,
        scale,
        needs_grad,
        pcc,
        make_mask,
    ) = _FUSED_CASES[case]
    kw = dict(is_causal=is_causal, scale=scale, enable_gqa=heads_kv != heads_q)
    if make_mask is not None:
        kw["attn_mask"] = make_mask()
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=_DT)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=_DT)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=_DT)
    refs = [t.clone().requires_grad_(g) for t, g in zip((q, k, v), needs_grad)]
    ref_out = F.scaled_dot_product_attention(*refs, **kw)
    ref_out.sum().backward()

    tts = [t.to("tt").requires_grad_(g) for t, g in zip((q, k, v), needs_grad)]
    tt_kw = {n: a.to("tt") if isinstance(a, torch.Tensor) else a for n, a in kw.items()}
    outs: list[torch.Tensor] = []

    def sdpa(a, b, c):
        outs.append(F.scaled_dot_product_attention(a, b, c, **tt_kw))
        return outs[-1].sum()

    ops = _run_backward(mode, sdpa, *tts)
    if mode == "eager":
        assert (
            outs[-1].grad_fn.name() == _FUSED_GRAD_FN
        ), f"training decomposed: {outs[-1].grad_fn.name()}"
    else:
        assert any(
            _SDPA_OVERRIDEABLE_BW in op for op in ops
        ), f"training decomposed; post-aot ops: {sorted(ops)}"
    assert _pcc(outs[-1].detach().cpu(), ref_out) >= pcc, "forward output mismatch"
    _check_grads(tts, refs, pcc)


# q/k/v shapes, sdpa kwargs (mask built lazily), modes
_EC = ("eager", "compile")
_DECOMPOSE_CASES = {
    "float-mask": (
        [(1, 8, 32, 64)] * 3,
        lambda: dict(attn_mask=torch.randn(1, 1, 32, 32, dtype=_DT)),
        _EC,
    ),
    # ttml takes one [1, 1, S, S] mask for all batches, so a per-batch padding mask decomposes.
    "bool-mask-per-batch": (
        [(2, 8, 32, 64)] * 3,
        lambda: dict(
            attn_mask=torch.stack([_padded_causal_mask(32, n) for n in (24, 16)])[
                :, None
            ]
        ),
        _EC,
    ),
    "seq-48": ([(1, 8, 48, 64)] * 3, lambda: dict(is_causal=True), _EC),
    "seq-48-noncausal": ([(1, 8, 48, 64)] * 3, lambda: dict(), _EC),
    "cross-seq": (
        [(1, 8, 32, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
        lambda: dict(),
        _EC,
    ),
    "3d": ([(8, 32, 64)] * 3, lambda: dict(is_causal=True), _EC),
    "3d-noncausal": ([(8, 32, 64)] * 3, lambda: dict(), _EC),
}


@pytest.mark.parametrize("mode", [_EAGER_DECOMPOSE, _COMPILE])
@pytest.mark.parametrize("case", list(_DECOMPOSE_CASES), ids=list(_DECOMPOSE_CASES))
def test_sdpa_backward_decomposes(case: str, mode: str) -> None:
    """Training outside ttml's reach falls back to the math decomposition, with correct grads."""
    shapes, make_kw, modes = _DECOMPOSE_CASES[case]
    if mode not in modes:
        pytest.skip(f"{case} not supported under {mode}")
    kw = make_kw()
    q, k, v = (torch.randn(*shape, dtype=_DT) for shape in shapes)
    refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(*refs, **kw).sum().backward()

    tts = [t.to("tt").requires_grad_(True) for t in (q, k, v)]
    tt_kw = {n: a.to("tt") if isinstance(a, torch.Tensor) else a for n, a in kw.items()}

    outs: list[torch.Tensor] = []

    def sdpa(a, b, c):
        outs.append(F.scaled_dot_product_attention(a, b, c, **tt_kw))
        return outs[-1].sum()

    ops = _run_backward(mode, sdpa, *tts, strict=False)
    if mode == "eager":
        assert (
            outs[-1].grad_fn.name() != _FUSED_GRAD_FN
        ), "expected the math decomposition"
    else:
        assert not any(
            _SDPA_OVERRIDEABLE in op for op in ops
        ), f"expected math decomposition; ops: {sorted(ops)}"
    _check_grads(tts, refs)


@pytest.mark.parametrize("mode", ["eager", _COMPILE])
def test_sdpa_dropout_not_implemented(mode: str) -> None:
    tts = [
        torch.randn(_B, _H, _S, _E, dtype=_DT).to("tt").requires_grad_(True)
        for _ in range(3)
    ]
    with pytest.raises(
        Exception, match="dropout is not supported"
    ):  # dynamo rewraps under compile
        _run_backward(
            mode,
            lambda a, b, c: F.scaled_dot_product_attention(
                a, b, c, dropout_p=0.1
            ).sum(),
            *tts,
        )


@_OPT1_ON_SIM
def test_sdpa_backward_f32_decomposes() -> None:
    """ttml's kernels are bf16-only; f32 training takes the math decomposition."""
    q, k, v = (torch.randn(_B, _H, _S, _E) for _ in range(3))
    refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(*refs, is_causal=True).sum().backward()
    tts = [t.to("tt").requires_grad_(True) for t in (q, k, v)]
    ops = _run_backward(
        "compile",
        lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=True).sum(),
        *tts,
    )
    assert not any(_SDPA_OVERRIDEABLE in op for op in ops), sorted(ops)
    _check_grads(tts, refs)


@pytest.mark.multichip
@pytest.mark.parametrize("mode", ["eager", _COMPILE])
@pytest.mark.parametrize("parallel", ["tp", "dp"])
def test_sdpa_multi_chip_causal_backward(tt_pg, parallel: str, mode: str) -> None:
    """TP (Shard(1)) and DP (Shard(0)) grads come back sharded on the same dim, no gather."""
    from torch.distributed.tensor import Shard, distribute_tensor

    n = torch.tt.num_chips()
    dim = 1 if parallel == "tp" else 0
    if parallel == "tp" and _H % n:
        pytest.skip(f"needs num_heads ({_H}) divisible by chip count ({n})")
    batch = _B if parallel == "tp" else n
    q, k, v = (torch.randn(batch, _H, _S, _E, dtype=_DT) for _ in range(3))
    refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(*refs, is_causal=True).sum().backward()

    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=(parallel,))
    tts = [
        distribute_tensor(t.to("tt"), mesh, [Shard(dim)]).requires_grad_(True)
        for t in (q, k, v)
    ]

    def sdpa(a, b, c):
        return F.scaled_dot_product_attention(a, b, c, is_causal=True).sum()

    if mode == "eager":
        # Capturing the DTensor output out of a compiled function trips aot's subclass handling, so the
        # autograd-node check is eager-only; compile asserts the op in the post-aot graph instead.
        with strict_no_fallback():
            out = F.scaled_dot_product_attention(*tts, is_causal=True)
            assert (
                out.grad_fn.name() == _FUSED_GRAD_FN
            ), f"training decomposed: {out.grad_fn.name()}"
            out.sum().backward()
    else:
        ops = _run_backward(mode, sdpa, *tts)
        assert any(
            _SDPA_OVERRIDEABLE_BW in op for op in ops
        ), f"training decomposed; post-aot ops: {sorted(ops)}"
    for name, got, ref in zip("qkv", tts, refs):
        assert got.grad is not None, f"no gradient for {name}"
        assert got.grad.placements == (
            Shard(dim),
        ), f"grad_{name}: {got.grad.placements}"
        assert (
            _pcc(got.grad.full_tensor().cpu(), ref.grad) >= _PCC
        ), f"grad_{name} mismatch"


def _main_func(ttir: str) -> str:
    """`@main` of a TTIR dump, without the private decomposition functions that follow it."""
    return ttir.split("func.func private")[0]


@pytest.mark.parametrize(
    "opt", [pytest.param(1, marks=_OPT1_ON_SIM), 0], ids=["opt1", "opt0"]
)
@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_sdpa_fused_lowering_ir(
    case: str, opt: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fused case lowers to one sdpa_fw and one sdpa_bw composite with the expected mask handling; OPT 1
    promotes them to the ttml kernels, OPT 0 inlines the decomposition."""
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)
    (
        (batch, heads_q, heads_kv, seq, head_dim),
        is_causal,
        scale,
        needs_grad,
        _,
        make_mask,
    ) = _FUSED_CASES[case]
    kw = dict(is_causal=is_causal, scale=scale, enable_gqa=heads_kv != heads_q)
    if make_mask is not None:
        kw["attn_mask"] = make_mask().to("tt")
    q = (
        torch.randn(batch, heads_q, seq, head_dim, dtype=_DT)
        .to("tt")
        .requires_grad_(needs_grad[0])
    )
    k = (
        torch.randn(batch, heads_kv, seq, head_dim, dtype=_DT)
        .to("tt")
        .requires_grad_(needs_grad[1])
    )
    v = (
        torch.randn(batch, heads_kv, seq, head_dim, dtype=_DT)
        .to("tt")
        .requires_grad_(needs_grad[2])
    )

    torch._dynamo.reset()
    with collect_artifacts("sdpa_lowering"):
        torch.compile(
            lambda a, b, c: F.scaled_dot_product_attention(a, b, c, **kw).sum(),
            backend="tt",
            fullgraph=True,
            options={CompileOption.OPT_LEVEL: opt},
        )(q, k, v).backward()
    (out_dir,) = list(tmp_path.iterdir())
    fw_ttir = _main_func((out_dir / "graph_0_forward.ttir.mlir").read_text())
    bw_ttir = _main_func((out_dir / "graph_1_backward.ttir.mlir").read_text())
    fw_ttnn = (out_dir / "graph_0_forward.ttnn.mlir").read_text()
    bw_ttnn = (out_dir / "graph_1_backward.ttnn.mlir").read_text()

    # One composite each way, typed by how the mask reaches ttml (see sdpa_mask in ttir_module_builder.cpp).
    mask_type = "causal" if is_causal else "arbitrary"
    assert fw_ttir.count('composite_name = "sdpa_fw"') == 1, fw_ttir
    assert bw_ttir.count('composite_name = "sdpa_bw"') == 1, bw_ttir
    for ttir in (fw_ttir, bw_ttir):
        assert f"mask_type = #ttcore.attention_mask_type<{mask_type}>" in ttir, ttir
    if is_causal:
        # The kernel does causal itself: no mask tensor, nothing to rebuild or zero.
        assert (
            "ttir.ones" not in fw_ttir
            and "ttir.eq" not in fw_ttir
            and "ttir.max" not in fw_ttir
        ), fw_ttir
    elif make_mask is None:
        # Stand-in for ttml's broken `none`: an all-ones keep-mask, no bool rebuild.
        assert "ttir.ones" in fw_ttir and "ttir.eq" not in fw_ttir, fw_ttir
    else:
        # Bool mask: rebuilt from torch's 0/-inf float (eq), rows keeping no key detected (max) and zeroed (multiply).
        for op in ("ttir.eq", "ttir.max", "ttir.multiply"):
            assert op in fw_ttir and op in bw_ttir, (op, fw_ttir)

    assert "composite" not in fw_ttnn and "composite" not in bw_ttnn
    if opt == 1:
        assert (
            "ttnn.sdpa_fw" in fw_ttnn and "ttnn.sdpa_bw" in bw_ttnn
        ), "composites not promoted to the ttml kernels"
        assert "ttnn.softmax" not in fw_ttnn
    else:
        assert "ttnn.sdpa_fw" not in fw_ttnn and "ttnn.sdpa_bw" not in bw_ttnn
        assert "ttnn.softmax" in fw_ttnn, "decomposition not inlined"


@_OPT1_ON_SIM
def test_sdpa_gqa_peel_feeds_unexpanded_kv() -> None:
    """HF's repeat_kv expansion is peeled off: sdpa_fw and sdpa_bw see [B, Hkv, S, D] K/V and the backward
    returns dK/dV at Hkv heads directly, with no expand/sum chain left in either graph."""
    from tt_crank.torch import _compile

    heads_q, heads_kv = 8, 2
    q = torch.randn(1, heads_q, _S, _E, dtype=_DT).to("tt").requires_grad_(True)
    k, v = (
        torch.randn(1, heads_kv, _S, _E, dtype=_DT).to("tt").requires_grad_(True)
        for _ in range(2)
    )

    def sdpa(a, b, c):
        rep = heads_q // heads_kv
        b = b[:, :, None].expand(1, heads_kv, rep, _S, _E).reshape(1, heads_q, _S, _E)
        c = c[:, :, None].expand(1, heads_kv, rep, _S, _E).reshape(1, heads_q, _S, _E)
        return F.scaled_dot_product_attention(a, b, c, is_causal=True).sum()

    graphs: list = []
    with post_aot_fx_hook(graphs.append):
        torch.compile(sdpa, backend="tt", fullgraph=True, options=_OPT["compile"])(
            q, k, v
        ).backward()
    torch._dynamo.reset()
    assert len(graphs) == 2, [g.graph for g in graphs]
    seen = set()
    for gm in graphs:
        for node in gm.graph.nodes:
            if node.target is _compile._SDPA_FUSED_FW:
                kv_args, seen = node.args[1:3], seen | {"fw"}
            elif node.target is _compile._SDPA_FUSED_BW:
                kv_args, seen = node.args[2:4], seen | {"bw"}
            else:
                continue
            assert [a.meta["val"].shape[1] for a in kv_args] == [
                heads_kv,
                heads_kv,
            ], node.format_node()
        # No [B, Hkv, rep, S, D] intermediate survives: neither the expansion nor the dK/dV reduction.
        five_d = [
            n
            for n in gm.graph.nodes
            if len(getattr(n.meta.get("val"), "shape", ())) == 5
        ]
        assert not five_d, [n.format_node() for n in five_d]
    assert seen == {"fw", "bw"}
    assert k.grad.shape == (1, heads_kv, _S, _E) and v.grad.shape == (
        1,
        heads_kv,
        _S,
        _E,
    )


@_OPT1_ON_SIM
def test_sdpa_training_is_two_device_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One forward and one backward program, each promoted to its ttml kernel, no host round-trip."""
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)
    tts = [
        torch.randn(_B, _H, _S, _E, dtype=_DT).to("tt").requires_grad_(True)
        for _ in range(3)
    ]

    def sdpa(a, b, c):
        return F.scaled_dot_product_attention(a, b, c, is_causal=True).sum()

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    with collect_artifacts("sdpa_training") as collection:
        torch.compile(sdpa, backend="tt", fullgraph=True, options=_OPT["compile"])(
            *tts
        ).backward()

    assert not torch._dynamo.utils.counters["graph_break"]
    assert collection.compile_stats().num_graphs == 2
    (out_dir,) = list(tmp_path.iterdir())
    index = json.loads((out_dir / "artifacts.json").read_text())
    assert [g["graph"] for g in index["graphs"]] == [
        "graph_0_forward",
        "graph_1_backward",
    ]
    host_ops = ("ttnn.from_device", "ttnn.to_device", "ttnn.to_layout", "ttnn.cpu")
    for graph, name in (
        ("graph_0_forward", "sdpa_fw"),
        ("graph_1_backward", "sdpa_bw"),
    ):
        ttir = (out_dir / f"{graph}.ttir.mlir").read_text()
        assert ttir.count(f'composite_name = "{name}"') == 1, ttir
        ttnn = (out_dir / f"{graph}.ttnn.mlir").read_text()
        assert f"ttnn.{name}" in ttnn, ttnn
        assert not [
            op for op in host_ops if op in ttnn
        ], f"{graph} moves tensors through the host"


@_OPT1_ON_SIM
def test_sdpa_fw_logsumexp_matches_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the ttml lse tile contract (value in column 0 of a [B, H, S, 32] f32 tile) against torch."""
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)
    q, k, v = (torch.randn(_B, _H, _S, _E, dtype=_DT) for _ in range(3))
    logits = (q.float() @ k.float().transpose(2, 3)) * _E**-0.5
    ref_lse = logits.masked_fill(
        ~torch.ones(_S, _S, dtype=torch.bool).tril(), float("-inf")
    ).logsumexp(-1)

    def fw(a, b, c):
        return torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            a, b, c, None, 0.0, True
        )[:2]

    torch._dynamo.reset()
    with collect_artifacts("sdpa_fw_lse"):
        compiled = torch.compile(
            fw, backend="tt", fullgraph=True, options=_OPT["compile"]
        )
        _, lse = compiled(*(t.to("tt") for t in (q, k, v)))
    (out_dir,) = list(tmp_path.iterdir())
    assert "ttnn.sdpa_fw" in (out_dir / "graph_0_inference.ttnn.mlir").read_text()
    assert lse.dtype == torch.float32 and lse.shape == (_B, _H, _S)
    assert _pcc(lse.cpu(), ref_lse) >= 0.999, "ttml logsumexp tile layout changed"
