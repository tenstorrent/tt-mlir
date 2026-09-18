# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F

from tt_crank.torch.testing import post_aot_fx_hook, strict_no_fallback

# Present in the post-aot graph iff torch doesn't decompose the
# `torch.nn.functional.scaled_dot_product_attention` op - this is prevented by
# our dispatch registration of `_fused_sdp_choice_stub` which tells torch
# we have our own kernel for sdpa.
_SDPA_OVERRIDEABLE = "_scaled_dot_product_fused_attention_overrideable"
_SDPA_OVERRIDEABLE_BW = _SDPA_OVERRIDEABLE + "_backward"

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


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                strict=False,
                reason="#7930",
            ),
        ),
        "compile",
    ],
)
def test_sdpa_backward(mode: str) -> None:
    """
    Autograd through SDPA works: the fused overrideable op has no differentiable
    backward (and no meta kernel for AOTAutograd's joint trace), so with grad live the
    choice function must pick MATH and decompose. Asserts the atomic op is *absent*
    under compile - its presence is what breaks training - and that grads match CPU.
    """
    q, k, v = (torch.randn(_B, _H, _S, _E, dtype=_DT) for _ in range(3))
    refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(*refs).sum().backward()

    tts = [t.to("tt").requires_grad_(True) for t in (q, k, v)]

    def sdpa(a, b, c):
        return F.scaled_dot_product_attention(a, b, c).sum()

    ops: set[str] = set()

    def record(gm):
        ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")

    if mode == "eager":
        sdpa(*tts).backward()
    else:
        with post_aot_fx_hook(record):
            torch.compile(sdpa, backend="tt", fullgraph=True)(*tts).backward()
        torch._dynamo.reset()
        assert not any(
            _SDPA_OVERRIDEABLE in op for op in ops
        ), f"SDPA stayed atomic under autograd, which has no backward; post-aot ops: {sorted(ops)}"

    for got, ref in zip(tts, refs):
        assert got.grad is not None
        assert _pcc(got.grad.cpu(), ref.grad) >= _PCC


# Eager training: the choice stub picks the fused op when the ttml sdpa_fw/sdpa_bw kernels can run the
# call, and MATH otherwise. Compiled training keeps MATH until the compile lowering lands.
def _run_eager_backward(fn, *tt_args, strict: bool = True) -> None:
    with strict_no_fallback() if strict else contextlib.nullcontext():
        fn(*tt_args).backward()


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
    # ttml scales by 1/sqrt(head_dim padded to 32); tight PCC catches a missing fold.
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


@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_sdpa_eager_backward_fused(case: str) -> None:
    """Eager training within ttml's reach runs the fused ttml pair (no CPU fallback) and matches CPU."""
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

    _run_eager_backward(sdpa, *tts)
    assert _pcc(outs[-1].detach().cpu(), ref_out) >= pcc, "forward output mismatch"
    _check_grads(tts, refs, pcc)


# q/k/v shapes, sdpa kwargs (mask built lazily)
_DECOMPOSE_CASES = {
    "float-mask": (
        [(1, 8, 32, 64)] * 3,
        lambda: dict(attn_mask=torch.randn(1, 1, 32, 32, dtype=_DT)),
    ),
    # ttml takes one [1, 1, S, S] mask for all batches, so a per-batch padding mask decomposes.
    "bool-mask-per-batch": (
        [(2, 8, 32, 64)] * 3,
        lambda: dict(
            attn_mask=torch.stack([_padded_causal_mask(32, n) for n in (24, 16)])[
                :, None
            ]
        ),
    ),
    "seq-48": ([(1, 8, 48, 64)] * 3, lambda: dict(is_causal=True)),
    "seq-48-noncausal": ([(1, 8, 48, 64)] * 3, lambda: dict()),
    "cross-seq": ([(1, 8, 32, 64), (1, 8, 64, 64), (1, 8, 64, 64)], lambda: dict()),
    "3d": ([(8, 32, 64)] * 3, lambda: dict(is_causal=True)),
    "3d-noncausal": ([(8, 32, 64)] * 3, lambda: dict()),
}


# torch's MATH backward calls aten::isneginf, whose ttnn.isfinite lowering returns f32 for bf16; the debug
# runtime (CI) asserts on the dtype mismatch, release runtimes do not (#9344).
@pytest.mark.xfail(
    strict=False,
    reason="#9344: ttnn.isfinite returns f32 for bf16, debug runtime asserts",
)
@pytest.mark.parametrize("case", list(_DECOMPOSE_CASES), ids=list(_DECOMPOSE_CASES))
def test_sdpa_eager_backward_decomposes(case: str) -> None:
    """Eager training outside ttml's reach falls back to the math decomposition, with correct grads."""
    shapes, make_kw = _DECOMPOSE_CASES[case]
    kw = make_kw()
    q, k, v = (torch.randn(*shape, dtype=_DT) for shape in shapes)
    refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(*refs, **kw).sum().backward()

    tts = [t.to("tt").requires_grad_(True) for t in (q, k, v)]
    tt_kw = {n: a.to("tt") if isinstance(a, torch.Tensor) else a for n, a in kw.items()}
    _run_eager_backward(
        lambda a, b, c: F.scaled_dot_product_attention(a, b, c, **tt_kw).sum(),
        *tts,
        strict=False,
    )
    _check_grads(tts, refs)


def test_sdpa_eager_dropout_not_implemented() -> None:
    tts = [
        torch.randn(_B, _H, _S, _E, dtype=_DT).to("tt").requires_grad_(True)
        for _ in range(3)
    ]
    with pytest.raises(Exception, match="dropout is not supported"):
        _run_eager_backward(
            lambda a, b, c: F.scaled_dot_product_attention(
                a, b, c, dropout_p=0.1
            ).sum(),
            *tts,
        )


@pytest.mark.multichip
@pytest.mark.parametrize("parallel", ["tp", "dp"])
def test_sdpa_eager_multi_chip_causal_backward(tt_pg, parallel: str) -> None:
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
    _run_eager_backward(
        lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=True).sum(),
        *tts,
    )
    for name, got, ref in zip("qkv", tts, refs):
        assert got.grad is not None, f"no gradient for {name}"
        assert got.grad.placements == (
            Shard(dim),
        ), f"grad_{name}: {got.grad.placements}"
        assert (
            _pcc(got.grad.full_tensor().cpu(), ref.grad) >= _PCC
        ), f"grad_{name} mismatch"
