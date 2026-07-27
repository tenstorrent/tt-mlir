from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tt_kurbla.torch.testing import post_aot_fx_hook, strict_no_fallback

# Present in the post-aot graph iff torch doesn't decompose the
# `torch.nn.functional.scaled_dot_product_attention` op - this is prevented by
# our dispatch registration of `_fused_sdp_choice_stub` which tells torch
# we have our own kernel for sdpa.
_SDPA_OVERRIDEABLE = "_scaled_dot_product_fused_attention_overrideable"

# SDPA's eager kernel allocates an f32 `logsumexp` slot, so it needs the f32 emitter.
pytestmark = pytest.mark.usefixtures("skip_if_sim")

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
    assert any(_SDPA_OVERRIDEABLE in op for op in ops), (
        f"SDPA did not stay atomic under compile; post-aot ops: {sorted(ops)}"
    )
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
        return F.scaled_dot_product_attention(a, b, c, attn_mask=mt, is_causal=is_causal)

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
    batch = _B if parallel == "tp" else n  # DP shards batch, so it must be >= chip count
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
    assert out.placements == (Shard(dim),), f"expected Shard({dim}) output, got {out.placements}"
    got = out.full_tensor().cpu()
    assert got.shape == ref.shape
    assert _pcc(got, ref) >= _PCC


@pytest.mark.parametrize("mode", ["eager", "compile"])
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
        assert not any(_SDPA_OVERRIDEABLE in op for op in ops), (
            f"SDPA stayed atomic under autograd, which has no backward; post-aot ops: {sorted(ops)}"
        )

    for got, ref in zip(tts, refs):
        assert got.grad is not None
        assert _pcc(got.grad.cpu(), ref.grad) >= _PCC
