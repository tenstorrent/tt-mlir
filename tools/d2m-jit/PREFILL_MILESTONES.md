# LLM prefill in d2m-jit — milestone plan

A concrete, staged plan for building a decoder-only LLM **prefill** forward
pass in `d2m-jit`, plus a set of pseudo-kernels (under
[`prefill/`](prefill/)) that are written as close to real d2m-jit as
possible, with the genuine gaps called out inline for implementation.

> **This doc supersedes the stale parts of [`KERNEL_AUDIT.md`](KERNEL_AUDIT.md)
> and [`TODO.md`](TODO.md).** Those were written before reductions,
> broadcast, `where`, multi-K `matmul` (incl. `transpose_b`), in-kernel
> `typecast`, `clamp_scalar`, and `tile_transpose` landed in `api.py`. The
> feasibility picture below reflects `api.py` as it stands today, verified
> against `test/d2m-jit/` (reductions, matmul, broadcasts all have on-device
> tests; only multicast is `@pytest.mark.skip`).

---

## Re-baselined surface (what actually exists today)

| Capability | Status | Evidence |
| --- | --- | --- |
| Reductions `reduce_sum/max/mean(x, dim)` (dim `0/1/-2/-1`) | ✅ on device | `test_reductions.py` incl. multi-tile-single-core |
| Broadcast `tile_bcast` / `_row` / `_col` / `_2d` | ✅ | `test_broadcasts.py` |
| `where` ternary select | ✅ | `test_zeros_full_where.py` |
| Matmul incl. multi-K and `transpose_b` | ✅ on device | `test_matmul.py` (`transpose_b_correctness`) |
| In-kernel `typecast(x, dtype)` | ✅ | `test_bespoke.py::test_typecast_*` |
| `clamp_scalar`, `tile_transpose` | ✅ | `test_bespoke.py` |
| Eltwise (41 unary / 13 binary), `silu`/`gelu`/… | ✅ | `test_eltwise.py` |
| Views / `permute` / `view_layout` | ✅ | `test_views.py` |
| Matmul accumulator **auto-init** | 🟡 partial | works via `d2m.zeros` out-param; see `_matmul_block` TODO |
| Multicast `remote_load` on grid > 1×1 | 🔴 | `test_mcast_overwrite_grid_2x2` skipped (`SplitUnifiedThread.cpp:127`) |
| Cross-core reduction (reduced axis sharded across grid) | 🔴 | `test_reduction_layout_rejects_cross_core_reduction` |
| Causal / triangular mask builder | 🔴 | not present |
| Top-k selection | 🔴 | not present |
| Data-dependent gather (`embedding`, `indexed_row_copy`) | 🔴 | not present |
| Kernel-body init helpers (`full_tile`, `arange`) | 🔴 | host `d2m.full` + `remote_load` workaround |
| Multi-output / online-softmax state ergonomics | 🟡 | `num_outs` exists; >1 untested |

**Headline:** a **single-core dense transformer prefill layer is buildable
essentially today** (norm → attention → residual → norm → FFN → residual).
The remaining blockers are about (a) *scale/throughput* (multicast), (b)
*MoE/sparse routing* (top-k, gather), and (c) *ergonomics* (accumulator init,
mask builder, init helpers).

---

## Missing-primitive registry (the implementation call-outs)

Each pseudo-kernel tags its gaps with `# ⛔ NEEDS[<ID>]`. Grep the
[`prefill/`](prefill/) tree for an ID to find every call site.

| ID | Primitive / fix | Priority | Status | Workaround | Gates |
| --- | --- | --- | --- | --- | --- |
| `ACC-INIT` | Matmul accumulator auto-init | P1 | 🟡 zeros-prefill works | `out = d2m.zeros(L)` out-param | all matmuls (ergonomics) |
| `MCAST` | Multicast on grid > 1×1 (`SplitUnifiedThread`) | **P0 (perf)** | 🔴 | single-core only | M5, M8 |
| `CMASK` | Causal row/col mask builder (`write_row_mask_tile`) | P1 | 🔴 | host mask tensor + `where` | M1, M6 |
| `TOPK` | Top-k token/expert selection | **P0 (MoE)** | 🔴 | host argmax/top-k | M7, M8 |
| `GATHER` | Data-dependent gather (`embedding`, `indexed_row_copy`) | P1 | 🔴 | host embed / dense-masked combine | M3, M7, M8 |
| `XREDUCE` | Cross-core reduction | P2 | 🔴 | keep reduced axis on one core | M5 |
| `INIT` | Kernel-body `full_tile` / `arange` | P3 | 🔴 | host `d2m.full` + `remote_load` | M0+ |
| `MOUT` | Multi-output / online-softmax running state | P2 | 🟡 | recompute / single-out | M6 |
| `ABSORB` | MLA weight-absorption fusion pattern | P2 | not built | keep two matmuls | M8 |

---

## Milestones

Single-core unless stated. Each milestone names the pseudo-kernel(s) that
specify it.

### M0 — Building blocks (compose + golden-test) — *mostly green today*
- Kernels: [`rmsnorm`](prefill/rmsnorm.py), [`softmax`](prefill/softmax.py),
  [`rope`](prefill/rope.py), multi-K GEMM via `zeros`, SwiGLU activation.
- Work: the primitives exist and are individually tested; M0 is composing
  them into named prefill sub-kernels with PCC golden tests.
- Blockers: none hard. Ergonomics: `ACC-INIT`, `INIT`.

### M1 — Attention block (single head, single core, materialized scores)
- Kernel: [`sdpa_prefill`](prefill/sdpa_prefill.py) — `QKᵀ` (via
  `matmul(transpose_b=True)`) → scale → causal mask → softmax → `@V` →
  output proj, with RoPE on Q/K.
- Blockers: `CMASK` (host-mask workaround), `ACC-INIT`.

### M2 — FFN block (single core)
- Kernel: [`swiglu_ffn`](prefill/swiglu_ffn.py) — `gate_proj`/`up_proj` →
  `silu·mul` → `down_proj`.
- Blockers: `ACC-INIT`.

### M3 — Single dense transformer layer (single core) — **first end-to-end prefill layer**
- Kernel: [`transformer_layer`](prefill/transformer_layer.py) — composes
  M1 + M2 with residual adds and the two RMSNorms.
- Blockers: `GATHER` for the embedding (host-embed workaround). Achievable
  ~today at toy width with the workarounds in place.

### M4 — Multi-layer dense prefill (single core, real width)
- Stack N layers via a host Python loop, staging activations through DRAM
  between layers. Llama-class dense model, correct but single-core-slow.
- Blockers: orchestration/capacity (the one-lazy-graph builder + L1 budget);
  activations must round-trip DRAM at layer boundaries.

### M5 — Grid parallelism — **throughput** unlock
- Multicast matmul + attention across the grid; sharded norm/softmax.
- Blockers: **`MCAST`** (P0 for any real prefill speed), `XREDUCE` (for
  reductions whose axis is sharded across cores).

### M6 — Flash-SDPA (fused online softmax) + causal-mask builder
- Kernel: [`flash_sdpa_prefill`](prefill/flash_sdpa_prefill.py) — long-context
  attention with running `m`/`l` state, no materialized `[S,S]` matrix.
- Blockers: `CMASK` builder, `MOUT` (running state), `ACC-INIT`.

### M7 — MoE prefill block
- Kernel: [`moe_block`](prefill/moe_block.py) — router (matmul + softmax,
  green) → **top-k** → expert SwiGLU (M2) → weighted combine (green,
  dense-masked) → sparse dispatch.
- Blockers: **`TOPK`** (P0), `GATHER` (sparse dispatch; dense-masked
  formulation sidesteps it for a testbed).

### M8 — DeepSeek-V4 MLA + sparse attention
- Kernel: [`mla_attention`](prefill/mla_attention.py) — latent down/up
  projections (green matmuls), `ABSORB` weight-absorption fusion, decoupled
  RoPE, sparse "lightning indexer" (top-k token selection), KV compressor.
- Blockers: `TOPK`, `GATHER`, `ABSORB`, `MCAST`.

---

## Dependency graph

```
M0 (blocks) ──┬──> M1 (attention) ──┬──> M3 (dense layer) ──> M4 (N layers, single core)
              │                      │                              │
              └──> M2 (FFN) ─────────┘                              └──> M5 (grid / multicast)  [MCAST]
                     │
M1 ──> M6 (flash-SDPA)  [CMASK, MOUT]
M2 ──> M7 (MoE)  [TOPK, GATHER]
M1 + M7 ──> M8 (DeepSeek MLA + sparse)  [TOPK, GATHER, ABSORB, MCAST]
```

## Recommended order

1. **M0 + M1 + M2 + M3** are gated on *ergonomics only* (`ACC-INIT`,
   `CMASK` host-workaround). Land these first — they prove an end-to-end
   single-core prefill layer with today's primitives.
2. **`MCAST` (M5)** is the single highest-value compiler fix for prefill
   *performance* — without it everything is single-core.
3. **`TOPK` (M7/M8)** is the only hard blocker with no on-device workaround;
   it gates all MoE/sparse-attention work.
4. `CMASK` builder and `ACC-INIT` auto-init are quality-of-life fixes that
   remove the two standing workarounds.

See [`prefill/`](prefill/) for the pseudo-kernels these milestones specify.
