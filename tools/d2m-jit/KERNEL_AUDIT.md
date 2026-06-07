# d2m-jit kernel-feasibility audit

A look at what kernels we can author in `d2m-jit` today, where the gaps
are, and concrete sketches of what flagship kernels (softmax, RMSNorm,
SDPA, MoE expert) would look like once the next round of API surface
lands.

Status legend matches [TODO.md](TODO.md): 🔴 blocker · 🟡 missing surface
· 🟢 nice to have · ✅ available today.

---

## 1. Surface available today

| Capability | Status | Notes |
| --- | --- | --- |
| Per-tile eltwise (41 unary, 13 binary) | ✅ | `_eltwise_block` wraps `d2m.tile_*` in `linalg.generic` over the tensor of tiles |
| `where` ternary select | ✅ | Mask-and-blend |
| Matmul (`@`, `d2m.matmul`) | 🟡 | Only with zeros-prefilled accumulator; single-tile-per-shard verified |
| `remote_load` / `remote_store` | ✅ | Per-shard DMA shortcut |
| `remote_load` w/ multicast | 🔴 | `SplitUnifiedThread` assertion on grids > 1×1 ([TODO §2](TODO.md)) |
| `scf.for` / `scf.if` in kernel body | ✅ | Loop & branch |
| `async` / `await` / semaphores | ✅ | Multi-thread sync primitives |
| **Views** (`view`, `view_layout`, `permute`) | ✅ | Metadata reinterpretation, no data movement |
| `tilize` / `untilize` / `to_layout` | ✅ | Layout conversions |
| `zeros`, `full`, `empty` | ✅ | Host-side fill + `to_layout` |
| Float reductions (`reduce_sum`, `reduce_max`, `reduce_mean`) | ✅ | Keepdim row/col reductions using `tile_reduce_*`, reduction output layouts, implicit eltwise broadcast; cross-core reductions need a core gather/redistribute op |
| Broadcast (`tile_bcast`) | ✅ | `d2m.tile_bcast`, row/col/2d shorthands, method forms; lit + pytest coverage |
| In-kernel typecast (`tile_typecast`) | 🔴 | Host-side only via `tilize(dtype=...)` |
| Per-tile transpose (`tile_transpose`) | 🔴 | Not exposed — but logical permute via views is free |
| Row/col mask helpers | 🔴 | Causal-mask building block missing |
| Multi-output kernels | 🟡 | `num_outs` exists in the API, untested for >1 |
| DMA primitives (`dma_read`, `embedding`, `indexed_row_copy`, ...) | 🟡 | Not exposed; shortcuts via `remote_load` cover many cases |

---

## 2. Views — the chain-kernels-without-DMA story

This is the single biggest leverage point d2m-jit has over a naive
eltwise-per-kernel testbed, and worth surfacing first because it
changes how the other gaps feel.

`view` / `view_layout` / `permute` all lower to `d2m.view_layout`, which
is a **metadata reinterpretation of the underlying buffer**, not a data
shuffle. They produce a `LazyTensor` with `is_view=True`. Consequences:

- **K-transpose in attention is free.** `K_T = d2m.permute(K, 1, 0)`,
  then pass `K_T` straight to the next matmul kernel. No physical
  transpose pass, no DMA round-trip.
- **Per-head / per-batch splits cost nothing.** A `view_layout` lambda
  carves heads/batches without copying.
- **Pipelined kernels share buffers.** Kernel A writes `out`, Kernel B
  reads `view(out, ...)` directly. No intermediate `to_layout`.
- **Sliding-window / cyclic KV cache** is a view rewrite, not a copy.

Constraint: `to_host(view)` is rejected — you must materialise the
*final* result with `to_layout(v, v.layout)`. Intermediate kernels see
views fine.

```python
# Free K-transpose: permute is a view, the next kernel reads K_T.
K_T = d2m.permute(K, 1, 0)
qk = d2m.zeros(L_qk)
qk_matmul_kernel(Q, K_T, qk, ..., grid=g)   # Q @ K_T, no DMA between
                                            # permute and matmul
```

---

## 3. Kernel-by-kernel feasibility

### ✅ Buildable today

- **Eltwise fusions:** residual + GELU, bias-add + activation, masked-add
  via `where`, dropout-shape kernels (sans RNG).
- **Activation kernels:** GELU, SiLU, ReLU, sigmoid, tanh, hardsigmoid,
  SELU — anything in the unary table.
- **Single-tile-per-shard matmul:** `out = d2m.zeros(L)` prefill + `@`
  inside the kernel. Verified by `test_matmul_correctness_via_zeros`.
- **GEMM → activation chained via views:** matmul kernel writes `out`,
  next kernel reads `view(out, ...)` for elementwise activation, no DMA
  between.
- **Per-shard outer product / hadamard** patterns.
- **KV-cache writeback** at a computed index via `remote_store`.
- **Pointwise quant/dequant** (mul + add + `floor`).
- **Single-tile reductions:** row/column `sum`, `max`, and `mean` within each
  32x32 tile. Results use one-row/one-column output layouts; elementwise ops
  implicitly broadcast them back when combined with unreduced blocks.
- **Cross-core reductions:** row/column reductions that span cores need a core
  gather/redistribute op so partials can be collected from multiple cores and
  placed back onto the cores that own the reduced output layout.

### 🟡 Buildable with workarounds / scope cuts

- **SDPA at 1-tile granularity** (S ≤ 32, single head): Q×Kᵀ via
  permute-view + zeros-prefilled matmul + manually composed row softmax from
  reductions.
- **Multi-head attention scaffolding:** per-head views are free; the
  *compute* hits broadcast and matmul-accumulator limits.
- **MoE expert MLP body** (linear + GELU + linear): chained via views,
  but only single-tile shards until the matmul accumulator bug lands.
- **Embedding lookup:** doable via `remote_load` with computed indices
  instead of the missing `embedding` DMA primitive.

### 🔴 Blocked on missing pieces

- **Softmax over wide rows** — within-tile pieces exist, but a single fused
  cross-tile implementation still needs accumulator/state handling.
- **LayerNorm / RMSNorm / GroupNorm over wide rows** — reduction tiles and
  one-tile output layouts exist, but reductions spanning cores need a core
  gather/redistribute op.
- **Flash Attention** — needs softmax + multi-K matmul accumulation
  ([TODO §1](TODO.md)) + multicast on real grids
  ([TODO §2](TODO.md)) + online-softmax state across blocks.
- **MoE top-k / sparse dispatch** — no top-k primitive; sparse dispatch
  needs `indexed_row_copy` / `embedding` DMA primitives.
- **Causal masks** — `where` covers the predicate, but no helper to
  *build* a triangular mask cheaply (`write_row_mask_tile`).
- **Real-grid matmul throughput** — multicast assertion + accumulator
  init together cap matmul to per-shard.

---

## 4. Concrete kernel sketches with reductions

These sketches use the landed reduction API:

```python
# Per-tile reductions: reduce over one tile axis. `dim` follows torch/numpy
# numbering (`0`/`-2`, `1`/`-1`) and keeps the reduced dimension as size 1.
# Elementwise ops implicitly broadcast reduced operands when needed.
d2m.reduce_sum(x, dim)
d2m.reduce_max(x, dim)
d2m.reduce_mean(x, dim)
```

The point of the sketches is the **kernel shape**; exact helper names may still
be adjusted as the DSL evolves.

### 4.1 Row-wise softmax (within-tile)

The simplest interesting case: each row of a 32×32 tile is one softmax
row. No cross-tile state needed.

```python
@d2m.kernel
def softmax_row(x, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])

            # Numerically-stable softmax: subtract row max, exp, divide.
            row_max = reduce_max(t, dim=1)               # keepdim row max
            t_shift = t - row_max
            t_exp   = exp(t_shift)
            row_sum = reduce_sum(t_exp, dim=1)           # keepdim row sum
            t_out   = t_exp * recip(row_sum)

            remote_store(out, [m_off + m, n_off + n], t_out)
```

### 4.2 Row-wise softmax (across N tiles per row)

When the row spans multiple tiles, the within-tile sketch above doesn't
suffice — we need three passes over N: row-max, row-sum, normalize.
This is also the building block for flash attention's outer loop.

```python
@d2m.kernel
def softmax_row_wide(x, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    NEG_INF = full_tile(value=-1e30)   # init helper, see TODO arange_block

    for m in range(m_blocks):
        # Pass 1: row max across N tiles.
        row_max = NEG_INF
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            row_max = maximum(row_max, reduce_max(t, dim=1))

        # Pass 2: row sum of shifted exp.
        row_sum = full_tile(value=0.0)
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            t_exp = exp(t - row_max)
            row_sum = row_sum + reduce_sum(t_exp, dim=1)
        row_inv = recip(row_sum)

        # Pass 3: normalise and write back.
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            t_out = exp(t - row_max) * row_inv
            remote_store(out, [m_off + m, n_off + n], t_out)
```

Notes:
- `full_tile(value=...)` is shorthand for the missing init helpers in
  [TODO §"Init helpers"](TODO.md) (`arange_block` / `fill_arange_tile`).
- The third pass re-loads `x` and re-computes `exp(t - row_max)`. The
  alternative is a scratch buffer of `t_exp` tiles, which exceeds L1 for
  realistic sequence lengths — the standard flash-attention move is to
  recompute, which is what we do here.

### 4.3 RMSNorm

`y = x * gamma / sqrt(mean(x²) + eps)` — a single per-row reduction.

```python
@d2m.kernel
def rmsnorm(x, gamma, out, m_blocks, n_blocks, eps_tile):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    INV_N = full_tile(value=1.0 / (n_blocks * 32))

    for m in range(m_blocks):
        # Pass 1: sum of squares across N tiles.
        sum_sq = full_tile(value=0.0)
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            sum_sq = sum_sq + reduce_sum(t * t, dim=1)
        inv_rms = rsqrt(sum_sq * INV_N + eps_tile)

        # Pass 2: scale and write.
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            g = remote_load(gamma, [0, n_off + n])    # broadcast row
            remote_store(out, [m_off + m, n_off + n], t * g * inv_rms)
```

This is the cleanest end-to-end demo of reductions: one small kernel that
immediately generalises to LayerNorm (add a mean reduction) and GroupNorm.

### 4.4 Single-head SDPA, chained via views

This is where the view-as-cheap-composition story pays off. Three
kernels stitched by views, no DMA between stages.

```python
def sdpa(Q, K, V, scale):
    # Q, K, V: tensor<S x D x !tile<32x32, bf16>>, sharded.

    # Stage 1: scores = Q @ K^T  (K^T is a view, not a copy)
    K_T    = d2m.permute(K, 1, 0)
    scores = d2m.zeros(L_scores)
    qk_kernel(Q, K_T, scores, ..., grid=g)

    # Stage 2: scores *= scale; softmax(scores) in place
    probs = d2m.empty(L_scores)
    softmax_row_wide(scale_kernel_view(scores, scale), probs,
                     m_blocks, n_blocks, grid=g)

    # Stage 3: out = probs @ V
    out = d2m.zeros(L_out)
    pv_kernel(probs, V, out, ..., grid=g)
    return out.to_host()
```

What's enabling each piece:

- **Free Kᵀ:** `d2m.permute(K, 1, 0)` returns a view; `qk_kernel` reads
  it directly.
- **In-place softmax shape:** `softmax_row_wide` from §4.2.
- **Chained matmuls:** `probs` from softmax flows into `pv_kernel`
  without a `to_layout` round-trip.

What's still blocking even this single-head version on real shapes:
- Multi-K matmul accumulation for D > 32 ([TODO §1](TODO.md)).
- Multicast for real grids ([TODO §2](TODO.md)).
- `tile_typecast` for the standard bf16-K/V + fp32-softmax recipe.

### 4.5 Flash attention (block-wise, online softmax)

The flash-attention recipe is the §4.2 softmax interleaved with the
matmul, using a running max `m` and running sum `l` updated per K
block. The shape after the API additions:

```python
@d2m.kernel
def flash_attn_inner(Q, K, V, O, l_state, m_state, S_q, S_kv, D):
    # Per (head, q-block) iteration. Outer driver handles the per-block
    # tiling; this body shows the online update.
    for kv_block in range(S_kv):
        K_blk = remote_load(K, [kv_block, 0])         # tile of K
        V_blk = remote_load(V, [kv_block, 0])         # tile of V
        Q_blk = remote_load(Q, [block_q_idx, 0])

        # 1. partial scores
        s = Q_blk @ permute_tile(K_blk)               # needs tile_transpose
                                                      # or pre-permuted K view

        # 2. running max
        m_prev = m_state
        m_cur  = maximum(m_prev, reduce_max(s, dim=1))

        # 3. correction factor for previous accumulator
        alpha  = exp(m_prev - m_cur)

        # 4. exp of shifted scores, partial sum
        p      = exp(s - m_cur)
        l_cur  = alpha * l_state + reduce_sum(p, dim=1)

        # 5. update O: O = alpha * O + p @ V
        O      = alpha * O + (p @ V_blk)

        m_state = m_cur
        l_state = l_cur

    # Final normalisation: O / l
    remote_store(out, [block_q_idx, 0], O * recip(l_state))
```

Blockers beyond reductions: per-tile `tile_transpose` (or pre-permuted
K, which views already give us), multi-K matmul
accumulation, multi-output kernels (we'd want `m_state` / `l_state` as
state we can read back for debugging).

### 4.6 MoE: gate softmax + expert dispatch sketch

The gate is a small dense layer + softmax + top-k. The dispatch is
the hard part — it needs indexed DMA we don't have today. Sketching
just the gate so the shape is on the page:

```python
def moe_gate(x, W_gate):
    # logits = x @ W_gate  -- single matmul kernel
    logits = d2m.zeros(L_logits)
    gate_matmul_kernel(x, W_gate, logits, ..., grid=g)

    # probs = softmax(logits) -- chained via view
    probs = d2m.empty(L_logits)
    softmax_row_wide(logits, probs, m_blocks, n_blocks, grid=g)

    # top-k: BLOCKED -- no top-k primitive today. The standard workaround
    # is host-side argmax, but that defeats the on-device story.
    return probs
```

The expert MLP body (per-expert linear → GELU → linear) is just a
chain of matmul + eltwise kernels — same shape as §4.4's pieces — so
it lands for free once §4.4 lands. What MoE specifically needs that
nothing else does:

- **Top-k primitive** (no equivalent in `D2MGenericRegionOps.td`
  today; would need new ops).
- **Indexed DMA** (`indexed_row_copy`, `embedding`) for sparse
  dispatch / combine.

---

## 5. Critical-path unlocks, ranked

1. **Add core gather/redistribute for cross-core reductions.** Reductions cover
   the per-core pieces, but the DSL needs an op that gathers partial results
   from multiple cores and redistributes the reduced values to the output-owning
   cores.
2. **Fix matmul accumulator init** ([TODO §1](TODO.md) —
   `D2MToTTKernel` fill-pattern handling). Unlocks multi-K matmul →
   real GEMM, real attention Q×Kᵀ across K, FFN.
3. **Fix `SplitUnifiedThread` for multicast** ([TODO §2](TODO.md)).
   Unlocks scalable matmul/attention across the grid.
4. **`tile_typecast`** — bf16 K/V with fp32 softmax accumulator is the
   standard SDPA recipe and isn't expressible until this lands.
5. **Row/col mask helpers** — final piece for causal attention once
   1–3 land.
6. **Init helpers** (`full_tile`, `arange_block`) — small but every
   sketch above relies on `full_tile(value=...)`-style seeding.

After (1) + (2), we can plausibly demo:

- softmax kernel (§4.1, §4.2)
- RMSNorm (§4.3)
- a multi-tile GEMM
- single-head SDPA chained from a Q×Kᵀ view-matmul kernel into a
  softmax kernel into a ×V kernel (§4.4) — all stitched with views so
  no data hits DRAM between stages.
