# d2m-jit TODO

A running list of known issues, pipeline gaps, and unfinished API
surface for the d2m-jit testbed. Tracked here (rather than in a tracker)
because this DSL is internal scratchpad and the items mostly need
compiler-side fixes rather than user-facing scoping.

Status legend: 🔴 active bug or blocker · 🟡 missing surface · 🟢 nice to
have.

---

## Pipeline gaps

### 🔴 Matmul accumulator init breaks `d2m → ttkernel` lowering

**Where:** `_matmul_block` in `api.py`, when the `outs` operand to the
inner `linalg.generic { d2m.tile_matmul }` is anything other than a
fresh `d2m.empty`.

**Symptom:** wiring a `linalg.generic { d2m.tile_fill }` as the
accumulator producer fails late in the pipeline:

```
error: failed to legalize unresolved materialization from
  ('memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>')
  to ('!ttkernel.cb<1, !ttcore.tile<32x32, f32>>')
  that remained live after conversion
 note: see existing live user here:
  ttkernel.pack_tile(%c0, %8, %c0, true)
    : (index, !ttkernel.cb<1, !ttcore.tile<32x32, f32>>, index) -> ()
```

**Root cause:** `D2MToTTKernel` doesn't recognise the
`linalg.generic { d2m.tile_fill }` pattern as the producer of the
matmul accumulator. The conversion emits an
`unrealized_conversion_cast` from `memref<...>` to `!ttkernel.cb<...>`
that no later pass can fold away, while `ttkernel.pack_tile` keeps
using the CB form.

**Workaround for users:** pre-fill the output via `out = d2m.zeros(L)`
and pass it as the out-param to a kernel that calls `@`. The current
test `test_matmul.py::test_matmul_correctness_via_zeros` does this.

**Fix shape:** teach `D2MToTTKernel` to handle a fill-pattern producer
(or to emit a CB-init prologue for the matmul kernel when one is
detected). A `tile_matmul` op that internally zero-initialises on the
first reduction step would also work.

---

### 🔴 Multicast kernels hit `SplitUnifiedThread` assertion

**Where:** any kernel that uses the multicast form of `remote_load`
(`mcast_start_index`, `mcast_shape`) on a grid larger than 1×1.

**Symptom:**

```
python: lib/Dialect/D2M/Transforms/SplitUnifiedThread.cpp:127:
  wrapComputeInSynchronizedRegion: Assertion
  `opsWithSynchronizableOps.size() == 1 && "synchronized scope must be
  unambiguous"' failed.
```

**Root cause:** `wrapComputeInSynchronizedRegion` expects exactly one
op-with-a-synchronizable-op inside a `d2m.GenericOp` when wrapping the
compute thread, but a multicast `remote_load` kernel produces multiple
such ops (the load itself plus the elementwise body), so the pass aborts.

**Impact:** the multicast smoke test in
`test/d2m-jit/test_matmul.py::test_mcast_overwrite_grid_2x2` is
currently marked `@pytest.mark.skip` for this reason.

**Fix shape:** teach `SplitUnifiedThread`'s
`wrapComputeInSynchronizedRegion` to handle multiple synchronizable ops
in the same scope (or split them into separate synchronized scopes).

---

## Missing API surface

`tile_bcast` is exposed today via `d2m.bcast(x, bcast_type)` for
`"row"`, `"col"`, or `"scalar"`, the `bcast_row` / `bcast_col` /
`bcast_scalar` shorthands, and matching `TensorBlock` methods. It has
both lit IR-shape coverage and pytest end-to-end coverage, so it is no
longer tracked as missing surface.

These ops live in `D2MGenericRegionOps.td` but are not yet exposed in
`api.py`. Each is a "wait for a use case" item — the DSL is a testbed,
so we add ops when we have something to test against rather than
speculatively.

### 🟡 Remaining bespoke-signature ops (need design)

| op | why it's interesting | what's blocking |
| --- | --- | --- |
| `tile_clamp_scalar(x, min, max)` | clamp with attribute (not operand) bounds | needs `FloatAttr` / `IntegerAttr` threading through `_eltwise_block`, plus a wrapper that picks the attr type from the tile's underlying dtype |
| `tile_typecast(x)` | in-kernel dtype conversion (host-side already covered by `tilize(dtype=...)`) | needs an `_eltwise_block` variant that takes a target element type different from the input |
| `tile_transpose(x)` | per-tile (32×32) element transpose -- distinct from logical `permute` / `view` | naming question — collides with `permute` / `view` semantics if called `transpose` |

### 🟡 Reductions (scoped — float blocked, int viable)

`tile_reduce_sum`, `tile_reduce_max`, `tile_reduce_mean` (float) and
`tile_sfpu_reduce_sum`, `tile_sfpu_reduce_max` (int).

**Locked V1 design** (deferred behind blockers below):

- Free functions `reduce_sum / reduce_max / reduce_mean(block, dim)`
  reducing along numpy-style axis (`dim=0 → R`, `dim=1 → C`).
- Two flavours per op: same-shape broadcast-back (default, for
  `x - x.max(dim, keepdims=True)`-style softmax/layernorm prefixes) and
  `*_collapse` (output axis collapsed to a single tile).
- Float vs int auto-dispatch via tile-element-type inspection.
- Multi-dim (RC) reduction deferred — call twice if needed.
- Anchor test: softmax-prefix `exp(x - max(x))` on a single shard.

#### Blocker A: float reductions need a scaler `b` operand

`tile_reduce_*` (float) signature is `(a, b, c, dim) -> reduce(a*b)+c`.
TTIR-to-D2M lowering supplies `b` as a separate 1×1-tile **tensor input**
to the outer `d2m.GenericOp`, built by a *separate* `d2m.GenericOp` whose
body fills the tile with `1.0` via `tile_fill`. The scaler is loaded into
the reduce kernel via `remote_load(scaler, [0, 0])`.

#### Blocker B: inline `tile_fill` in the reduce body fails to lower

Attempted shortcut: emit `tile_fill(1.0)` and `tile_reduce_sum` as
sibling ops inside the same `linalg.generic` body, so no host-side
scaler tensor is needed. Builds clean IR but `D2MToTTKernel` cannot fold
the resulting conversion cast:

```
error: failed to legalize unresolved materialization from
  ('memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>')
  to ('index')
  that remained live after conversion
```

Root cause: `D2MTileFillRewriter` replaces the tile_fill result with a
DST-register index, but the reduce rewriter's `getCB(op.getB())` call
expects `b` to be a CB-backed tile (not a DST tile). Same family as the
matmul accumulator init bug above.

#### What V1 would actually cost

If we go with the host-allocated scaler (the only path that lowers
today):

1. Walk the kernel AST at call time and detect `reduce_*` calls.
2. Auto-host-allocate a 1×1-grid scaler tensor via
   `d2m.full((32, 32), 1.0)`.
3. Append the scaler to the outer `d2m.GenericOp` inputs and to the
   inner kernel func signature as a synthetic argument.
4. Inside the reduce primitive, emit `remote_load(scaler, [0, 0])`
   once at body entry and cache the resulting tile (re-used across
   multiple reduce calls in the same kernel).
5. Plumb compiler state (thread-local or `D2MCompiler.symbol_tables`
   sentinel) so the primitive can find the scaler from inside
   `visit_Call`.
6. Handle dtype matching (scaler must match the input tile's float
   dtype: f32 vs bf16 vs f16).

Estimated cost: **2–3 days** of DSL plumbing. Plus risk: the
`remote_load(scaler, [0, 0])` from a >1×1 grid is structurally a
multicast read from shard (0,0), which exercises the same path that
SplitUnifiedThread blows up on (see TODO above). So grid>1×1 float
reductions are likely blocked behind the same compiler bug.

#### Recommended near-term

- Land int-only `reduce_sum / reduce_max` via `tile_sfpu_reduce_*` (no
  scaler). ~30 min of work. Provides the API shape for users to mirror.
- Defer float reductions until either (a) `D2MToTTKernel` accepts a
  tile_fill-produced `b` operand directly (Blocker B fix), or (b) the
  scaler plumbing in the DSL is worth the 2–3 day investment.
- Defer softmax/layernorm anchor tests until float lands.

### 🟡 Lower-level kernel primitives (advanced)

These are for users writing data-movement-heavy kernels that don't
shortcut through `remote_load` / `remote_store`:

- **DMA:** `dma_read`, `dma_write`, `dma_wait`, `local_copy`,
  `indexed_row_copy`, `embedding`, `null_tx`.
- **Dst register / scratch:** `acquire_dst`, `dst_reinterpret_cast`,
  `scratch_allocate`, `scratch_init`, `set_l1_accumulate`,
  `unpack_stall_on_pack`, `operand_alias`.
- **CB management:** `push`, `pop`, `reserve` (we had these informally
  on the deleted `Stream` class).
- **Sync:** `synchronized_region`, `device_synchronize`.
- **Mesh:** `mesh_position`.
- **Runtime args:** `get_arg`, `get_block_factor`, `get_cb`.
- **Masks:** `write_col_mask_tile`, `write_row_mask_tile`.

### 🟡 Index queries inside a kernel body

`iter_index`, `block_index`, `block_offset` — alternative ways to query
the current iteration position from a kernel body. `core_index` is the
only one we expose today. Trivial one-liners modelled on `core_index`,
but they're meaningful only in generic-op forms we haven't surfaced
(`block_factors` / `indexing_maps` / `iterator_types` on `@kernel`).

### 🟡 Init helpers

- `arange_block(layout)` — fill a block with `0, 1, 2, …` (per the
  d2m op).
- `fill_arange_tile()` — per-tile arange.

Useful for golden tests and for replacing the current
`test/d2m-jit/utils.py:arange_tile` host-side helper with a true
device-side fill. Need the same plumbing the device-side `zeros`/`full`
fill would need (see the host-scope linalg gap above).

### 🟡 Kernel-side `print`

`d2m.print` exists but we ripped out the `visit_Print` path when we
stripped ttkernel-specific code from `D2MCompiler`. Useful for
debugging kernel bodies; would need a thin wrapper that emits
`d2m.print` (not the `ttkernel.dprint` of the old code).

---

## Documentation / DX

### 🟢 Lit-side IR-shape FileCheck tests

`test/d2m-jit/lit/` now has coverage for captures, error paths,
pattern rewrites, and broadcast lowering. Worth expanding that into
lit + FileCheck tests that dump pre-pipeline IR (the builder already
supports `print_ir_before_pipeline`) and check the shape of more DSL
primitives — this locks down the IR contract without going to silicon.

Sketch:

```python
# RUN: %python %s | FileCheck %s
# REQUIRES: d2m-jit
import d2m_jit as d2m
import torch
d2m.config.print_ir_before_pipeline = True
t = torch.zeros(64, 64)
L = d2m.Layout(shape=(64,64), dtype=d2m.float32, block_shape=[1,1], grid_shape=[2,2])
try:
    d2m.to_layout(t, L).to_host()
except Exception:
    pass  # don't actually run on device
# CHECK: d2m.to_layout
# CHECK: d2m.view_layout
# CHECK: return
```

### 🟢 More worked examples

The README's "At a glance" has eltwise add. Multi-output, reductions,
and views-into-kernel examples would help newcomers. Defer until the
reduction / multi-output paths exist.

### 🟢 `_eltwise_block` as a public helper

Users writing their own ops can mirror what `api.py` does, but the
helper is currently private (`_`-prefixed). If we want to encourage
DSL extension from outside this directory, drop the leading underscore
and document it.

---

## Internal cleanup

### 🟢 `_BUILDER._instance` lifecycle

The builder is a class-level singleton. Test files reset it manually
(`_b._Builder.reset()`) in a few places to avoid contamination across
test functions. The `conftest.py::_set_seed` fixture could also reset
the builder per-test for hygiene.

### 🟢 Stale `_fill_block_value` reference

The `_matmul_block` TODO mentions a host-scope fill prototype that no
longer exists in the code. Update the comment when the gap above is
resolved, or rephrase to be implementation-agnostic.
