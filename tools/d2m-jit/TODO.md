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

### 🔴 Host-scope `linalg.generic` doesn't lower

**Where:** any DSL emission that produces a `linalg.generic` at
host-`func.func` scope (outside a `d2m.GenericOp` region).

**Symptom:**

```
error: 'builtin.module' op found linalg.generic operations that were
  not converted to affine loops.
 Please run --d2m-linalg-to-affine before the d2m-insert-dst-register-
  access-unscheduled / d2m-insert-dst-register-access-scheduled passes.
```

**Root cause:** `d2m-linalg-to-affine` only walks `linalg.generic`s
**inside** `d2m.GenericOp` regions. Host-scope `linalg.generic`s
survive into later passes that don't expect them.

**Impact:** blocks every host-scope linalg pattern. We hit it when
prototyping a device-side `d2m.zeros(L)` via
`linalg.generic { d2m.tile_fill }`; we ended up using a host-side
`torch.zeros` + `to_layout` round-trip instead.

**Workaround for the DSL:** avoid emitting `linalg.generic` at host
scope. Use `to_layout` from a host-side `torch` tensor instead.

**Fix shape:** make `d2m-linalg-to-affine` (or a sibling pass) also
handle the host-func-scope case, or have the DSL synthesise a
`d2m.GenericOp` wrapper around host-side fills so they go through the
existing path.

---

## Missing API surface

These ops live in `D2MGenericRegionOps.td` but are not yet exposed in
`api.py`. Each is a "wait for a use case" item — the DSL is a testbed,
so we add ops when we have something to test against rather than
speculatively.

### 🟡 Bespoke-signature ops (need design)

| op | why it's interesting | what's blocking |
| --- | --- | --- |
| `tile_clamp_scalar(x, min, max)` | clamp with attribute (not operand) bounds | needs `FloatAttr` / `IntegerAttr` threading through `_eltwise_block`, plus a wrapper that picks the attr type from the tile's underlying dtype |
| `tile_typecast(x)` | in-kernel dtype conversion (host-side already covered by `tilize(dtype=...)`) | needs an `_eltwise_block` variant that takes a target element type different from the input |
| `tile_transpose(x)` | per-tile (32×32) element transpose -- distinct from logical `permute` / `view` | naming question — collides with `permute` / `view` semantics if called `transpose` |
| `tile_bcast(x, bcast_type)` | broadcast row / col / scalar tile to full tile | needs a `BcastTypeAttr` argument; small but bespoke |

### 🟡 Reduction follow-ups

- Integer reductions via `tile_sfpu_reduce_sum` / `tile_sfpu_reduce_max`.
- Collapsed-output helpers (`*_collapse`) for APIs that want a logical
  single-row or single-column result rather than same-shape reduce lanes.
- Cross-tile or multi-dim (RC) reductions; call kernels in multiple passes for
  now.

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

`test/d2m-jit/lit/` only has `captures.py` (AST inspection). Worth
adding lit + FileCheck tests that dump pre-pipeline IR (the builder
already supports `print_ir_before_pipeline`) and check the shape of
what each DSL primitive emits — would lock down the IR contract
without going to silicon.

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
