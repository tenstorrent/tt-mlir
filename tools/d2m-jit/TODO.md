# d2m-jit TODO

A running list of known issues, pipeline gaps, and unfinished API
surface for the d2m-jit testbed. Tracked here (rather than in a tracker)
because this DSL is internal scratchpad and the items mostly need
compiler-side fixes rather than user-facing scoping.

Status legend: 🔴 active bug or blocker · 🟡 missing surface · 🟢 nice to
have.

---

## Pipeline gaps

---

## Missing API surface

These ops live in `D2MGenericRegionOps.td` but are not yet exposed in
`api.py`. Each is a "wait for a use case" item — the DSL is a testbed,
so we add ops when we have something to test against rather than
speculatively.

### 🟡 Reduction follow-ups

- Integer reductions via `tile_sfpu_reduce_sum` / `tile_sfpu_reduce_max`.
- Cross-core or multi-dim (RC) reductions. Reductions spanning multiple cores
  need a core gather/redistribute op that can collect partials from the cores
  that own the reduced dimension and place the reduced values on the
  output-owning cores.

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

- Kernel-body `full([m, n], value)` for block literals. `zeros([m, n])` exists
  for accumulator blocks; a general fill helper would make the pattern reusable.
- `arange_block(layout)` — fill a block with `0, 1, 2, ...` (per the d2m op).
- `fill_arange_tile()` — per-tile arange.

Useful for golden tests and for replacing the current
`test/d2m-jit/utils.py:arange_tile` host-side helper with a true device-side
fill.

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
