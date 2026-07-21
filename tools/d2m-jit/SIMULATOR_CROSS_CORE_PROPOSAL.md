<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Design Proposal: Cross-core stores in the d2m-jit simulator

**Status:** Phase 0 + Option C landed; Option A investigated and found N/A
(2026-07-21). See §8–§9.
**Author:** (d2m-jit simulator work, branch `jgrim/d2m-jit-sim`)
**Scope:** the one remaining Phase 2 simulator gap — kernels where multiple
cores write into a shared output whose per-core target depends on the
`d2m.generic` op's grid / indexing maps / iterator types. This covers both the
deferred `test_mcast_overwrite_grid_2x2` and distributed ("cross-core")
reductions, which are the same underlying problem.

Related docs: `SIMULATOR_SPEC.md` (the simulator's design), `README.md` (the
DSL surface).

---

## 1. Problem

The simulator runs a `@d2m.kernel` body **natively** — it rebinds the body's
globals to torch-backed builtins and calls the Python function once per grid
core (`_src/sim/runtime.py::run_kernel`). It never constructs the `d2m.generic`
op that the device path builds, so it has no model of the op's grid iteration
space, `indexing_maps`, or `iterator_types`.

`remote_load` / `remote_store` take **grid-dimension indices** and move an
entire shard, per the op contract:

> RemoteStoreOp indices correspond to the _grid dimensions only_ (first N/2
> dimensions of the device shape). The RemoteStoreOp always stores an entire
> shard to the corresponding operand.
> — `include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td:2236`

The simulator implements this literally: `remote_store(out, [i, j], blk)` writes
global shard `[i, j]` of `out` (`runtime.py::remote_store` → `SimTensor.write_block`).

This is correct for **all-parallel** kernels, which compute the global target
explicitly from the core index. E.g. the eltwise add (paraphrased from
`test/d2m-jit/test_simulator.py`):

```python
m_off = core_index(0) * m_blocks
n_off = core_index(1) * n_blocks
...
remote_store(out, [m_off + m, n_off + n], a + b)   # global target, per core
```

Each core writes a distinct global shard; the sim matches device. ✅

It is **wrong** for the multicast/reduction shape. In
`test/d2m-jit/test_matmul.py::mcast_overwrite_kernel` every core stores to a
*local* index with no core offset:

```python
cy = core_index(0); cx = core_index(1)
for k in range(K):
    for m in range(M):
        lhs_shard = remote_load(lhs, [cy*M + m, k], mcast_start_index=[cy, 0], mcast_shape=[1, GX])
        for n in range(N):
            rhs_shard = remote_load(rhs, [k, cx*N + n], mcast_start_index=[0, cx], mcast_shape=[GY, 1])
            remote_store(out, [m, n], lhs_shard + rhs_shard)   # local index [0,0] when M=N=1
```

With `K=M=N=1` every core stores to `[0, 0]`, yet the test's golden expects each
core's result to land at its **own** `[cy, cx]` shard. The simulator runs the
cores sequentially and all four write global `[0, 0]`, so the last writer wins
and the other three shards stay zero. ❌ (This is the single numeric failure in
`D2M_JIT_SIM=1 pytest test/d2m-jit/`.)

The same structure blocks distributed reductions: a kernel where each core
reduces its locally-owned blocks and the partials must combine into a shared
output shard. (Per-core reductions — one core loads several blocks along the
reduce axis and reduces them — already work, because those loads use global
indices; that is why `test_reductions.py` passes in sim.)

### The core tension

Whether a store index is "global" or "core-relative" is **not** expressed in the
kernel body. It is a property of the `d2m.generic` op the body is spliced into —
its `grid`, `block_factors`, `indexing_maps`, and `iterator_types` — which the
device builder assembles in `_src/builder.py::_emit_kernel_generic`. A purely
syntactic simulator cannot tell the two kernels apart, and any heuristic that
"fixes" the mcast kernel risks breaking the all-parallel convention that
currently passes.

---

## 2. What is not yet known (must be pinned first)

Static reading is not enough to design the fix, because the op contract says
indices are global, yet the mcast golden implies a core-relative store. One of
these must be true, and we do not currently know which:

1. **The store really is global on device**, and the mcast golden is satisfied
   by some other mechanism (e.g. the store is lowered against the output
   operand's indexing map such that a "local" index is offset by the core's grid
   position; or `block_factors` / the empty `indexing_maps` default changes the
   addressing).
2. **The store is effectively core-relative** for this generic shape (reduction
   iterator on `K`, output map projecting `K` out), and the all-parallel
   eltwise kernel only *looks* global because its explicit offset happens to
   equal the core-relative resolution.
3. **The test passes on device for a different reason** than the sim assumes
   (e.g. it is a routing smoke test tolerant under PCC, or it is xfail/skipped
   on some device configs).

**Phase 0 — empirical investigation (blocking, ~0.5–1 day).** Before writing any
simulator code:

- Run `mcast_overwrite_kernel` and the eltwise add on device; capture the actual
  output tensors.
- Dump the `d2m.generic` op for each **before** and **after** the pipeline
  (`d2m.config.print_ir_before/after`, or the hooks the lit tests use). Record
  `grid`, `block_factors`, `indexing_maps`, `iterator_types`, and the operand
  device shapes.
- Trace how `remote_store`'s `[i, j]` is lowered against the output operand —
  specifically whether the DMA address is `[i, j]` verbatim or `[i, j]` composed
  with the output indexing map and the core's grid coordinate.
- Confirm the eltwise and mcast kernels genuinely resolve their store targets by
  the *same* rule (they must, since they use the same op with default maps).

The output of Phase 0 is a one-paragraph precise statement of the store-target
rule as a function of `(grid, block_factors, indexing_maps, iterator_types,
core_index, store_index)`. Everything below assumes that rule is known; the
recommended option is chosen to be robust to whichever of (1)/(2) turns out true.

---

## 3. Design options

All options keep native execution of the body (the simulator's core design
decision in `SIMULATOR_SPEC.md`); they differ in how `remote_store`/`remote_load`
resolve a store index to a global shard.

### Option A — Model the generic op's grid + indexing maps (faithful)

Give the simulator a first-class notion of the generic op that the body runs in:

- At dispatch, `SimCompiledKernel.__call__` already receives `grid`,
  `block_factors`, `indexing_maps`, `iterator_types` (currently accepted and
  ignored — `host.py`). Build a small `SimGenericContext` capturing them plus
  each operand's grid/shard shape.
- `run_kernel` iterates the grid exactly as today, but stores the context in the
  per-core `CoreContext`.
- `remote_load`/`remote_store` resolve their index through the **operand's
  indexing map** evaluated at the current grid point, using the same
  global-vs-core-relative rule established in Phase 0. Reduction iterator dims
  are projected out of the output map (matching the device), which is what makes
  a "local" store land at the core's own parallel-grid position.
- Distributed reduction correctness: writes to a shard that several cores target
  become read-modify-write accumulation *iff* the device semantics say so;
  otherwise last-writer-wins, faithfully.

**Pros:** correct by construction for reductions, multicast, and future generic
shapes; matches the device's own abstraction; unlocks the whole class, not one
test. **Cons:** largest change; needs the Phase 0 rule exactly right; must
reproduce affine-map projection logic that today lives in the compiler.

### Option B — Grid-context shim with an explicit core-relative mode

Add the `SimGenericContext` (grid + operands) but *not* full indexing-map
evaluation. Default `remote_store` stays global (unchanged, protects eltwise).
Detect the reduction/multicast shape — reduction `iterator_types`, or a store
index provably independent of `core_index` while the grid is >1 — and resolve
those stores core-relative (offset the local index by the core's grid origin).

**Pros:** much smaller; keeps the passing convention untouched; fixes the mcast
test and simple distributed reductions. **Cons:** heuristic; the "independent of
core_index" detection is fragile under native execution (we see concrete integer
indices per core, not the symbolic dependence); risks silently mis-resolving a
kernel that is neither clearly global nor clearly core-relative.

### Option C — Honor explicit maps only; reject the ambiguous shape

The kernel-call API already accepts `indexing_maps` / `iterator_types`. Make the
simulator honor them when supplied (Option-A resolution, but only when the user
is explicit), and raise a clear `NotImplementedError` for map-less kernels whose
stores are not manifestly global (i.e. the mcast/reduction shape). Keep the
all-parallel default working.

**Pros:** smallest; no heuristic guessing; honest failure instead of a wrong
number; composes toward Option A later. **Cons:** does not actually fix the
deferred test unless that test is updated to pass explicit maps; pushes work onto
kernel authors.

---

## 4. Recommendation

**Phase 0 (investigate) → Option C (land) → Option A (grow into).**

1. Do Phase 0 to pin the exact device rule. This is non-negotiable — every
   option depends on it, and it is cheap relative to guessing wrong.
2. Land **Option C** first: teach `remote_load`/`remote_store` to resolve through
   supplied `indexing_maps`/`iterator_types`, and replace the current silent
   wrong-answer on the mcast shape with a precise `NotImplementedError` (mirrors
   how async-generator kernels are handled today — fail loud, not silent). This
   is a strict improvement and carries no risk to the passing suite.
3. Grow Option C's resolver into **Option A** by inferring the default maps the
   device builder uses when the user omits them, so map-less reduction/multicast
   kernels work without annotation. At that point `test_mcast_overwrite_grid_2x2`
   moves from "deferred" to a passing sim gate.

Option B is not recommended as the endpoint: its heuristic detection is
especially unreliable under native execution, where the simulator observes
concrete per-core index *values* and cannot see a store index's symbolic
dependence on `core_index`.

---

## 5. Testing strategy

- **Gate:** `test/d2m-jit/test_matmul.py::test_mcast_overwrite_grid_2x2` becomes
  the acceptance test for the multicast shape. It is currently the one numeric
  sim failure; flipping it green (without regressing the other 125 numeric
  passes) is the definition of done for the multicast case.
- **New distributed-reduction test:** add a kernel where each core reduces its
  own blocks and the partials combine into one shard, with a hand-computed torch
  golden. This exercises the accumulation path Option A introduces and is not
  redundant with the per-core reductions already in `test_reductions.py`.
- **Parity harness:** the sim-vs-device PCC harness (`test/d2m-jit/test_parity.py`,
  `runner.py`, `SIMULATOR_SPEC.md §12`) is the strongest check — run both kernels
  through device and sim and assert PCC. This catches a resolver that is
  self-consistent but wrong.
- **Regression:** the full `D2M_JIT_SIM=1 pytest test/d2m-jit/` numeric suite must
  stay green (currently 126 pass / 2 skip; the 19 fails are out-of-scope-by-design
  compiler-path tests + this one mcast test).

---

## 6. Risks & non-goals

- **Risk — wrong Phase 0 rule.** Mitigated by landing Option C (which only acts
  on explicit maps) before any inference, and by gating Option A on the parity
  harness against real device output.
- **Risk — regressing the all-parallel convention.** Mitigated by keeping global
  resolution the default and adding map-driven resolution as an additive path
  guarded by the full numeric suite.
- **Risk — reproducing compiler affine-map logic in the sim.** The projection of
  reduction dims out of the output map must match the compiler. Keep the
  resolver small and test it directly against dumped device maps from Phase 0.
- **Non-goal — performance/ordering fidelity.** Consistent with
  `SIMULATOR_SPEC.md`: semaphores and multicast routing remain functional
  no-ops; this proposal only concerns *where* a store lands, not *when* or *how*
  it is routed.
- **Non-goal — arbitrary generic shapes.** Target the two shapes that matter now
  (all-parallel; parallel-grid + reduction/multicast). Broader shapes can extend
  the resolver later.

---

## 7. Rough sizing

- Phase 0 (investigate + write the rule): ~0.5–1 day.
- Option C (map-honoring resolver + loud error + tests): ~1 day.
- Option A (infer default maps, flip the mcast gate green, distributed-reduction
  test): ~2–3 days, dominated by matching the compiler's default-map inference
  and validating via the parity harness.

---

## 8. Phase 0 results + what landed (2026-07-21)

**Device investigation.** On a Wormhole device
(`SYSTEM_DESC_PATH=ttrt-artifacts/system_desc.ttsys`):

- `test_mcast_overwrite_grid_2x2` **passes on device** — each core's local
  `remote_store(out, [0, 0], ...)` lands at that core's own `[cy, cx]` shard.
  So the store is effectively core-relative for this shape.
- Pre-pipeline `d2m.generic` dumps for both kernels (all-parallel eltwise vs the
  multicast kernel) show **identical, empty** `block_factors`, `indexing_maps`,
  and `iterator_types`, and `grid = 2x2`. The divergence is entirely in the
  body:
  - Eltwise: `remote_store %out[core0*m_blocks + m, core1*n_blocks + n]` — the
    index already carries the core offset (**global**).
  - Multicast: `remote_store %out[m, n]` with `m,n` the loop vars and **no** core
    offset; the loads carry `mcore[...] mshape[...]` — lhs multicasts along grid
    dim 1, rhs along dim 0, so the multicast collapses **both** grid dims.

**Rule (confirmed):** a `remote_store` index is core-relative along exactly the
grid dims a multicast collapses, and global elsewhere. The trigger the simulator
can observe is the multicast arguments on the `remote_load`s.

**What landed (Option C, blended toward A).** `_src/sim/runtime.py`:

- `CoreContext` tracks `mcast_dims` — grid dims collapsed by multicast during the
  current core-run. `remote_load` records a dim as collapsed when its
  `mcast_shape` span is > 1 (or it is named in `mcast_dims`).
- `remote_store` resolves core-relative along those dims:
  `target[d] = core_coord[d] + index[d]` (one output block per core). Dims not
  collapsed stay global, so all-parallel kernels are unchanged.
- The multi-block-per-core case (`block_factors > 1`) raises a loud
  `NotImplementedError` pointing at Option A, rather than silently mis-placing.

**Validation.** `D2M_JIT_SIM=1 pytest test/d2m-jit/` → 127 pass / 2 skip / 18
fail; the 18 are all out-of-scope-by-design compiler-path tests (`test_errors`
compiler diagnostics, `test_config` pipeline prints, `test_patterns` MLIR
rewrite/e2e). **The entire numeric suite is green, including
`test_mcast_overwrite_grid_2x2`, which runs in both device and sim mode and
whose golden is the device output** — so the resolver is device-validated, not
just self-consistent.

**Remaining (Option A).** Multi-block-per-core (`block_factors > 1`) stores and a
dedicated distributed-reduction test still need the generic op's indexing maps
modeled. The `NotImplementedError` marks exactly where that work plugs in.

---

## 9. Option A investigation — found not applicable (2026-07-21)

Following the proposal's own recommendation (validate against device before
implementing), both Option A targets were probed on device. **Neither has a
valid, device-runnable form**, so there is no correct behavior for the simulator
to match, and no Option A code was written.

**Target 1 — multi-block-per-core core-relative (multicast) stores.** A
multicast collapses a grid dim, and the layout verifier requires a collapsed dim
to be evenly divisible by the grid extent:

> Collapsed dimension must be evenly divisible by grid dimension, got 1 % 2 != 0.
> — `lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:973` (constructing a multicast
> operand layout with 2 blocks per core along the collapsed dim)

So a multicast dim is **always exactly one block per core**; more than one block
per core along a collapsed dim cannot even be constructed. The `per_core == 1`
resolver from §8 is therefore complete for every valid multicast kernel, and its
`NotImplementedError` guard covers a state that valid kernels cannot reach — a
safety net, not a missing feature.

**Target 2 — distributed (cross-core) reductions.** Reducing along a dim the grid
splits produces an output grid smaller than the compute grid, which the generic
op rejects:

> 'd2m.generic' op output grid shape must be divisible by the generic op's grid
> shape

(probe: input grid `2x2`, reduce a grid-split dim → output grid `2x1`, not
divisible by `2x2`). This is structural: any reduction that collapses a
grid-split dim shrinks the output grid below the compute grid. Consequently
**cross-core reductions are not expressible in a single `d2m.generic` today** —
this is a DSL/compiler limitation, not a simulator gap. `reduction_layout`'s
`allow_cross_tile=True` lets you build the output *layout*, but the op verifier
still rejects the grid mismatch, and no kernel in the tree uses it. The
reductions that *are* supported keep the reduced dim on one core (`grid[dim] ==
1`); those store with global indices and already pass in sim
(`test_reductions.py`, including the multi-block and rectangular-grid variants).

**Conclusion.** The simulator's cross-core story is complete for the kernels the
compiler actually accepts. Option A becomes relevant only if the compiler later
grows real multi-core reductions / gather-redistribute ops; at that point the
`NotImplementedError` in `_resolve_core_relative` is the plug-in point, and this
section records the device evidence for why it was deferred.
