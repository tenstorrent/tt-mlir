# Status & roadmap — fused matmul + all_gather toward 8×8 (fabric→DRAM)

Living status doc for the d2m-jit fused matmul + all_gather effort on branch
`nsmith/d2m-ccl`. Companion to the design docs:
`fused_matmul_allgather_8x8_design.md` (the 8×8 plan + Option 1/Option 2),
`fabric_router_cores_design.md`, `core_read_write_spec.md`,
`unified_semaphore_design.md`, `CCL_SPEC.md`.

_Last updated: 2026-06-17._

---

## The goal

A fused **matmul → all_gather** on Tenstorrent hardware, scaling to the full
**8×8** worker grid, with the gathered output staged through **DRAM** so it is
not bounded by a single link core's L1 capacity. The end target is the design
doc's **Option 2 (DRAM staging)**: matmul writes its per-device shard `C_d` to a
DRAM scratch tensor; the link core reads `C_d` from DRAM and fabric-writes it to
the peer device's DRAM (DRAM↔DRAM), avoiding the 64→1 L1 gather wall.

---

## Where we are

### Fused matmul + all_gather (the main line)

Two-generic structure (matmul on 8×8, no fabric → all_gather on the 1×1 link
core, fabric), per `fused_matmul_allgather_8x8_design.md`. Progress:

- ✅ Single-device rungs: two-generic matmul→gather; multicore matmul +
  core_read gather; streaming all_gather (bounded L1); scf.if-gated fabric
  on a DM thread.
- ✅ Fabric all_gather on a **1×2 mesh** (`483c3cc82`).
- ✅ **Option 1 (in-L1 `reblock` gather)** fused matmul + all_gather scaled to a
  **4×4** grid, 16→1 gather (`85ac40ca3`, `6595dc0dc`).
- ✅ DRAM-sharded tensors may exceed the worker-grid volume (`699b980fe`) — DRAM
  round-robins its 12 channels, so there is no ≤64 grid-volume cap like L1/SRAM.
- ✅ Fabric **router-cores** design steps 1–3 (`76177badb` surface,
  `48aa9024c` connection-manager gating, `a968a70ae` flatbuffer + runtime
  subset). Steps 4–5 of `fabric_router_cores_design.md` remain.

### TensorAccessor DMA + DPRINT (prerequisite detour — DONE)

Option 2 needs DRAM-addressed reads/writes; DRAM is globally addressable and the
**TensorAccessor** handles arbitrary buffer distribution, so the accessor DMA
path had to work and be the default. It now is:

- ✅ `c3355c01e` — **DPRINT fixed**. The `ttkernel.dprint`→emitc lowering emitted
  legacy per-arg `DPRINT("{}", arg)`, which under tt-metal's new fmt-style
  `DEVICE_PRINT` printed string literals (and `"\n"`) as pointer addresses, so
  lines never flushed. Now lowers to fmt-style `DPRINT(fmt, args…)`. Works in
  fast and slow dispatch.
- ✅ `37c5028bc` — **TensorAccessor page-granularity fix**. The accessor rewriter
  looped over the *element* shard shape with the *element* page size, but the
  runtime spec/accessor/strides work in *page* units. Tiled buffers were fine
  (element = tile = page); row-major buffers are stick-paged `(1, shardWidth)`,
  so the element loop used the stick stride → wrong source pages → scrambled
  tiles through `to_layout`'s tilize. Fixed by expressing the loop shape + page
  size in page units.
- ✅ `6125f5e13` — **accessor on by default + mcast/local coexistence**.
  `D2MLowerDMAToFullyIndexedForm` now always runs and, when the accessor is on,
  lowers only multicast + local-destination DMAs (plain shard DMAs go to the
  accessor); the accessor write rewriter defers multicast. Flipped the d2m-jit
  default on.

Full d2m-jit suite with the accessor on **matches the accessor-off baseline** —
no regressions. (All four commits are pushed to `origin/nsmith/d2m-ccl`.)

---

## Where we're going (next: Option 2, DRAM staging)

`fused_matmul_allgather_8x8_design.md` incremental step 3. On a **multi-device**
machine:

1. **matmul → DRAM scratch.** matmul generic (grid 8×8) writes `C_d` to a DRAM
   tensor. The matmul→DRAM→read-back half is single-device-testable (no fabric)
   and can be prototyped first.
2. **all_gather from DRAM over fabric.** Link core (grid 1×1) `remote_load`s
   `C_d` from DRAM and `remote_store`s over fabric to the peer's DRAM. **This is
   where the open `remote_store`→DRAM deadlock must first be reproduced, then
   fixed.**
3. **Validate** the 8×8 fused matmul + all_gather end-to-end on the 1×2 mesh with
   PCC (matmul f32 K-reduction routes through SFPU fp19 ≈1% error — use PCC, not
   tight abs-diff).
4. Finish **router-cores steps 4–5** (`fabric_router_cores_design.md`) as needed
   for the fabric-on-a-subset story.

---

## Hard constraint: hardware

The `nsmith/d2m-ccl` localdev box (`/localdev/nsmith/src/tt-mlir`) is a
**single-device Wormhole n150** (one chip, no inter-device fabric links).

- Fabric / multi-device-mesh work **cannot run or be validated there** —
  `ttcore-register-device{mesh-shape=1,2}` aborts (no 2nd device). All
  fabric/mesh d2m-jit tests (`test_mesh`, `test_scf_if_gated_fabric`, the 1×2
  all_gather) abort there for this reason; **this is environmental, not a code
  bug.**
- The fabric→DRAM track (and the deadlock) therefore needs a **multi-device
  machine**. The single-device matmul→DRAM-scratch half can be prototyped on the
  n150.

---

## Open issues

- **`remote_store`→DRAM deadlock** (the headline blocker). A hang when fabric
  `remote_store` targets a DRAM buffer (vs L1). Not yet root-caused; reproduce on
  multi-device hardware first.
- **Router-cores steps 4–5** remain (`fabric_router_cores_design.md`).
- **`test_mcast_overwrite_grid_2x2`** fails on PCC (a pre-existing multicast
  *routing correctness* bug) — independent of the accessor; fails the same way
  with the accessor off. Worth a separate look.
- **L1 capacity** bounds Option 1's in-L1 gather; Option 2 (DRAM) is the way past
  it. A 256×256 f32 shard (256 KB) fits the link core's ~1.5 MB L1 but leaves
  little headroom — beyond that, DRAM staging or chunked streaming is required.

---

## Debugging notes

- **DPRINT now works** (commit `c3355c01e`) for kernel-side debugging:
  `D2M_ACCESSOR_DPRINT=1 TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_FILE=<path>`
  (the env-gated `D2M_ACCESSOR_DPRINT` probe prints per-core
  `my_logical y/x`, pageId, bank, resolved NoC addr for each accessor DMA), and
  `ttkernel.dprint` generally.
- **Avoid `TT_METAL_WATCHER=1`** — it crashes (`std::unexpected` in
  `poll_watcher_data`) in this tt-metal pin.
- Accessor opt-out: `D2M_JIT_USE_TENSOR_ACCESSOR_DMA=0`.

---

## Key references

| What | Where |
|---|---|
| 8×8 design + Option 1/2 | `tools/d2m-jit/fused_matmul_allgather_8x8_design.md` |
| Fabric router cores | `tools/d2m-jit/fabric_router_cores_design.md` |
| core_read / core_write | `tools/d2m-jit/core_read_write_spec.md` |
| Semaphore model | `tools/d2m-jit/unified_semaphore_design.md` |
| CCL primitives | `tools/d2m-jit/CCL_SPEC.md` |
| Accessor DMA path | `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp` (`D2MDMAViaTensorAccessorRewriter`) |
| DMA coexistence | `lib/Dialect/D2M/Transforms/LowerDMAToFullyIndexedForm.cpp`, `lib/Dialect/D2M/Pipelines/D2MPipelines.cpp` |
