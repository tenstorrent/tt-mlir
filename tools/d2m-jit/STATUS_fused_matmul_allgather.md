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

## Hardware (two boxes)

There are now **two** machines in play:

1. The `nsmith/d2m-ccl` localdev box (`/localdev/nsmith/src/tt-mlir`) is a
   **single-device Wormhole n150** (one chip, no inter-device fabric links).
   Fabric / multi-device-mesh work cannot be validated there — all fabric/mesh
   d2m-jit tests abort for lack of a 2nd device (environmental, not a code bug).
2. **`qb2-120-p04t06` (`/home/ttuser/src/tt-mlir`) is a 4-chip Blackhole p300c**
   with real inter-chip ethernet/fabric (the 4 chips form a degree-2 ring). This
   box **can** run multi-device/fabric work — the n150's hardware constraint no
   longer applies here. (_Added 2026-06-17._)

Caveats on the Blackhole box:
- **Worker grid is 10×11, not Wormhole's 8×8.** `global_semaphore()` used to
  hardcode `grid_shape=(8,8)`; it now defaults to reading the device worker grid
  from the system descriptor (`_device_worker_grid()` in `builder.py`), so it is
  arch-portable. The device tests were updated to drop the explicit `(8,8)`; the
  lit tests keep `(8,8)` because they register a **mock Wormhole** device.
- **Build/env gotchas** (must do every fresh session here): the d2m-jit Python
  pkg symlinks to source, but the C++ build (`build-d2m-jit/`) and the system
  descriptor must be rebuilt/regenerated to match HEAD or you get
  `no such option use-tensor-accessor-dma` / `system desc schema mismatch`. Run:
  `cmake --build build-d2m-jit --target d2m-jit ttrt` then `ttrt query
  --save-artifacts`, and add `build-d2m-jit/{python_packages,runtime/python}` to
  `PYTHONPATH` (env/activate points at `build/`, the wrong dir name).

### Multi-device test status on Blackhole (2026-06-17)

After rebuild + grid fix, `test/d2m-jit/test_mesh.py`:
- ✅ Pass: `test_mesh_shard*_1x2`, `test_mesh_compute_roundtrip_1x2`,
  `test_two_generic_matmul_gather`, `test_all_gather_1x2_lowers` (compile-only),
  and all of `test_semaphore.py`.
- ⚠️ **All fabric-enabled CCL tests** (`test_all_gather_1x2_roundtrip`,
  `test_*matmul_all_gather*`, `test_fused_*_all_gather`, multicore/streaming)
  failed at **device-open** with a **Fabric Router Sync timeout** in
  `initialize_fabric_and_dispatch_fw()`. **Root-caused (not a hardware/link
  problem): the tests hardcode `mesh((1,2))`, a 2-chip _subset_ of the 4-chip
  ring.** Device 0's ethernet cores wired to the out-of-mesh chips hang at
  `STARTED` (no partner kernel on the unopened chips). A no-kernel probe
  confirms: **1×2 throws for FABRIC_1D/1D_RING, but 1×4 and 2×2 bring up fine
  with any fabric config.** → On this box, **open the full physical mesh
  (1×4 / 2×2); sub-meshes can't train fabric.**

### Fabric works on the full mesh; 4-device all_gather FIXED (2026-06-18)

A 1×4 all_gather kernel (adapted from `test_all_gather_1x2_roundtrip`) **runs
end-to-end on Blackhole with no timeout/deadlock** — proving the CCL fabric path
works on the full mesh, and that the `remote_store`→DRAM deadlock is *not* what
was blocking here.

The 1×4 result was first a **block-diagonal** (PCC 0.5): device `d` held only its
own shard `s_d` at row-band `d`, zeros elsewhere — the cross-device mcast wasn't
distributing payload. **Root cause (a latent regression from enabling the
TensorAccessor DMA by default, `6125f5e13`):** `D2MLowerDMAToFullyIndexedForm`
deferred *all* non-mcast/non-local shard writes to the accessor path, including
cross-device fabric `remote_store`s. The accessor lowers a write to a **local
NOC write**, silently dropping the fabric multicast — so only the local
self-write landed (the sem-inc still went cross-device, which is why the kernel
didn't hang). The 1×2 tests never caught it because they never ran on real
multi-device hardware.

**Fix:** in `LowerDMAToFullyIndexedForm.cpp`, the accessor-deferral guard now
also excludes cross-device writes (`&& op.getStartDevice().empty()`), so fabric
writes lower to the fully-indexed form that `D2MToTTKernel` turns into a
`fabric_mcast_fast_write_any_len`. With the accessor on (default), the 1×4
all_gather is now correct (PCC 0.9999, full vstack on every column-band) and the
kernel emits the real cross-device payload write. Regression guard:
`test_mesh.py::test_all_gather_1x4_roundtrip` (gated ≥4 devices).

**Fused matmul+all_gather on the full mesh — WORKS (2026-06-18).** With the
fabric-write fix above, a fused distributed-matmul + all_gather (each device
computes `C_d = A_d @ B_d`, then all-gathers `C_d`) runs correctly on the 1×4
mesh: every device ends with `vstack(C_0..C_3)`, PCC passes. This is the
compute-produced-CB source feeding the multi-hop fabric mcast (the
`split-unified-thread-v2` compute→DM handoff) on the full mesh. Regression guard:
`test_mesh.py::test_matmul_all_gather_fused_1x4_roundtrip` (gated ≥4 devices),
alongside `test_all_gather_1x4_roundtrip`.

**Larger shards — WORKS (2026-06-18).** The fused 1×4 path scales beyond a
single tile per device: each device computing a 2×2-tile output with a K=2
reduction (A_d/B_d/C_d = 64×64) all-gathers correctly (PCC ~1.0; abs-diff ~0.07
because the f32 K-reduction routes through the SFPU). Verified to scale further
to 4×4-tile / K=4 shards (128×128 per device → 512×512 gathered, PCC 1.0).
Regression guard: `test_matmul_all_gather_fused_1x4_large_shards_roundtrip`
(2×2/K=2, gated ≥4 devices).

**Streaming all_gather on the full mesh — WORKS (2026-06-18).** Two-generic
streaming (grid-N `_mm_col` matmul writes `C_d`; a grid-1 link core streams it,
reading ONE tile and fabric-mcasting it across the 4-device line, reusing a
single 1-tile buffer) runs correctly on 1×4. The link core's L1 stays bounded to
one tile regardless of shard size. Regression guard:
`test_mesh.py::test_streaming_matmul_all_gather_1x4` (gated ≥4 devices).

**The `remote_store`→DRAM deadlock is now REPRODUCED (2026-06-18).** "Much larger
shards" needs the *gathered* output (num_devices × per-device tiles, stacked in a
column) to exceed the worker-grid height — which forces a **DRAM-staged output**
(Option 2). Minimal repro: take the working 1×4 streaming all_gather, N=2, and
flip `L_out` to `mem_space="dram"`. Fabric initializes fine on all 4 devices
(FABRIC_1D_RING), the program launches, then the kernel **hangs in the fabric
`remote_store`→DRAM** — no `Finish`, no completion (killed at timeout; the device
recovers cleanly afterward). So the hang is in the fabric write to a DRAM
destination, *after* a healthy bringup — distinct from the (now-fixed) accessor
mcast-drop and the (resolved) sub-mesh bringup timeout.

A no-kernel-change repro recipe lives at `/tmp/stream_mm_ag_1x4_dram.py`
(`N=2 MEM=dram`).

**Root-cause narrowed (2026-06-18)** by bisection:
- single-device **local** DRAM write works; fabric write to **L1** works; only
  the fabric write to a **DRAM** destination hangs.
- propagation: the stalled data fabric-write blocks the fabric connection, so the
  subsequent fabric `sem_inc`s never send and every device's trailing
  `semaphore_wait` hangs.
- **NOC-encoding ruled out:** the DRAM kernel runs on noc1 and the dst is
  `get_noc_addr_from_bank_id(bank, addr, noc_index)`; forcing it to noc0 did NOT
  fix the hang (reverted). (L1 fabric dst uses noc-independent translated coords.)
- remaining suspect is tt-metal-side: the EDM receiver writes the payload via
  `noc_async_write_one_packet_with_trid(dest, …, forward_and_local_write_noc_vc)`
  (`fabric_edm_packet_transmission.hpp`); the trid-tracked single-packet write to
  a DRAM dest seems not to complete (vs L1). See the
  `remote-store-dram-deadlock-repro` memory for the full bisection + next probes.

**Root cause found (2026-06-18)** by comparing to tt-metal's production CCL
`broadcast_tile_writer.cpp` (an all_gather/broadcast that mcasts to DRAM). The
EDM **connection setup is identical** (both use `RoutingPlaneConnectionManager`
via `append_routing_plane_connection_manager_rt_args`) — so the connection is not
the issue. The fabric **write** differs: production uses the official
`linear::experimental::fabric_multicast_noc_unicast_write_*` API with the dst
computed by a **TensorAccessor** (`linear::addrgen_detail::get_noc_address`),
whereas d2m uses a hand-rolled `fabric_mcast_fast_write_any_len` LLK with the dst
computed manually via `get_noc_addr_from_bank_id` (hard-coded round-robin
banking). The manual address works for L1 (noc-independent worker coords) but is
not a fabric-deliverable DRAM address → the receive-side write never completes →
backpressure → hang. (Consistent with every negative result: not noc-encoding,
mcast/unicast, ring/line, flush, or semaphore.)

**REAL root cause found via EDM DPRINT trace (2026-06-20).** Instrumenting the
EDM receive path + a verified during-hang DPRINT flush shows: for a DRAM output
the receiving EDM gets **zero** packets (no writes, no sem-incs), while L1 shows
full traffic. So the hang is **sender-side** — the DRAM kernel blocks at its
*first* fabric send (the `device_synchronize` barrier, before the data write).
Cause: the fabric datamovement thread runs on **NoC1** for DRAM output vs **NoC0**
for L1 (`nocIdx = 1 - processorIdx`; changing the output to DRAM shifts the DMA
lowering → processor index → NoC1), and the worker↔EDM fabric connection doesn't
work on NoC1 (the EDM expects worker comms on `edm_fabric_write_noc_index = 0`).
The "fabric write to DRAM" framing was wrong — the write is never reached; every
kernel-write-path fix failed because the problem is upstream.

**FIXED (2026-06-20).** Hypothesis confirmed by a bidirectional NoC-swap test
(forcing the L1 fabric kernel onto NoC1 also hangs; forcing DRAM onto NoC0 works).
The fix is in `ScheduleDMA.cpp`: its single-DM-thread path assigned
`processorIndex = writesDRAM ? 0 : 1` (procIdx 0 → NoC1), so a fabric store to
DRAM ran on NoC1 and deadlocked. Now a cross-device (fabric) store forces
procIdx 1 (NoC0); the NoC1 preference applies only to non-fabric DRAM writes.
Validated: the minimal repro (`MEM=dram`) and a **1×4 streaming all_gather to a
DRAM output** now pass (PCC ~1.0); all 1×4 fabric tests + `test_semaphore` still
pass. **Option 2 (DRAM staging) is unblocked.**

**Gather-geometry scaling — done for L1-resident shards (2026-06-20).** The naive
streaming gather laid the output on a `[num_devices*tiles, 1]` grid (one worker
core per tile), which exceeds the worker grid's ~10 rows once tiles/device grows.
A 2D reblock can't fix this (the gather dim maps to grid rows; the output is one
tile wide). The fix is **block-packing**: put each device's whole shard on ONE
core — input `[1,1]` grid + `block_shape=[N,1]`, output `[num_devices,1]` grid +
`block_shape=[N,1]`. The grid stays tiny (1 / num_devices cores) while
`block_shape` carries the N tiles, so the gather scales to large shards bounded by
per-core L1 (verified to N=32 = 4096×128 gathered; ~hundreds of tiles/device fit).
Regression guard: `test_all_gather_1x4_large_block_roundtrip` (N=16).

**Beyond per-core L1 → DRAM (still blocked; precisely localized 2026-06-20).** For
shards exceeding one core's L1, the gather must be DRAM-resident. A large DRAM
output (grid e.g. `[32,1]`) fails at execution with
`get_virtual_coordinate_from_logical_coordinates(... TENSIX row 31 ...)`. Localized:
the DRAM **buffer** itself is fine (createShardedBufferConfigForDRAMMemref uses
DRAM bank cores `(0,0)..(0,7)`; no buffer has a core ≥10). The failure is the
**`to_host` read-back**: `_emit_returns_and_finalise` lowers the gathered DRAM
tensor through `ToLayout`/`shard_to_full`, and that read-back path materializes it
as an **L1-sharded** `[32,1]` layout → 32 worker cores → row 31 invalid. Commit
699b980fe's DRAM "skip worker-grid virtualization" covered `EmptyOp::build`,
`getMemoryMapImpl`, `getDramMapShapeSymbols` — but NOT the `ToLayout`/`from_device`/
`shard_to_full` read-back lowering that `to_host` emits.

**Localization REVISED (2026-06-21) — it is NOT the read-back.** Isolation shows
the DRAM read-back works on its own: a single-device DRAM `[32,1]` round-trip
(`to_layout`+`to_host`) and a mesh DRAM `[32,1]` `mesh_shard`→`mesh_gather`→
`to_host` both succeed (maxdiff 0.0). The failure only appears when a **generic op
(kernel) writes** the large DRAM output and that result is then gathered. So the
blocker is the generic-op-writes-large-DRAM-output lowering, not `to_host`. NOT
yet pinned down: a minimal single-device kernel writing a DRAM `[16,1]` output
**timed out** (hang or very slow compile of the large unrolled DRAM-write kernel),
so there may be a *second* issue (slow-compile/hang of big DRAM-write kernels) on
top of the placement question. Needs a less flaky, deeper investigation.

**Next:** characterize the generic-op-writes-large-DRAM path (why the minimal
kernel hangs/compiles slowly; whether the placement still maps to worker cores).
Then ring topology and the 8×8 grid. (Much-larger shards within per-core L1
already work via the block-packing gather above —
`test_all_gather_1x4_large_block_roundtrip`.)

- The fabric→DRAM track (and the deadlock) now has a working fabric handshake to
  build on (use the full mesh). The single-device matmul→DRAM-scratch half can be
  prototyped on either box.

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
