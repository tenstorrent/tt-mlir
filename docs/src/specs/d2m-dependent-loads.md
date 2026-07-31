# Dependent loads in D2M

*Status: design note + working prototype. **Verified on n150 silicon** — see
"Silicon verification" below.*

A *dependent load* is a memory access whose address is computed from data that
itself lives in device memory: `tl.load(base + tl.load(idx_ptr + i))`. Triton
emits these freely. D2M today cannot express them, because every access in a
generic region is tile-granular and every address is an affine function of
loop induction variables and the core index — all resolved at compile time.

This note specifies how to express dependent loads by giving `memref.load` /
`memref.store` a second, scalar meaning that is only legal in a data movement
thread.

## Background: `memref.load` is already overloaded

In a D2M generic region `memref.load` does not read anything. On a tile-typed
memref it *materializes a CB slot index*, and the actual data movement is fused
into the consuming `memref.store`:

```mlir
%0 = memref.load %cb[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
memref.store %0, %dst[%c0] : memref<1x!ttcore.tile<32x32, f32>, #dst>
// becomes: ttkernel.copy_tile(%cb, 0, 0)
```

`MemrefLoadRewriter` in `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`
implements the load half: it simply replaces the op with its linearized index.
`MemrefStoreRewriter` pattern-matches the load/store pair and emits
`copy_tile` or `pack_tile`.

This is the tile contract the Triton frontend runs into. Note it is a *contract
of the compute region*, not of the dialect: nothing about `memref.load` on an
`i32` memref has a tile interpretation.

## The design

Split the meaning of `memref.load` / `memref.store` by (thread type, element
type):

| context | op / element type | meaning |
| --- | --- | --- |
| compute region | `memref.load` of `!ttcore.tile<...>` | CB slot index; fused into `copy_tile`/`pack_tile` (today's behavior) |
| datamovement region | `memref.load` of a signless 8/16/32-bit integer in `l1` | **real scalar read of L1 by the RISC-V core** |
| host command function | any non-tile, memory space `system` | CPU-side buffer access (today's behavior) |

Concretely: run `d2m-split-unified-thread` as normal; write dependent loads as
plain `memref.load`; and lower them *conditionally* in `ConvertD2MToTTKernel`
based on the enclosing function's `d2m.thread` attribute.

**Reads only.** Scalar *stores* to L1 are not supported and are rejected with a
diagnostic. Supporting them would make the datamovement thread a scalar CB
*producer*, which needs the read/write pointer question settled and the CB
accounting extended; nothing in the dependent-load use case needs it.

**Signless 8/16/32-bit integers only.** `ttkernel.l1_addr_ptr` only has
`tt_l1_ptr uint{8,16,32}_t` flavors, and the arith ops a loaded index feeds are
signless, so `ui32`/`si32` cannot be used for addressing at all. `index`-element
CBs are not representable downstream. Each of these is rejected by name — see
`scalar_l1_access_invalid.mlir` — because otherwise they fail deep in the
conversion with an unresolved materialization, an invalid `arith.index_cast`, a
`ttkernel` op verifier message, or an assertion in `getIntOrFloatBitWidth`.

### Lowering

A scalar L1 access is a pointer dereference on the CB's base address. The
TTKernel ops for this already exist and already have EmitC lowerings — they are
used today by the `d2m.indexed_row_copy` path:

```
memref.load %cb[%i] : memref<32xi32, #l1>
=>
%base = ttkernel.get_read_ptr(%cb)          : (!ttkernel.cb<32, i32>) -> i32
%ptr  = ttkernel.reinterpret_cast(%base)    : (i32) -> !ttkernel.l1_addr_ptr
%off  = arith.index_cast %i                 : index to i32
%v    = ttkernel.load_from_l1(%ptr, %off)   : (!ttkernel.l1_addr_ptr, i32) -> i32
```

which `ConvertTTKernelToEmitC` turns into exactly the C the frontend wants:

```cpp
tt_l1_ptr uint32_t* v14 = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_arg_0.get_read_ptr());
tt_l1_ptr uint32_t  v17 = v14[v16];
```

Multi-dimensional accesses linearize with the existing `computeLinearIndex`
helper. The element width picks the `tt_l1_ptr` flavor.

`get_read_ptr` is the right base because `d2m-insert-scalar-access-cb` has
bracketed the read with a CB `wait`, making the read pointer address the page the
transfer filled.

### Why not "leave `memref.load` intact and let EmitC handle it"

`ConvertD2MToTTKernel` already marks `memref.load` on integer element types as
legal, so the obvious plan is to leave the op alone and let the
`MemRefToEmitC` patterns (already wired into `ConvertTTKernelToEmitC`) turn it
into a subscript. That does not work, and the failure is immediate:

```
error: failed to legalize unresolved materialization from ('!ttkernel.cb<32, i32>')
to ('memref<32xi32, #ttcore.memory_space<l1>>') that remained live after conversion
```

The memref a datamovement thread loads from is the result of `d2m.wait` /
`d2m.reserve` on a CB, and the type converter rewrites it to
`!ttkernel.cb<...>`. A *legal* `memref.load` is left with an operand whose type
has changed underneath it and no target materialization to bridge the gap.
Upstream `MemRefToEmitC` also only handles identity-layout memrefs it can turn
into `emitc.array`; it has no notion of a CB handle. Hence: a conversion
pattern, not a legality exemption.

## Ordering: what makes the read see the data

Three separate things have to hold for a dependent load to observe the
transferred data. All are enforced in the passes rather than assumed:

1. **The scalar read must participate in the CB protocol.** A scalar reader is a
   CB *consumer*, but it is not one of the things `d2m-insert-compute-cb` knows
   to look for. Without a `wait`/`pop` pair the transfer half reserves and pushes
   with nothing popping, so the read addresses the front of the buffer rather
   than the page just written, and a loop exhausts the CB's pages and blocks.
   `d2m-insert-scalar-access-cb` supplies the pair.
2. **The transfer filling the buffer and the transfer consuming the loaded value
   must be on the same data movement thread**, so that program order applies at
   all. `d2m-schedule-dma` otherwise load-balances CBs freely across threads and
   will separate them.
3. **The filling transfer's barrier must not sink past the scalar read.**
   `d2m-optimize-dma` coalesces transfers by sinking each `dma_wait` as late as
   legal, and a scalar read is not one of the ops it previously knew to stop at.

The target is the sequence `d2m.indexed_row_copy` already lowers to by hand in
`loadI32FromNocPacketThroughScratch` — reserve, transfer, barrier, push, wait,
read, pop, all on one thread. That helper is the existence proof that a CB
produced and consumed by the same data movement thread works: the counters are
local L1, and the embedding path has been doing it inside a loop all along.

Miss (2) or (3) and the result is a silent race whose outcome depends on DRAM
latency; miss (1) and a loop reads stale data and then hangs.

## Silicon verification

Run on an n150 (wormhole_b0), via a fixture built by injecting a dependent load
into pipeline-generated pre-split IR: the weights tilize generic streams shard
`[core0, core1]` from DRAM, and the injection adds an i32 index buffer as a
second `ins` operand plus its own L1 CB, scalar-reads one i32 out of it, and uses
that value as the *row* the weights transfer reads.

With `ttrt run --init arange` the index tensor is `index[r][c] = r*128 + c`, so
core `(i, j)` reads `index[32i][32j] = 4096i + 32j`, and `divui 4096` recovers `i`
exactly. Two variants were built from the same fixture, differing only in the
arithmetic applied to the loaded value:

| variant | shape | row formula | predicted mapping | result |
| --- | --- | --- | --- | --- |
| reversal | straight line | `7 - (perm / 4096)` | out tile-row `i` ← weights tile-row `7-i` | all 8 rows correct |
| rotation | straight line | `(perm / 4096 + 3) % 8` | out tile-row `i` ← weights tile-row `(i+3)%8` | all 8 rows correct |
| refill | wait/pop **inside** a 4-iteration loop | `7 - (sum(4 reads) / 4 / 4096)` | out tile-row `i` ← weights tile-row `7-i` | all 8 rows correct, no hang |
| hoisted | wait/pop **bracketing** the loop | same | same | all 8 rows correct, no hang |

The two loop variants are what make the pop load-bearing. In `refill` the index
transfer is inside the loop, so the pair must balance per iteration: the CB has 2
pages and the loop runs 4 times, so without the pop the third `reserve_back` would
block forever. In `hoisted` the transfer is outside and only the reads are inside,
so the pair must bracket the whole loop -- a pair placed *inside* would wait a
second time on a CB pushed once and hang. Both completed in ~2.6s (the same as the
straight-line runs) against a 240s timeout, so neither deadlocked.

Full-tensor max absolute error was 15.0 in both cases (0.09% relative), which is
exactly the error the *baseline* `abs` program shows on the same input — a
pre-existing math-fidelity characteristic of the TTMetal path, not something the
dependent load introduces.

The test discriminates: against the wrong hypotheses the same output is off by
~24000-28000 (identity permutation: 28673; rotate-by-1: 24579), three orders of
magnitude above the noise floor. And the two variants disagree with each other by
16391, so the loaded value provably determines the address rather than the
mapping being an artifact of the fixture.

What this establishes on hardware:

- The `reserve` / transfer / barrier / `push` / `wait_front` / `get_read_ptr` /
  dereference / `pop` sequence on a CB produced *and* consumed by the same data
  movement thread does not deadlock, and the read observes the transferred data
  rather than a stale page.
- A value read scalar-wise out of L1 by the RISC-V core correctly reaches the
  address of a subsequent NoC read.
- `d2m-schedule-dma`'s affinity grouping keeps the index transfer and the
  dependent transfer on one DM core in a real program (both landed in
  `datamovement_kernel0`, `dm_core = 1`).

Building the loop variants found a real bug: `collectDependentDMAOps` in
`d2m-schedule-dma` did not follow a value out of a loop through `scf.yield`, so a
dependent load whose value is accumulated across iterations looked like it reached
no transfer at all, the two CBs were never united, and they could land on different
DM cores. The `d2m-insert-scalar-access-cb` "does not fill it" diagnostic caught
this at compile time rather than letting it race. Fixed, with lit coverage
(`dependent_load_through_loop_carried_value`) confirmed to fail without the fix.

Not covered by these runs: multi-core grids beyond 8x4, and loop trip counts
larger than the CB depth by more than 2x.

## What was prototyped

Working tree changes (not committed):

- `include/ttmlir/Dialect/D2M/Utils/Utils.h`, `lib/Dialect/D2M/Utils/Utils.cpp`:
  `isScalarL1AccessType` / `getScalarL1AccessMemref`, the shared predicate for
  "this access is a real RISC-V read/write of L1".
- `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`: `isScalarL1Access`,
  `buildScalarL1Ptr`, `MemrefScalarL1LoadRewriter`,
  `MemrefScalarL1StoreRewriter`, registered at benefit 2 so they win over the
  tile-index interpretation.
- `lib/Conversion/D2MToTTKernel/D2MToTTKernelPass.cpp`: scalar L1 accesses in a
  datamovement thread are now illegal, so the patterns run.
- `include/ttmlir/Dialect/D2M/Utils/CBUtils.h`, `lib/Dialect/D2M/Utils/CBUtils.cpp`:
  `getScalarL1AccessPort`, the shared resolution from a scalar access to the CB
  port it touches. Scalar accesses name the operand buffer directly rather than a
  `!d2m.cb` handle, so the buffer is traced through views back to the generic's
  operand list; once bracketed it names the `d2m.wait` result instead, which
  resolves through the CB handle.
- `lib/Dialect/D2M/Transforms/InsertScalarAccessCB.cpp` (new pass
  `d2m-insert-scalar-access-cb`): brackets the scalar reads of each CB with one
  `d2m.wait`/`d2m.pop` pair and rewires the reads onto the `wait` result (point 1
  above). Runs after `d2m-schedule-dma`, so each datamovement region is final and
  the pair lands only on the thread that owns the CB. The pair is placed at the
  *transfer's* loop depth, not the read's: an index block re-filled every
  iteration gets a pair inside the loop, one filled once and indexed many times
  gets a pair bracketing the whole loop. Shapes a single pair cannot express — a read outside
  the block that fills the CB, or one preceding the transfer — are diagnosed
  rather than silently mis-synchronized.
- `lib/Dialect/D2M/Transforms/ScheduleDMA.cpp`: CB affinity groups (point 2
  above). CBs are load-balanced as union-find groups rather than individually;
  a scalar read's CB is united with the CBs of every DMA op transitively
  consuming the loaded value. Scalar accesses are also filtered per thread
  alongside DMA ops, since a scalar *store* has write effects and would
  otherwise be replicated into every thread the region was cloned into. An
  access that cannot be attributed to a CB port (a region-local scratch
  allocation) cannot be filtered, so the pass declines to split at all.
- `lib/Dialect/D2M/Transforms/DMAOptimizations.cpp`: `canBarrierSinkPast` now
  blocks a `dma_wait` from sinking past a scalar L1 access of the same CB
  (point 3 above). This is now belt-and-braces for a bracketed read — the
  inserted `d2m.wait` is already a barrier blocker — but it is the only
  protection for a CB the new pass leaves alone.
- Tests: `test/ttmlir/Conversion/D2MToTTKernel/scalar_l1_access.mlir`,
  `test/ttmlir/Dialect/D2M/Transforms/schedule_dma_dependent_load.mlir`,
  `test/ttmlir/Dialect/D2M/Transforms/dma_optimizations_dependent_load.mlir`,
  `test/ttmlir/Dialect/D2M/Transforms/insert_scalar_access_cb.mlir`,
  `test/ttmlir/Dialect/D2M/Transforms/insert_scalar_access_cb_invalid.mlir`.
  The schedule-dma and dma-optimizations tests were confirmed to fail with their
  respective guards disabled.

The scheduling change is a no-op for any generic without scalar L1 accesses:
groups come out as singletons in the order CBs were previously enumerated, so
the greedy assignment and the resulting NoC choice are unchanged. No existing
test expectations needed updating.

Generated loop body, index block re-filled each iteration:

```cpp
for (size_t i19 = v5; i19 < v3; i19 += v4) {
  cb_arg_1.reserve_back(v6);
  noc0.async_read(dram_ep, CoreLocalMem<uint32_t>(cb_arg_1.get_write_ptr()), ...);
  noc0.async_read_barrier();
  cb_arg_1.push_back(v6);
  cb_arg_1.wait_front(v6);
  tt_l1_ptr uint32_t* v20 = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_arg_1.get_read_ptr());
  tt_l1_ptr uint32_t v23 = v20[v22];
  cb_arg_1.pop_front(v6);
  cb_arg_0.reserve_back(v9);          // the dependent transfer, addressed by v23
  ...
}
```

and filled once, indexed many times — one of each, bracketing the loop:

```cpp
cb_arg_1.reserve_back(v6);
noc0.async_read(...);
noc0.async_read_barrier();
cb_arg_1.push_back(v6);
cb_arg_1.wait_front(v6);
for (size_t i19 = v5; i19 < v3; i19 += v4) {
  tt_l1_ptr uint32_t* v20 = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_arg_1.get_read_ptr());
  ...
}
cb_arg_1.pop_front(v6);
```

Verified:

- Constant and loop-carried dynamic indices, 16- and 32-bit element widths, 1D
  and 2D (linearized) accesses, loads and stores.
- The loaded scalar flows into a `d2m.dma_read` grid index and reaches the NoC
  read address in the generated C++.
- `d2m-split-unified-thread` places the scalar load on the datamovement side
  and dead-code-eliminates it from the compute side, with no changes needed to
  `AssignThreads`/`SplitThreads`. `memref::LoadOp` is read-only, so once its
  datamovement-only consumers are erased from the compute region it is
  trivially dead there.
- The full backend pipeline (`split-unified-thread` → `schedule-dma` →
  `lower-load-store-ops-to-dma` → `lower-dma-to-fully-indexed-form` →
  `normalize-thread-args` → `generic-regions-to-funcs` →
  `convert-d2m-to-ttkernel`) carries the dependent load through: the index
  buffer resolves to a `cb_port` runtime arg and the load becomes
  `get_read_ptr`/`reinterpret_cast`/`load_from_l1`.
- All 172 existing `test/ttmlir/Dialect/D2M` + `test/ttmlir/Conversion/D2MToTTKernel`
  lit tests still pass.

## Open issues

### 1. Cross-thread producer for a scalar-read CB

The wait/pop pair is inserted only when a transfer fills the CB **in the same
datamovement region**. Waiting on a CB nothing pushes locally would hang, so a CB
filled by another thread and scalar-read here is left unsynchronized.

That combination cannot currently arise: the affinity grouping in
`d2m-schedule-dma` puts a scalar read's CB and its producing transfer on the same
thread by construction, since they share a port. It would become reachable if a
CB were ever produced by a remote core (multicast into an index block, say). The
fix is to insert the pair anyway when a cross-thread producer can be proven, which
needs the CB accounting to name which thread pushes.

### 2. Page granularity for integer CBs

`cb_wait_front`/`cb_reserve_back` counts are in pages, and for an integer CB the
whole buffer is one page. This is *self-consistent* rather than a hazard: every
side derives its count from the same `getMemrefCBNumPages` call on the same memref
type, so reserve/push/wait/pop cannot drift apart. It does mean an index block is
acquired and released whole, so double-buffering it or consuming it as it arrives
would need the page size for non-tile CBs pinned down first.

### 3. TTIR normalizes integer tensors to `si32`, which the scalar path rejects

Compiling a program with an `i32` input through the real frontend produces
`memref<...xsi32>` buffers, and `si32` is not a signless integer, so a scalar load
of one is rejected (see the element-type restriction above). This is not a defect
in the lowering -- `arith` is signless-only, so an `si32` value genuinely cannot
feed address arithmetic -- but it does mean a frontend emitting a dependent load
has to produce signless `i32` index buffers. The silicon fixture declares its
index input signless for exactly this reason.

Resolving it properly is a frontend/TTIR question: either integer index tensors
keep signless element types through normalization, or the dependent-load lowering
gets a sanctioned way to reinterpret the loaded bits as signless.

### 4. Cosmetic: `tt_l1_ptr` leaks onto value declarations

`TTKernelLoadFromL1OpToEmitCOpRewriter` reuses the pointee's opaque type name
for the loaded value, giving `tt_l1_ptr uint32_t v17 = v14[v16];`. `tt_l1_ptr`
is `__attribute__((rvtt_l1_ptr))`, which is meaningless on a value. Pre-existing
(the `indexed_row_copy` path emits the same), so not a blocker.

## Question 2: slices of a block

Dependent *slices* are a harder problem than dependent scalars, and the answer
is different: don't try to express them as tile-granular D2M accesses.

`d2m.indexed_row_copy` is the existing precedent for breaking the tile
abstraction, and it is instructive. It gathers rows from a table using an i32
index buffer, and it works on **row-major (untilized) memrefs**, not tiles:
`memref<1x1x8x16xf32>`, not `memref<...x!ttcore.tile<32x32, f32>>`. It stages
each index through a scratch CB (`loadI32FromNocPacketThroughScratch`) and then
issues one NoC read per row into a second scratch CB. `d2m.embedding` is the
tensor-level op that bufferizes to it.

Its constraints show where the real limits are, and they are NoC alignment
limits rather than tile limits:

- Transfer sizes are quantized to `noc_l1_address_align_bytes` (16 on WH) or
  `noc_dram_address_align_bytes` (32), via `getNocAddressAlignmentBytes`.
- Shard widths must be a multiple of the resulting elements-per-transfer, or
  the pattern bails: *"indexed row copy requires source and destination shard
  widths to be NoC transfer aligned"*.
- Indices in L1; source and destination in L1 or DRAM; collapsed-2D sharded
  memrefs; 2D grid.

So: a slice that does not line up with tile rows is not a problem *for the NoC*
— it is a problem for whatever consumes the result. The workable shape is

1. gather at row/element granularity into an untilized L1 scratch buffer, using
   dependent scalar loads (this note) for the bounds, then
2. tilize that buffer for the compute region.

Both halves exist. The gap is a `d2m` op that generalizes
`indexed_row_copy` from "rows selected by an index tensor" to "a contiguous
element range `[start, end)` with dynamic bounds", plus the alignment
head/tail handling — a misaligned `start` needs the leading partial transfer
split off, which `indexed_row_copy` avoids by requiring alignment up front.

That is a D2M extension, not a reason to mix in a lower-level dialect. The
lower-level dialect is already reachable: `ttkernel` has the pointer ops, and
`memref`/`arith`/`scf` survive into the datamovement thread. D2M's tile
contract binds the *compute* region, and dependent addressing belongs on the
data movement side, where it can be as scalar as it needs to be.
