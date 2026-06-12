# Spec: `d2m.core_read` / `d2m.core_write` — core↔core L1 data movement

## Motivation

Fusing a multi-core compute (e.g. an 8×8 matmul) with a CCL collective needs one
core to move data **produced by other cores** into its own L1 — without going
through the device-tensor (grid/shard) layout. Today the only cross-core data
movement is `remote_load`/`remote_store`, which require a **device layout** on
the addressed tensor. That forces the produced data into a device-laid-out
tensor, and a tensor used as both a per-core write target *and* a cross-core read
source within one generic loses its device layout at the store site and is
rejected (`'d2m.remote_store' op memref/tensor must be remote (have a device
layout)`). See `fused_matmul_allgather_8x8_design.md` "Probe results".

`core_read`/`core_write` are the missing primitive: **direct core→core L1
transfers with no device-layout requirement**, mapping straight to `dma_read`/
`dma_write`. They unblock two things the device-tensor path can't:

1. **No device-layout wall.** The matmul output lives in a plain local L1
   scratch buffer; `core_read` pulls it cross-core. The only device-layout
   tensor left is the fabric `remote_store` target (write-only — no dual-use).
2. **Streaming (no L1 wall).** A core can pull one tile, forward it over fabric,
   and reuse the buffer — O(one tile) resident, not O(whole shard). This is the
   pull-and-forward stream the atomic gather couldn't express.

PR #8531 already added the underlying mechanism: a `dma_read` mode whose NoC
source is another core (`srcCore`), with "no device-layout requirement on the
source buffer". This spec exposes that (and its write dual) as first-class ops
instead of burying it inside the `gather_core` collective. `gather_core` is
explicitly **out of scope** here.

## Ops

Both are `D2M_GenericRegionDatamovementOp`s — peers of `remote_load`/
`remote_store`, sitting one level above `dma_read`/`dma_write`.

### `d2m.core_read`

Read an entire local buffer from another core in the grid into this core's local
buffer. Pure unicast L1→L1 over the NoC.

```mlir
%tx = d2m.core_read %src core [%srcY, %srcX] into %dst
    : (memref<MxN x tile, #l1>, memref<MxN x tile, #l1>) -> !d2m.mem_tx
```

- `%src` — local L1 buffer (or CB), **no device layout**. The remote core's copy
  at the *same L1 offset* is the data read (see Invariant).
- `core [%srcY, %srcX]` — logical grid coordinates of the source core.
- `%dst` — this core's local L1 buffer (or CB) the data lands in.
- result — `!d2m.mem_tx<read>`, waited on with `d2m.dma_wait` like any DMA.

CB / mixed surface forms mirror `gather_core` / `remote_load` (`from %srcCb`,
`into %dstCb`) so the op can sit on either side of the DM↔compute boundary.

### `d2m.core_write`

Symmetric: write this core's local buffer into another core's local buffer.

```mlir
%tx = d2m.core_write %src into %dst core [%dstY, %dstX]
    : (memref<MxN x tile, #l1>, memref<MxN x tile, #l1>) -> !d2m.mem_tx
```

- `%src` — this core's local L1 buffer (read).
- `%dst` — local L1 buffer; the copy on core `[%dstY, %dstX]` (same L1 offset) is
  the write target.
- result — `!d2m.mem_tx<write>`.

`core_write` is the push dual of `core_read`. The matmul→all_gather pattern
below only needs `core_read` (the fabric core pulls); `core_write` is specced
for completeness and for push-style collectives. **Implement `core_read`
first** — PR #8531 already added the `dma_read` source-core mode; the
`dma_write` dest-core mode for `core_write` still needs adding.

## Operands / traits (tablegen sketch)

```
def D2M_CoreReadOp : D2M_GenericRegionDatamovementOp<"core_read",
  [ AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>,
    DestinationStyleOpInterface, DeclareOpInterfaceMethods<BufferizableOpInterface, [...]>,
    D2M_SynchronizableOpInterface ]> {
  let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcCore,
                       AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                       I64Attr:$numElems);
  let results = (outs Optional<AnyRankedTensorOrMemRef>:$result);   // tensor (DPS) form
  let hasVerifier = 1;
}
```

- Not a `ShardDMAOpInterface` (no CB-port/shard semantics) — distinct from
  `remote_load`. It is `Synchronizable` (carries the DM/compute handshake) and
  `DestinationStyle` (tensor form aliases `%dst`, like `remote_load`).
- `numElems`/`dstIndices`: support the shard-level (whole buffer, `numElems==0`)
  and fully-indexed forms, matching `dma_read`, so `LowerDMAToFullyIndexedForm`
  can expand it.

## The uniform-L1-offset invariant

`core_read`/`core_write` identify the remote data by **(srcCore, the buffer's
own L1 offset)** — they assume `%src`/`%dst` sit at the *same L1 address on every
core in the grid*. This is the same invariant `gather_core` relies on. The
allocator (`D2MAllocate`) must place these buffers identically across cores;
buffers that can't satisfy it (e.g. per-core-variable allocations) are illegal
operands — enforce in the verifier where detectable, document otherwise.

## Lowering

- `core_read` → `dma_read` with the cross-core source mode from PR #8531: same
  L1 layout as local mode (no grid component, uniform offset), NoC source =
  `srcCore`. No device-layout requirement on `%src`. Reuse #8531's
  `LowerLoadStoreOpsToDMA` / `LowerDMAToFullyIndexedForm` additions as-is where
  possible.
- `core_write` → `dma_write` with a symmetric dest-core mode (**to add**: #8531
  only did the read side).
- `srcCore`/`dstCore` are logical grid coords; the existing
  `mapVirtualToPhysicalCoreIndex` (moved to D2M Utils by #8531) converts to
  physical NoC coords at `GenericRegionsToFuncs`.

## Thread-split / scheduling behavior

These are datamovement ops, so the work already landed this session applies
directly:

- **DM-resident**: classified onto the datamovement thread by
  `split-unified-thread-v2` (`collectOpsToErase`) and kept on a DM thread by
  `ScheduleDMA` — same bucket as `ShardDMAOpInterface` / `device_synchronize`.
  (They are not `ShardDMAOpInterface`, so add them to the DM-resident `isa<>`
  lists alongside the existing DMA ops.)
- **scf.if-gatable**: the passes now recurse into `scf.if` (committed
  `8fa5c4d71a`), so `if core == collector { ...core_read...; remote_store }`
  lowers onto the DM thread inside the guard.
- **collectDMAOps / filterOpsForThread**: decide whether `core_read`/`write`
  participate in the NoC-processor split. Simplest first cut: treat a kernel
  containing them like the explicit-semaphore case (single DM thread, Option D)
  to avoid cross-thread duplication until the streaming sync is proven.

## Synchronization contract

`core_read`/`core_write` are **pure data movement** — they carry no cross-core
ordering of their own (unlike `gather_core`, which bundled a barrier). The caller
establishes readiness and buffer-reuse ordering with explicit semaphores (now
supported on the DM thread). Two edges to manage:

1. **Producer-ready**: the source core must have written its buffer before the
   reader `core_read`s it. Producer `semaphore_inc`s a readiness sem on the
   reader; reader `semaphore_wait`s.
2. **Buffer-reuse (back-pressure)**: if the source core will overwrite its buffer
   (next loop iteration), it must wait until the reader has consumed it. Reader
   `semaphore_inc`s a "consumed" sem back; producer waits before reusing.

(These are exactly `gather_core`'s `sourceReady`/`collectorDone` — now explicit
and per-tile, which is what enables streaming.)

### Streaming matmul → all_gather (the target pattern)

```python
@d2m.kernel
def matmul_all_gather(lhs, rhs, out, ready, consumed, start_sem, end_sem):
    cy, cx = core_index(0), core_index(1)
    c = matmul(remote_load(lhs, [cy, cx]), remote_load(rhs, [cy, cx]))  # -> local scratch
    # publish my tile, signal the collector
    semaphore_inc(ready, 1, core=[0, 0])
    if cy == 0 and cx == 0:                       # the single fabric/injector core
        device_synchronize(start_sem, ...)
        for j in range(NUM_CORES):                # stream: pull one tile, forward it
            semaphore_wait(ready, j + 1)          # core j's tile is ready
            t = core_read(scratch, core=[0, j])  # pull core j's tile (no device layout)
            remote_store(out, [dx, j], t, start_device=..., semaphore=end_sem, ...)  # fabric forward
        semaphore_wait(end_sem, num_devices)
    else:
        semaphore_wait(consumed, 1)               # back-pressure if reusing scratch
```

No dual-use device-layout tensor anywhere: matmul output in local scratch
(no layout) → `core_read` (no layout) → fabric `remote_store` (peer target is a
write-only device tensor). O(one tile) resident on the injector core.

## DSL surface (d2m-jit)

```python
def core_read(dst, src, *, core):          # core = [y, x] logical source coords
def core_write(src, dst, *, core):         # core = [y, x] logical dest coords
```

Mirror `remote_load`/`remote_store` in `api.py` (allocate-dst and explicit-dst
forms); they emit `d2m.core_read`/`d2m.core_write`.

## Verifier rules

- `%src`/`%dst` must be local (no device layout) L1 memrefs/CBs.
- `srcCore`/`dstCore` rank == grid rank (2 for the current grids).
- same element type, same shape on src/dst.
- (best-effort) flag operands that can't meet the uniform-L1-offset invariant.

## Open questions

- **dma_write dest-core mode** for `core_write` — confirm scope of adding it
  (PR #8531 added read only).
- **Allocator guarantee** of identical L1 offsets for these buffers — does
  `D2MAllocate` already place in-kernel scratch / CBs uniformly across the grid,
  or does it need a constraint?
- **Streaming sync correctness** — the two-semaphore (ready/consumed) handshake
  with double-buffering on the injector is easy to get subtly wrong; needs a
  dedicated test (deadlock/hang is the failure mode).
- **NoC processor split** — whether a kernel with `core_read` + a fabric
  `remote_store` should stay single-DM-thread (Option D) or can use both NoCs.

## Out of scope

`gather_core` (the atomic collective). `core_read`/`core_write` are the
lower-level primitives; an atomic gather can be re-expressed as a fixed
`core_read` loop + a wait-for-all barrier if ever wanted, but it is not part of
this spec.
