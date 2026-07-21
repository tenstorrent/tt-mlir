# Refactor the L1 spill pass to fully leverage captured allocator state

## Problem

The greedy L1 spill pass runs two memory models at once. `SumL1MemoryTracker`
simulates a top-down first-fit allocator (a `freeList`, per-tensor addresses,
alias groups) to approximate fit / fragmentation / CB-overlap. The newer
`MockAllocatorL1Tracker` feeds the currently-live L1 buffers as allocation
records into tt-metal's *real* stateful op-model query, which models placement
and fragmentation accurately. Today the mock tracker **inherits** the sum
tracker and reuses its simulated addresses for eviction ordering and CB-overlap
checks — so the pass reasons about two allocators at once. That dual model is
the direct source of the #9064 CB-vs-L1 clash friction: the simulated address
and the real fit decision diverge.

Goal: on the stateful path, let the captured allocator state answer fit,
fragmentation, placement, and CB-clash entirely. Remove the address *simulator*
and all explicit fragmentation/CB reasoning from the spill pass. Keep the
`SumL1MemoryTracker` path intact as a legacy fallback behind
`use-mock-allocator-state=false`.

## Key constraint discovered

`tt::tt_metal::experimental::MockAllocatorState` exposes only:
- `with_allocations(vector<AllocationRecord>) const` — reconstructs a state with
  buffers pinned at the **explicit addresses in the records** (it does not
  re-flow / reassign),
- `extract_mock_allocator_state` / `override_mock_allocator_state`,
- read-only `total_allocated_size`, `lowest_occupied(BufferType)`.

There is **no** `allocate(size)` primitive. The allocator only ever *assigns* a
fresh address inside an op query (`output_allocations`).

Consequence: the records we hold for still-live tensors were assigned by the
allocator **with the eventual victim present in the history**. Evicting a tensor
allocated at some earlier position rewrites history — in the victim-free
counterfactual, every buffer allocated after it would land at a different
address. Because `with_allocations` pins addresses verbatim, we cannot fake this
by dropping the victim's record; that produces a state (survivors pinned high, a
phantom hole where the victim was) the real allocator never produces. **Eviction
must rebuild the surviving live set's records under the victim-free history.**

## Chosen approach

### Structure: shared base class + two subclasses

Extract `L1SpillManagementBase<MemoryTracker>` holding everything the two paths
share; the current class becomes the Sum subclass; a new small subclass drives
the stateful path.

```
L1SpillManagementBase<MemoryTracker>
  shared state:  func, deviceGrid, l1Budget, observer_, memoryTracker,
                 liveSet (FLU max-heap), liveValues, ScheduleData
  shared infra:  buildScheduleData, computeLastUsePositions, processDeadTensors,
                 evictFarthestUse, evictAllFromL1, spillToDram(+Impl/BeforeTrigger),
                 demoteToDram, insertReshardForConsumer, insertReshardIntoSchedule,
                 revalidateConsumers, collectDownstreamConsumers, applyOutputConfig,
                 extractOpConfigFromIR;
                 the run() forward-sweep skeleton;
                 the event-log framework: l1EventLog, campaignMin, allocEventIndex,
                 insertEventIntoLog, the checkpoint map, the replayFrom DRIVER,
                 and the evict -> replay -> recheck loop.
  hooks:         placeValidatedOutput(op,pos,data,result,tensorResults)
                 recoverFromOOM(op,pos,tensorResults,data)
                 commitAllocation(val,size)          // add one result to live state
                 replayAllocEvent / replayDeallocEvent // per-event rebuild action
                 stillFits()                          // per-alloc fit predicate

  SumL1SpillManagement : the address simulator (freeList, tensorAddresses,
     wouldAllocateAt, getLowestOccupiedAddress, alias groups) plus the explicit
     fit/frag/CB methods (ensureFitsL1, handleNoFit, handleFragmentation,
     wouldCBsOverlapTensors, evictForCBOverlap, evictForDramCBGrowth). Its hooks
     reproduce today's exact behavior -> use-mock-allocator-state=false is
     byte-for-byte identical.

  StatefulL1SpillManagement : NO address simulator, NO explicit fit/frag/CB.
     placeValidatedOutput = just commit the output (no fit check).
     recoverFromOOM       = evict FLU victim -> spill+reshard -> mark skipped
                            -> replayFrom(campaignMin) -> re-query current op; loop.
     replay action        = re-run the stateful op-model query for the surviving
                            op, take output_allocations as its fresh record.
     stillFits()          = the re-query returned success (not OOM).
```

The `run()` skeleton (dead-tensor processing, `ToLayoutOp`/sink/`NotImplemented`
handling, `validate()`, then success / backend-error / OOM branch dispatch)
lives once in the base and calls the hooks. Both paths log allocation order into
the shared event log; only the per-event replay *action* and the checkpoint
payload type differ.

### The slimmed stateful tracker

`MockAllocatorL1Tracker` stops inheriting `SumL1MemoryTracker` and holds only:

```
struct MockAllocatorL1Tracker {
  DenseMap<Value, RecordVec> liveRecords;              // the real live-set state
  mutable DenseMap<Operation*, RecordVec> pendingRecords; // const validate() -> addTensor handoff
  DenseMap<Value, Value>    aliasOf;                   // viewOutput -> canonical buffer owner
  DenseMap<Value, unsigned> aliasRefcount;             // owner -> # live aliasers

  ValidationResult validate(op, inputs, config) const; // flatten liveRecords (dedupe aliases) -> stateful query
  void addTensor(Value, size);                          // liveRecords[v] = pendingRecords[op][idx]
  void addTensorAlias(Value out, Value src);            // alias, bump refcount, no new record
  void removeTensor(Value);                             // drop record / decrement refcount
  uint64_t getOccupiedL1() const;                       // sum of live record sizes (logging/observer)

  struct Snapshot { DenseMap<Value,RecordVec> liveRecords; /* + alias maps */ };
  Snapshot takeSnapshot() const;  void restoreSnapshot(const Snapshot&);
};
```

Gone from the tracker: `freeList`, `tensorAddresses`, `aliasGroups`,
`wouldAllocateAt`, `getLowestOccupiedAddress`, `allocateAddress`, the first-fit
simulator. The `Snapshot` payload is a `RecordVec` set (a replay checkpoint), not
address-sim state.

`pendingRecords` stays: `validate()` is `const` and called speculatively many
times before commit, so per-output records are stashed per-op and moved to
`liveRecords[result]` at `addTensor` (the commit point), keyed positionally by
`result.getResultNumber()`.

**Aliasing** is record-level, not address-level: a view op's output shares its
source's buffer, so we never emit a second record for it, dedupe by canonical
owner when flattening `liveRecords` into the query state, and keep the buffer
live until the last aliaser dies (refcount). `isAliasingViewOp` (#9054) drives
`addTensorAlias` in place of the old `addTensorAtAddress`.

### Eviction: rebuild-by-replay

When `validate()` returns OOM, `recoverFromOOM` loops:

```
loop:
  victim = evictFarthestUse()          // FLU policy, shared
  if no victim: demoteToDram(op); return  // self-demote as last resort
  evictValue(victim, ...)              // spill victim to DRAM + reshard past consumers (shared IR)
  mark victim's alloc event skipped; fold its index into campaignMin
  replayFrom(campaignMin)              // rebuild surviving records under victim-free history
  result = tracker.validate(op, ...)   // re-query with the rebuilt live set
  if result.isSuccess():
    placeValidatedOutput(...); return
  // else still OOM (raw size OR fragmentation, per the query) -> loop
```

`replayFrom(campaignMin)` (stateful override) starts from the checkpoint
captured just before `campaignMin` and walks `[campaignMin, end)`:
- **kAlloc, not skipped, not alias**: re-run the stateful op-model query for that
  op (input layouts + config from *current* IR, so spills are reflected), take
  `output_allocations` as the value's fresh record, add to the accumulated set.
- **kAlloc, alias**: alias the source's record (no query).
- **kAlloc, skipped**: skip (evicted).
- **kDealloc**: drop the value's records from the accumulated set.

After the walk, the accumulated set is the current live set under the victim-free
history. Termination: each iteration removes one tensor from a finite live set,
ending at self-demote.

Correctness of the tricky cases falls out of ordered replay:
- **Reshard for a past consumer** (position >= `campaignMin`) gets its own
  alloc+dealloc events, so it is re-queried into the accumulated set during its
  live window and dropped before the current op — transient L1 pressure is
  accounted for, but it does not appear in the current op's live set.
- **Multi-result** ops use real per-result records instead of the even-split
  approximation.
- **View ops** replay via alias, matching the happy path.

## Edge cases

- **`ToLayoutOp` with L1 output** (pre-decomposition, unqueryable): the Sum path
  runs a `cbPeakUsage=0` fit check. The stateful path has no `ensureFitsL1`; the
  op stays out of `liveValues` exactly as today, and any real OOM surfaces at the
  *next* op's stateful query (which sees the L1 input). Pinned down by a
  no-regression test rather than assumed.
- **`NotImplemented`** ops -> `evictAllFromL1` (shared, unchanged).
- **Non-OOM backend error** -> `demoteToDram` (shared, unchanged). The #9064
  CB-clash classification (clash -> OOM) is unchanged and now flows through the
  generic evict -> replay -> re-query loop.

## Known cost

Happy path (no eviction) = one query per op, unchanged. Op-query replay fires
only inside eviction campaigns, bounded to `[campaignMin, pos]` by the
checkpoint. Op-model queries are the optimizer's dominant cost, so
eviction-heavy funcs pay more than the Sum path did. The replay action is behind
a hook, so a future cheap `allocate(size)/free` primitive on `MockAllocatorState`
could replace op-query replay with buffer-level replay with no pass-side change.
(Ships-today path is op-query replay, per decision.)

## Testing

Extend `test/unittests/Optimizer/TestL1SpillManagementMockAllocator.cpp`
(mock-device, HW-free):
1. eviction-with-replay: post-eviction live set re-queries clean.
2. reshard-past-consumer during a stateful eviction.
3. multi-result op -> per-result records.
4. view-alias tripwires still hold (query returns aliased/zero address for view
   ops; our `isAliasingViewOp` keeps them off a fresh record).
5. `ToLayoutOp`-L1 no-regression.
6. Sum path unchanged: existing `L1SpillManagementTests` stay green (byte-identical
   behavior for `use-mock-allocator-state=false`).

Plus: TTNN optimizer lit, and n150 + p150 `Performance Benchmark` sweeps as the
integration gate before marking ready.

## Non-goals

- Retiring the `SumL1MemoryTracker` path (kept as legacy fallback).
- Adding the cheap `allocate/free` tt-metal primitive (future optimization; the
  hook boundary is left in place for it).
- Changing beam-search, which keeps its cached stateless query.
