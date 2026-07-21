# Stateful L1 Spill — Fully Leverage Captured State — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** On the stateful spill path, let the captured tt-metal allocator state
answer fit/fragmentation/placement/CB-clash entirely; delete the address
*simulator* and explicit fit/frag/CB reasoning; keep the `SumL1MemoryTracker`
path byte-identical behind `use-mock-allocator-state=false`.

**Architecture:** Extract `L1SpillManagementBase<MemoryTracker>` (shared sweep,
IR-mutation, FLU, event-log framework, replay *driver*) with virtual hooks; two
subclasses — `SumL1SpillManagement` (keeps the address simulator) and
`StatefulL1SpillManagement` (op-query replay, no addresses). Slim
`MockAllocatorL1Tracker` so it no longer inherits `SumL1MemoryTracker`.

**Tech Stack:** C++17, MLIR/LLVM, tt-metalium mock-allocator experimental API,
GoogleTest unit tests, CMake/Ninja.

## Global Constraints

- Build: `source env/activate && cmake --build build -- -j$(nproc)` (OpModel ON).
- `cmake --build build` does NOT rebuild unittests — always rebuild the test
  target explicitly (`--target L1SpillManagementTests`,
  `--target L1SpillManagementMockAllocatorTests`, `--target ValidationTests`).
- Do NOT pipe `source env/activate` (subshell loses exports).
- LLVM code style; follow the surrounding file.
- The `use-mock-allocator-state=false` (Sum) path MUST stay behavior-identical;
  existing `L1SpillManagementTests` are the guard.
- Mock-device unit tests are HW-free (open/close mock device in fixture SetUp).

---

## File structure

- `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h` — declares the two
  trackers, `L1SpillManagementBase<T>` + virtual hooks, and the two subclasses.
- `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp` — definitions; base methods,
  Sum subclass (address sim + fit/frag/CB), Stateful subclass (op-query replay).
- `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp` —
  pass driver: instantiate `StatefulL1SpillManagement` / `SumL1SpillManagement`.
- `test/unittests/Optimizer/TestL1SpillManagementMockAllocator.cpp` — new
  stateful eviction/replay tests.

---

### Task 1: Add virtual hook seams to the current class (no behavior change)

Introduce the three primary hooks as `virtual` methods on the *current*
`L1SpillManagement<MemoryTracker>` template, with bodies that call today's exact
code. This is a pure seam: both instantiations still behave identically because
the default hook bodies ARE the current inline logic lifted out of `run()`.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`
- Modify: `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

**Interfaces produced:**
- `virtual uint64_t placeValidatedOutput(Operation *op, int64_t pos, ScheduleData &data, const op_constraint_validation::ValidationResult &result)` — returns L1 size to add (0 = demoted); default body = today's `ensureFitsL1(op,pos,data,result.cbPeakUsage,result.outputL1Usage)`.
- `virtual void recoverFromOOM(Operation *op, int64_t pos, llvm::ArrayRef<OpResult> tensorResults, ScheduleData &data, std::function<void(uint64_t)> addResultsToLiveSet)` — default body = today's `handleOOM(...)` (rename the existing body into this hook; keep `handleOOM` deleted or as the Sum override in Task 3).
- `virtual void commitAllocation(Value val, uint64_t perResultL1, ScheduleData &data)` — default body = today's per-result block inside `addResultsToLiveSet` (event-log push + snapshot + `willAliasSourceInL1`?addTensorAtAddress:addTensor + liveSet/liveValues insert).

- [ ] **Step 1:** In `run()`, replace the inline `ensureFitsL1(...)` success-branch call with `placeValidatedOutput(op, pos, data, result)`; replace the `handleOOM(...)` call with `recoverFromOOM(...)`; rewrite the `addResultsToLiveSet` lambda body to loop calling `commitAllocation(val, perResultL1, data)`.
- [ ] **Step 2:** Move the current `ensureFitsL1` success-path body into `placeValidatedOutput` (it calls `ensureFitsL1` internally — leave `ensureFitsL1` in place for now). Move the `handleOOM` body into `recoverFromOOM` (leave a thin `handleOOM` that forwards, or inline). Move the per-result commit block into `commitAllocation`.
- [ ] **Step 3: Build.** Run `cmake --build build -- -j$(nproc)`. Expected: 0 errors.
- [ ] **Step 4: Guard tests.** `cmake --build build --target L1SpillManagementTests L1SpillManagementMockAllocatorTests && ./build/test/unittests/Optimizer/L1SpillManagementTests && ./build/test/unittests/Optimizer/L1SpillManagementMockAllocatorTests`. Expected: all PASS (behavior unchanged — this is a pure seam).
- [ ] **Step 5: Commit.** `git add -A && git commit -m "[optimizer] Introduce spill hook seams (no behavior change)"`

---

### Task 2: Rename to `L1SpillManagementBase`, add two subclasses (still both address-sim)

Rename the template class to `L1SpillManagementBase<MemoryTracker>`. Create
`SumL1SpillManagement : L1SpillManagementBase<SumL1MemoryTracker>` and
`MockAllocatorSpillManagement : L1SpillManagementBase<MockAllocatorL1Tracker>`,
both initially inheriting the default hooks (still address-sim; Mock tracker
still inherits Sum tracker). Update explicit instantiations + pass driver. No
behavior change yet — this only splits the type so Task 3/4 can diverge hooks.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`
- Modify: `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`
- Modify: `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp`

**Interfaces produced:**
- `template <typename MemoryTracker> class L1SpillManagementBase { ... };`
- `class SumL1SpillManagement final : public L1SpillManagementBase<SumL1MemoryTracker> { using L1SpillManagementBase::L1SpillManagementBase; };`
- `class MockAllocatorSpillManagement final : public L1SpillManagementBase<MockAllocatorL1Tracker> { using L1SpillManagementBase::L1SpillManagementBase; };`

- [ ] **Step 1:** Rename `L1SpillManagement` → `L1SpillManagementBase` throughout `.h`/`.cpp` (class, ctor, all `L1SpillManagement<MemoryTracker>::` qualifiers). Make hooks and `run()`/`hasFailed()`/`getObserver()`/`getMemoryTracker()` accessible to subclasses (protected where needed).
- [ ] **Step 2:** Add the two `final` subclasses in the `.h` inheriting the ctor. Replace the two `extern template class` lines with `extern template class L1SpillManagementBase<SumL1MemoryTracker>;` and `extern template class L1SpillManagementBase<MockAllocatorL1Tracker>;`. In `.cpp` update the explicit instantiations identically.
- [ ] **Step 3:** In `GreedyL1SpillManagement.cpp`, change `L1SpillManagement<MockAllocatorL1Tracker>` → `MockAllocatorSpillManagement` and `L1SpillManagement<SumL1MemoryTracker>` → `SumL1SpillManagement`.
- [ ] **Step 4: Build.** `cmake --build build -- -j$(nproc)`. Expected: 0 errors.
- [ ] **Step 5: Guard tests.** Same two test binaries as Task 1 Step 4. Expected: all PASS.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "[optimizer] Split spill manager into base + Sum/Mock subclasses"`

---

### Task 3: Move address-sim + fit/frag/CB into `SumL1SpillManagement`

Move the address-simulation-specific members and methods off the base into
`SumL1SpillManagement`, so the base no longer references the address simulator.
The base keeps: sweep, IR mutation, FLU, event log framework
(`l1EventLog`/`campaignMin`/`allocEventIndex`/`insertEventIntoLog`/checkpoint
map), and the `replayFrom`/`markEvictedAndRebuild`/`evictValue`/`evictUntil`
*drivers* — but the per-event alloc/dealloc action and the "fits?" check become
`virtual` hooks. Sum overrides them with the address-sim logic (moved verbatim).

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`
- Modify: `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`

**Interfaces produced (new virtual hooks on base, Sum overrides):**
- `virtual bool replayFrom(size_t startIdx)` — base default asserts (must override); Sum override = current `replayFrom` body.
- `virtual uint64_t placeValidatedOutput(...)` — Sum override = current `ensureFitsL1`-based body.
- `virtual void recoverFromOOM(...)` — Sum override = current `handleOOM` body.
- `virtual void commitAllocation(Value, uint64_t, ScheduleData &)` — Sum override = current block using `willAliasSourceInL1`/`addTensorAtAddress`/`addTensor` + snapshot push.

- [ ] **Step 1:** Move these methods' declarations+definitions from base to `SumL1SpillManagement`: `ensureFitsL1`, `handleNoFit`, `handleFragmentation`, `wouldCBsOverlapTensors`, `evictForCBOverlap`, `evictForDramCBGrowth`, `willAliasSourceInL1`, and `replayFrom` (as the override). Keep `cbFragCushion` on Sum only. These use `SumL1MemoryTracker` address APIs (`wouldAllocateAt`, `allocateAddress`, `hasTensorAddress`), which are valid because `MockAllocatorL1Tracker` still inherits them for now.
- [ ] **Step 2:** In the base, make `replayFrom`, `placeValidatedOutput`, `recoverFromOOM`, `commitAllocation` pure-virtual (`= 0`) or virtual-with-assert. `evictValue`/`markEvictedAndRebuild`/`evictUntil` stay on the base and call `replayFrom(...)` through the vtable.
- [ ] **Step 3:** Give `SumL1SpillManagement` overrides of the four hooks whose bodies are the code moved/lifted in Task 1–2 (address-sim behavior). Verify `commitAllocation`'s event-log push (`allocEventIndex`/`addressSnapshots`/`l1EventLog`) stays in the base (shared) and only the tracker call (`addTensor`/`addTensorAtAddress`) is in the override — OR keep the whole block in Sum's override and have Stateful's override write its own event-log push. Choose: **event-log push stays in base `commitAllocation` non-virtual prologue; the tracker-add is the virtual part** (`virtual void trackAllocation(Value,uint64_t)`), so both paths share event logging. Adjust interfaces accordingly.
- [ ] **Step 4: Build.** Expected: 0 errors. (`MockAllocatorSpillManagement` still uses Sum's overrides? No — subclasses don't share overrides. Temporarily give `MockAllocatorSpillManagement` the SAME overrides by having both subclasses inherit an intermediate `AddressSimSpillManagement<T>` that holds the address-sim overrides. Mock stays on address sim until Task 5.)
- [ ] **Step 5: Guard tests.** Both binaries PASS (behavior unchanged for both paths).
- [ ] **Step 6: Commit.** `git add -A && git commit -m "[optimizer] Move address-sim + fit/frag/CB into Sum path; base is address-agnostic"`

---

### Task 4: Slim `MockAllocatorL1Tracker` (stop inheriting `SumL1MemoryTracker`)

Rewrite the tracker to hold only `liveRecords` + `pendingRecords` + record-level
alias maps + a `RecordVec`-set `Snapshot`. Remove address-sim inheritance.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`
- Modify: `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`
- Test: `test/unittests/Optimizer/TestL1SpillManagementMockAllocator.cpp`

**Interfaces produced:**
```cpp
struct MockAllocatorL1Tracker {
  using RecordVec = llvm::SmallVector<op_model::OpModelAllocationRecord>;
  llvm::DenseMap<Value, RecordVec> liveRecords;
  mutable llvm::DenseMap<Operation *, RecordVec> pendingRecords;
  llvm::DenseMap<Value, Value> aliasOf;          // aliaser -> canonical owner
  llvm::DenseMap<Value, unsigned> aliasRefcount; // owner  -> # live aliasers (incl self)

  SumL1MemoryTracker::BackendValidatorFn backendValidator; // keep test hook shape
  void init(uint64_t l1BudgetPerCore);
  op_constraint_validation::ValidationResult
  validate(Operation *, llvm::ArrayRef<TTNNLayoutAttr>, const OpConfig &) const;
  void addTensor(Value result, uint64_t l1SizePerCore);
  void addTensorAlias(Value out, Value src);
  void removeTensor(Value result);
  void removeTensorFromSizes(Value result); // == removeTensor for records
  bool hasTensor(Value result) const;
  uint64_t getTensorSize(Value result) const; // sum of this value's record sizes
  uint64_t getOccupiedL1() const;             // sum over liveRecords (dedup aliases)
  struct Snapshot { llvm::DenseMap<Value, RecordVec> liveRecords;
                    llvm::DenseMap<Value, Value> aliasOf;
                    llvm::DenseMap<Value, unsigned> aliasRefcount; };
  Snapshot takeSnapshot() const; void restoreSnapshot(const Snapshot &);
};
```
- `validate()` flattens `liveRecords` into a single `RecordVec` (dedupe by canonical owner via `aliasOf`), calls the stateful path (backendValidator hook when set, else `op_constraint_validation::validateOperation(op, inputs, config, flatRecords, /*additionalL1=*/0)`), stashes `result.outputAllocations` into `pendingRecords[op]`.
- `addTensor` moves `pendingRecords[op][result.getResultNumber()]` → `liveRecords[result]`; sets `aliasRefcount[result]=1`, `aliasOf[result]=result`.
- `addTensorAlias(out, src)`: `Value owner = aliasOf.lookup(src)` (fallback `src`); `aliasOf[out]=owner`; `++aliasRefcount[owner]`; NO record for `out`.
- `removeTensor(v)`: `Value owner=aliasOf[v]; if(--aliasRefcount[owner]==0){ liveRecords.erase(owner); aliasRefcount.erase(owner);} aliasOf.erase(v);` (if `v==owner` but refcount>0, keep the record under owner — handle owner death while aliasers live by leaving the record under `owner` key; since owner==v only when it is its own alias, and refcount tracks all, erasing the record only at 0 is correct as long as the record is stored under `owner`).

- [ ] **Step 1: Write failing test** in `TestL1SpillManagementMockAllocator.cpp`:
```cpp
TEST_F(L1SpillMockAllocatorFixture, TrackerAliasRefcountKeepsBufferLive) {
  MockAllocatorL1Tracker t;
  t.init(kL1Budget);
  // owner + one alias; buffer stays live until both die.
  Value owner = /* mkValue(...) */; Value view = /* mkValue(...) */;
  t.pendingRecords[owner.getDefiningOp()] = {rec(/*L1*/, 0, 2048)};
  t.addTensor(owner, 2048);
  t.addTensorAlias(view, owner);
  EXPECT_EQ(t.getOccupiedL1(), 2048u);   // deduped, one buffer
  t.removeTensor(owner);
  EXPECT_EQ(t.getOccupiedL1(), 2048u);   // alias still holds it
  t.removeTensor(view);
  EXPECT_EQ(t.getOccupiedL1(), 0u);
}
```
- [ ] **Step 2: Run, expect FAIL to compile** (`addTensorAlias`/new shape not present).
- [ ] **Step 3:** Rewrite `MockAllocatorL1Tracker` to the interface above; delete its inheritance of `SumL1MemoryTracker` and the old `addTensorAtAddress`/base `Snapshot`.
- [ ] **Step 4:** Build the tracker changes only enough to compile the test file will fail elsewhere (base still references address APIs on the Mock tracker via the intermediate `AddressSimSpillManagement`). That is fixed in Task 5. To keep Task 4 self-contained and green, gate: DO Task 5 in the same commit if the build cannot be green with Mock still on the address-sim intermediate. (See Task 5 note — Tasks 4 and 5 land together.)
- [ ] **Step 5: Commit** with Task 5.

---

### Task 5: `StatefulL1SpillManagement` — op-query replay hooks

Give the Mock path its own subclass with the stateful hooks; remove it from the
address-sim intermediate. This is what makes the build green after Task 4.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h`
- Modify: `lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`
- Modify: `lib/Dialect/TTNN/Transforms/OptimizerPasses/GreedyL1SpillManagement.cpp`

**Interfaces produced:**
- `class StatefulL1SpillManagement final : public L1SpillManagementBase<MockAllocatorL1Tracker>` overriding `placeValidatedOutput`, `recoverFromOOM`, `commitAllocation`'s `trackAllocation`, and `replayFrom`.

Hook bodies:
- `trackAllocation(val, size)`: if `willAliasView(val)` (see below) → `memoryTracker.addTensorAlias(val, val.getDefiningOp()->getOperand(0))`; else `memoryTracker.addTensor(val, size)`. `willAliasView(op)` = `isAliasingViewOp(defOp) && memoryTracker.liveRecords.count(defOp->getOperand(0)|owner)`.
- `placeValidatedOutput(op,pos,data,result)`: return `result.outputL1Usage` unchanged (NO fit/frag/CB check). (The base still calls `commitAllocation` afterwards.)
- `recoverFromOOM(...)`: the drop-and-replay loop:
```cpp
auto config = extractOpConfigFromIR(op);
auto result = op_constraint_validation::ValidationResult::outOfMemoryError("");
bool fits = evictUntil(pos, data, [&]{
  auto inputs = utils::extractInputLayouts(op);
  result = memoryTracker.validate(op, inputs, config);
  return result.isSuccess();
});
if (compilationFailed) return;
if (fits) {
  uint64_t l1 = result.outputL1Usage;
  if (data.schedule[pos] != op) return;   // reshard shifted op; sweep reprocesses
  if (l1 > 0) { addResultsToLiveSet(l1); observer_->onLiveAdded(op,pos,l1,pos,memoryTracker.getOccupiedL1()); }
} else { observer_->onSelfSpill(op,pos); demoteToDram(op); }
```
  (`evictUntil`/`evictValue`/`markEvictedAndRebuild` are the shared base drivers;
  they call `replayFrom` through the vtable.)
- `replayFrom(startIdx)` (stateful): restore the `RecordVec`-set checkpoint at
  `startIdx`, then walk `[startIdx, end)`:
```cpp
memoryTracker.restoreSnapshot(checkpoints[startIdx]);  // RecordVec-set
for (size_t i = startIdx; i < l1EventLog.size(); ++i) {
  auto &e = l1EventLog[i];
  if (e.skipped) continue;
  if (e.kind == L1Event::kDealloc) { memoryTracker.removeTensor(e.tensor); continue; }
  checkpoints[i] = memoryTracker.takeSnapshot();
  Operation *defOp = e.tensor.getDefiningOp();
  Value src = defOp ? defOp->getOperand(0) : Value();
  if (defOp && isAliasingViewOp(defOp) && memoryTracker.hasTensor(src)) {
    memoryTracker.addTensorAlias(e.tensor, src); continue;
  }
  auto inputs = utils::extractInputLayouts(defOp);
  auto cfg = extractOpConfigFromIR(defOp);
  auto r = memoryTracker.validate(defOp, inputs, cfg);   // re-query under new history
  if (!r.isSuccess()) return false;                      // still-live op no longer fits
  memoryTracker.addTensor(e.tensor, r.outputL1Usage);    // commit fresh record
}
return true;
```
  Note: the base's `addressSnapshots` map is renamed generically to
  `checkpoints` of type `DenseMap<size_t, typename MemoryTracker::Snapshot>` (it
  already is `MemoryTracker::Snapshot`), so no base change beyond the rename.

- [ ] **Step 1: Write failing test** `EvictionRebuildsLiveSetByReplay` in `TestL1SpillManagementMockAllocator.cpp`: schedule three L1 ops A,B,C where A+B+C exceed budget and C's last use is nearest; drive `run()`; assert the farthest-use victim (A or B) is spilled to DRAM (a `ToMemoryConfigOp` appears) and the surviving live set re-validates clean (no `hasFailed()`).
- [ ] **Step 2: Run, expect FAIL** (Mock path still address-sim / not compiling after Task 4).
- [ ] **Step 3:** Add `StatefulL1SpillManagement` with the four overrides above; rename base `addressSnapshots` → `checkpoints`; point the pass driver's mock branch at `StatefulL1SpillManagement`; remove `MockAllocatorL1Tracker` from the address-sim intermediate so only `SumL1SpillManagement` uses it.
- [ ] **Step 4: Build.** `cmake --build build -- -j$(nproc)`. Expected: 0 errors.
- [ ] **Step 5: Run tests.** `cmake --build build --target L1SpillManagementTests L1SpillManagementMockAllocatorTests ValidationTests` then run all three binaries. Expected: Sum tests unchanged PASS; new + existing Mock tests PASS.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "[optimizer] Stateful spill path: op-query replay, drop address simulator"`

---

### Task 6: Edge-case tests — reshard-past-consumer, multi-result, view tripwire, ToLayout

**Files:**
- Test: `test/unittests/Optimizer/TestL1SpillManagementMockAllocator.cpp`

- [ ] **Step 1:** Add `ReshardPastConsumerDuringStatefulEviction` — a victim with a
  past L1-requiring consumer; after eviction assert a reshard (`ToMemoryConfigOp`
  back to L1) is inserted before that consumer AND the current op re-validates
  clean; assert the reshard value is not a live record at the current op.
- [ ] **Step 2:** Add `MultiResultOpPerResultRecords` — an op with two L1 results;
  assert each result gets its own `liveRecords` entry sourced from
  `output_allocations` (not an even split).
- [ ] **Step 3:** Keep/port the existing `MockAllocatorViewTripwireTest` cases so
  they run against the slimmed tracker (view op query returns aliased/zero
  address; `isAliasingViewOp` keeps it off a fresh record; `getOccupiedL1`
  unchanged by the view).
- [ ] **Step 4:** Add `ToLayoutL1OutputNoRegression` — a `ToLayoutOp` with L1
  output followed by its consumer; assert `run()` does not fail and the consumer
  validates (OOM, if any, surfaces at the consumer, not the ToLayoutOp).
- [ ] **Step 5: Run** `L1SpillManagementMockAllocatorTests`. Expected: all PASS.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "[optimizer] Stateful spill edge-case tests"`

---

### Task 7: Integration — lit + local models + sweeps

- [ ] **Step 1:** `llvm-lit test/ttmlir/Dialect/TTNN/optimizer/ -v`. Expected: all pass.
- [ ] **Step 2:** `cmake --build build --target check-ttmlir`. Expected: pass.
- [ ] **Step 3:** Trigger n150 + p150 `Performance Benchmark` sweeps on the branch
  head (via the xla-perf-pipeline flow). Record run IDs; gate marking the PR ready
  on green sweeps (target LLMs + the #9064 CB-clash models).
- [ ] **Step 4:** Update the PR body's "Essence" to note the spill path now derives
  fit/frag/CB entirely from captured state (no address simulator on the stateful
  path). Commit doc-only change if PR body is tracked in-repo; otherwise update via
  `gh pr edit`.

---

## Self-review notes

- **Spec coverage:** base+subclass structure (T1–3), slimmed tracker + record
  aliasing (T4), op-query replay + drop-and-rebuild eviction (T5), reshard/
  multi-result/view/ToLayout edges (T6), Sum byte-identical guard (every task's
  Step "guard tests"), testing + sweeps (T6/T7). All spec sections mapped.
- **Ordering risk:** Tasks 4+5 must land in one commit (the slimmed tracker breaks
  the address-sim intermediate for Mock until the stateful subclass replaces it).
  This is called out explicitly in T4 Step 4 / T5.
- **Type consistency:** hook names (`placeValidatedOutput`, `recoverFromOOM`,
  `trackAllocation`, `replayFrom`), tracker methods (`addTensorAlias`,
  `removeTensor`, `takeSnapshot`), and the `checkpoints` rename are used
  identically across tasks.
