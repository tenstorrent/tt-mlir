// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1SpillObserver.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>

namespace mlir::tt::ttnn {

/// Sum-based L1 memory tracker with address simulation. Tracks total L1 as
/// sum of per-result tensor sizes, and simulates top-down allocation to detect
/// fragmentation (CB clash with low-address tensors).
struct SumL1MemoryTracker {
  /// Backend validator hook. When null, calls go to
  /// op_constraint_validation::validateOperation(). When set (test-only),
  /// forwards to the supplied callback. Used by every backend-validator
  /// call in the pass (validate() and the two direct probes).
  using BackendValidatorFn =
      std::function<op_constraint_validation::ValidationResult(
          Operation *, llvm::ArrayRef<TTNNLayoutAttr>, const OpConfig &,
          uint64_t /*additionalL1*/)>;
  BackendValidatorFn backendValidator;

  op_constraint_validation::ValidationResult
  validate(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
           const OpConfig &config) const;

  /// Bypass input-overlap accounting and call the backend (or hook) with an
  /// explicit additionalL1Usage. Used by consumer-reshard probes and DRAM
  /// CB re-queries inside L1SpillManagementBase — those call sites already know
  /// the L1 pressure they want to express.
  op_constraint_validation::ValidationResult validateBackendDirect(
      Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
      const OpConfig &config, uint64_t additionalL1Usage) const;

  /// Initialize address simulation with the L1 budget. Must be called before
  /// addTensor/removeTensor.
  void init(uint64_t l1BudgetPerCore);

  uint64_t getOccupiedL1() const;

  void addTensor(Value result, uint64_t l1SizePerCore);

  /// Add `result` as an alias of `srcAtSameAddr`'s buffer (e.g. a
  /// `ttnn.reshape` that tt-metal realizes as a zero-copy view).
  ///
  /// Aliasing model: a buffer (address slot) is allocated once at first
  /// `addTensor` and reclaimed once when the last alias dies. While
  /// aliased, both src and result have entries in `tensorAddresses`
  /// pointing to the same `(start, size)`, and `aliasGroups[start].count`
  /// counts the live aliases. `currentOccupied` is bumped on first
  /// allocation only; this method does NOT bump it.
  ///
  /// `tensorSizes[result]` IS set so `validate`'s inputOverlap accounting
  /// works for downstream consumers of the view.
  void addTensorAtAddress(Value result, uint64_t l1SizePerCore,
                          Value srcAtSameAddr);

  void removeTensor(Value result);

  /// Remove tensor from size tracking only (tensorSizes, currentOccupied).
  /// Does NOT touch address simulation (freeList, tensorAddresses).
  /// Used in eviction paths where addresses are rebuilt via replay.
  void removeTensorFromSizes(Value result);
  bool hasTensor(Value result) const;
  uint64_t getTensorSize(Value result) const;

  /// Return the lowest simulated address of any allocated tensor.
  /// Returns l1Budget if no tensors are allocated.
  uint64_t getLowestOccupiedAddress() const;

  /// Speculative allocation query: returns the address where a tensor of the
  /// given size would be placed (top-down first-fit, 32-byte aligned), without
  /// actually allocating. Returns nullopt if no contiguous block fits.
  std::optional<uint64_t> wouldAllocateAt(uint64_t l1SizePerCore) const;

  void logState() const;

  // --- Public types for snapshot/replay ---

  struct FreeBlock {
    uint64_t start;
    uint64_t end;
    uint64_t size() const { return end - start; }
  };

  /// Per-slot alias-group bookkeeping. See `addTensorAtAddress`.
  struct AliasGroup {
    unsigned count;
    uint64_t rawSize;
  };

  /// Snapshot of the address simulation state (freeList + tensorAddresses
  /// + aliasGroups + currentOccupied). Captured before each allocation
  /// event so that eviction can replay from any point in the schedule.
  struct Snapshot {
    llvm::SmallVector<FreeBlock> freeList;
    llvm::DenseMap<Value, std::pair<uint64_t, uint64_t>> tensorAddresses;
    llvm::DenseMap<uint64_t, AliasGroup> aliasGroups;
    uint64_t currentOccupied;
  };

  /// Take a snapshot of the current address simulation state.
  Snapshot takeSnapshot() const;

  /// Restore the address-simulation state from a snapshot. Restores
  /// freeList, tensorAddresses, aliasGroups, and currentOccupied. Does
  /// NOT touch tensorSizes — it is the original-sweep mirror that
  /// eviction maintains directly via `removeTensorFromSizes`.
  void restoreSnapshot(const Snapshot &snapshot);

  /// Allocate an address block for a tensor (top-down first-fit, aligned),
  /// open a fresh alias group at refcount 1, and bump currentOccupied for
  /// the new slot. Does not touch tensorSizes (callers set it).
  ///
  /// HARD CONTRACT: the tensor MUST fit (asserts otherwise). Callers should
  /// pre-check with `wouldAllocateAt`.
  void allocateAddress(Value result, uint64_t l1SizePerCore);

  /// Add `result` to `srcAtSameAddr`'s alias group at the same address
  /// slot. Bumps the group's refcount only — does not carve a new slot,
  /// bump currentOccupied, or touch tensorSizes. See `addTensorAtAddress`
  /// for the full aliasing model.
  void allocateAddressAt(Value result, Value srcAtSameAddr);

  /// Drop `result` from its alias group. When the last alias is freed,
  /// the slot returns to `freeList` (with adjacent-block merge) and
  /// currentOccupied is reclaimed. Does not touch tensorSizes.
  void freeAddress(Value result);

  /// Check whether `result` currently has an entry in `tensorAddresses`
  /// (i.e. occupies a simulated L1 address slot). Distinct from
  /// `hasTensor`, which checks `tensorSizes`. The two can disagree during
  /// replay (sizes is the original-pass state, addresses is being rebuilt).
  bool hasTensorAddress(Value result) const;

private:
  uint64_t currentOccupied = 0;
  llvm::DenseMap<Value, uint64_t> tensorSizes;

  // --- Address simulation state ---
  static constexpr uint64_t kL1Alignment = 32;
  uint64_t l1Budget = 0;

  llvm::SmallVector<FreeBlock> freeList;

  // Per-Value -> (start, alignedSize). Multiple Values may map to the same
  // slot when an alias group is active. See `addTensorAtAddress`.
  llvm::DenseMap<Value, std::pair<uint64_t, uint64_t>> tensorAddresses;

  // Per-slot refcount + raw size, keyed by slot start. See
  // `addTensorAtAddress` for the aliasing model.
  llvm::DenseMap<uint64_t, AliasGroup> aliasGroups;
};

/// L1 memory tracker backed by tt-metal's stateful op-constraints query.
///
/// Holds only the set of currently-live L1 allocation records (no address
/// simulation). validate() flattens the live records into the op-model's
/// stateful query, so fit / fragmentation / placement / CB-clash are all
/// modeled by tt-metal's real allocator. Records are keyed by the producing
/// Value; a view op's output aliases its source's buffer via `aliasOf` /
/// `aliasRefcount` (record-level aliasing, no addresses), so it never adds a
/// second record and the buffer stays live until the last aliaser dies.
struct MockAllocatorL1Tracker {
  using RecordVec = llvm::SmallVector<op_model::OpModelAllocationRecord>;

  /// Backend validator hook (test-only). Same shape as SumL1MemoryTracker's:
  /// when set, every validate()/validateBackendDirect() call forwards to it.
  using BackendValidatorFn =
      std::function<op_constraint_validation::ValidationResult(
          Operation *, llvm::ArrayRef<TTNNLayoutAttr>, const OpConfig &,
          uint64_t /*additionalL1*/)>;
  BackendValidatorFn backendValidator;

  /// Currently-live L1 allocations, keyed by the owning Value (an alias does
  /// NOT get its own entry). Flattened into each validate()'s initial state.
  llvm::DenseMap<Value, RecordVec> liveRecords;

  /// Records produced by the most recent successful validate(), keyed by op,
  /// awaiting association with result Values at addTensor time. `mutable` so
  /// the const validate() can populate it.
  mutable llvm::DenseMap<Operation *, RecordVec> pendingRecords;

  /// Maps a view-op output (aliaser) to the canonical Value that owns the
  /// shared buffer's record.
  llvm::DenseMap<Value, Value> aliasOf;

  /// Per-owner count of live aliasers (including the owner itself). The owner's
  /// record is dropped from liveRecords when this reaches zero.
  llvm::DenseMap<Value, unsigned> aliasRefcount;

  /// Optimizer per-core L1 budget (a ~0.95 cap minus reserved L1 — tighter than
  /// the device's physical L1 that the query models). validate() enforces it as
  /// a byte ceiling on top of the query so the optimizer keeps its headroom.
  uint64_t l1Budget = 0;

  void init(uint64_t l1BudgetPerCore);

  /// Stateful fit decision: flatten liveRecords (deduped by owner) into the
  /// op-model's stateful query and stash the resulting per-output records for
  /// association at addTensor time.
  op_constraint_validation::ValidationResult
  validate(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
           const OpConfig &config) const;

  /// Query a single op in isolation with an explicit additionalL1Usage (no live
  /// records). Used by the eviction path's consumer-reshard probes.
  op_constraint_validation::ValidationResult validateBackendDirect(
      Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
      const OpConfig &config, uint64_t additionalL1Usage) const;

  /// Associate `result`'s allocation record from the pending stash (positional:
  /// i-th tensor result <-> i-th output-buffer record). No-op if the op
  /// produced no records. `l1SizePerCore` is unused (records carry sizes) and
  /// kept only for interface parity with SumL1MemoryTracker.
  void addTensor(Value result, uint64_t l1SizePerCore);

  /// Record `out` as an alias of `src`'s buffer: bump the owner's refcount and
  /// add no new record.
  void addTensorAlias(Value out, Value src);

  /// Drop `result` from the live set (dead or spilled to DRAM): decrement its
  /// owner's refcount and erase the record when it reaches zero.
  void removeTensor(Value result);
  void removeTensorFromSizes(Value result);

  bool hasTensor(Value result) const;
  uint64_t getTensorSize(Value result) const;
  uint64_t getOccupiedL1() const;

  /// Replay checkpoint: the live record set + alias bookkeeping.
  struct Snapshot {
    llvm::DenseMap<Value, RecordVec> liveRecords;
    llvm::DenseMap<Value, Value> aliasOf;
    llvm::DenseMap<Value, unsigned> aliasRefcount;
  };
  Snapshot takeSnapshot() const;
  void restoreSnapshot(const Snapshot &snapshot);
};

/// L1SpillManagementBase enforces L1 budget constraints using farthest-last-use
/// eviction with validation-based budget enforcement.
/// Each op is re-validated during the sweep using validateOperation() with
/// the sum of live tensor L1 sizes as additionalL1Usage. When validation
/// fails (OOM), tensors are evicted until it succeeds.
///
/// Templatized on a MemoryTracker type to allow pluggable L1 tracking
/// strategies (simple sum today, allocator-backed tomorrow).
///
/// After processing, it modifies the IR:
/// - Inserts ToMemoryConfigOp after spilled ops (L1 -> DRAM)
/// - Reconnects uses to read from spilled tensor
template <typename MemoryTracker = SumL1MemoryTracker>
class L1SpillManagementBase {
public:
  L1SpillManagementBase(func::FuncOp func, ttcore::GridAttr deviceGrid,
                        uint64_t l1BudgetPerCore,
                        std::unique_ptr<L1SpillObserver> observer = nullptr);

  virtual ~L1SpillManagementBase() = default;

  /// Run farthest-last-use eviction with validation-based enforcement and apply
  /// spills directly to the IR.
  void run();

  /// Access the observer (always non-null; NullObject when tracing disabled).
  L1SpillObserver *getObserver() { return observer_.get(); }

  /// Access the memory tracker. Intended for test-only use: install a
  /// backendValidator before calling run().
  MemoryTracker &getMemoryTracker() { return memoryTracker; }
  const MemoryTracker &getMemoryTracker() const { return memoryTracker; }

  /// True if run() hit an unrecoverable condition and emitted an error. The
  /// driving pass must signalPassFailure() when this is set (emitError alone
  /// does not fail the pass).
  bool hasFailed() const { return compilationFailed; }

protected:
  func::FuncOp func;
  ttcore::GridAttr deviceGrid;
  uint64_t l1BudgetPerCore;

  /// Precomputed CB fragmentation cushion (bytes). See kCBFragCushionFraction.
  uint64_t cbFragCushion;

  /// Set when run() emits an error for an unrecoverable condition.
  bool compilationFailed = false;

  /// Observer (NullObject pattern: always non-null).
  std::unique_ptr<L1SpillObserver> observer_;

  /// Pluggable memory state tracker.
  MemoryTracker memoryTracker;

  /// Farthest-last-use ordering: max-heap by lastUsePosition, keyed by Value.
  using LiveEntry = std::pair<int64_t, Value>;
  struct LiveEntryCompare {
    bool operator()(const LiveEntry &a, const LiveEntry &b) const {
      return a.first < b.first; // max-heap: higher lastUse = higher priority
    }
  };
  using LiveSet =
      std::priority_queue<LiveEntry, std::vector<LiveEntry>, LiveEntryCompare>;
  LiveSet liveSet;
  llvm::DenseSet<Value> liveValues;

  /// Full sweep state captured before processing a schedule position, so the
  /// stateful eviction (StatefulL1SpillManagement) can rewind and re-run the
  /// forward sweep from there. Bundles the tracker snapshot with the FLU heap
  /// and live-value set (which are not cheaply re-derivable once eviction has
  /// changed the L1-resident set).
  struct SweepCheckpoint {
    typename MemoryTracker::Snapshot tracker;
    llvm::DenseSet<Value> liveValues;
    LiveSet liveSet;
  };
  /// Per-position sweep checkpoints (stateful path only; keyed by schedule
  /// pos).
  llvm::DenseMap<int64_t, SweepCheckpoint> sweepCheckpoints;

  /// When set by the stateful recoverFromOOM, run() restores the checkpoint at
  /// this position and re-runs the sweep from it.
  std::optional<int64_t> pendingRewindTo;

  /// True for spill strategies that rebuild allocator state by rewinding and
  /// re-running the forward sweep (StatefulL1SpillManagement) rather than the
  /// address-sim event-log replay. Gates checkpoint capture + rewind wiring so
  /// the Sum/address-sim path stays byte-identical.
  virtual bool usesRewindEviction() const { return false; }

  /// Capture / restore the full sweep state (tracker snapshot + liveValues +
  /// liveSet) at a schedule position. Used by the rewind eviction.
  void captureCheckpoint(int64_t pos);
  void restoreCheckpoint(int64_t pos);

  /// Event log entry for address reconstruction. Records every L1 allocation
  /// and deallocation in schedule order so that eviction can replay the full
  /// history (including dead tensors) to compute accurate addresses.
  /// Reshards inserted for past consumers also get kAlloc/kDealloc entries
  /// (via insertEventIntoLog) so future replays account for their transient
  /// slot without needing a separate injection map.
  struct L1Event {
    enum Kind { kAlloc, kDealloc };
    Kind kind;
    Value tensor;
    uint64_t sizePerCore; // meaningful for kAlloc only
    bool skipped = false; // set true when tensor is evicted
  };

  /// Ordered log of all L1 alloc/dealloc events during the sweep.
  llvm::SmallVector<L1Event> l1EventLog;

  /// Snapshots of tracker state taken before each alloc event, keyed by
  /// event-log index. Invariant: addressSnapshots[i] = tracker state
  /// immediately before event i fires. Maintained by insertEventIntoLog.
  /// Used as starting points for replay after eviction.
  llvm::DenseMap<size_t, typename MemoryTracker::Snapshot> addressSnapshots;

  /// Maps each live-tensor Value to its alloc event index in l1EventLog.
  /// O(1) restore-point lookup in markEvictedAndRebuild. NOT populated for
  /// reshard events for past consumers — irrelevant as reshards are never
  /// selected as eviction victims.
  llvm::DenseMap<Value, size_t> allocEventIndex;

  /// Results of reshards inserted by insertReshardIntoSchedule (future
  /// consumers). These must never be selected as eviction victims — evicting
  /// them defeats the purpose of the reshard. A DenseSet rather than checking
  /// isa<ToMemoryConfigOp> avoids falsely blocking regular to_memory_config
  /// ops from MemoryLayoutPropagation, which ARE valid eviction candidates.
  llvm::DenseSet<Value> insertedReshardValues;

  /// Mark victim's alloc/dealloc events as skipped, fold its alloc index into
  /// `campaignMin` (the smallest alloc-event index evicted so far in the
  /// enclosing campaign), then rebuild by replaying the whole suffix from that
  /// minimum (see `replayFrom`). Returns true iff every still-live tensor was
  /// placed; false means a still-live tensor has no contiguous fit
  /// (fragmentation).
  bool markEvictedAndRebuild(Value victim, size_t &campaignMin);

  /// Insert event at pos in l1EventLog and shift all allocEventIndex entries
  /// and addressSnapshots keys >= pos by 1, preserving the index invariants.
  void insertEventIntoLog(size_t pos, L1Event event);

  /// Extract OpConfig from op's current IR state (result type + op-specific
  /// attrs like Conv2dConfig, MatmulProgramConfig).
  static OpConfig extractOpConfigFromIR(Operation *op);

  /// Evict one result tensor (farthest last-use). Returns evicted Value, or
  /// null.
  Value evictFarthestUse();

  /// Update op's IR to reflect the output layout from a validation result
  /// (result type, L1 usage attr).
  void
  applyOutputConfig(Operation *op,
                    const op_constraint_validation::ValidationResult &result);

  /// After spilling a victim to DRAM, re-validate consumers via worklist.
  /// Cascades through already-processed ops (pos < currentPos) until no more
  /// IR changes occur. Ops after currentPos are handled by the main loop.
  void
  revalidateConsumers(Operation *changedOp, int64_t currentPos,
                      const llvm::DenseMap<Operation *, int64_t> &positionMap);

  /// For each tensor result of ops in the schedule, compute the schedule
  /// position of its last consumer (per-result granularity).
  llvm::DenseMap<Value, int64_t>
  computeLastUsePositions(const llvm::SmallVector<Operation *> &schedule);

  /// Demote an op's output from L1 to DRAM interleaved in place.
  /// For ToMemoryConfigOp, also updates the memory_config attribute.
  void demoteToDram(Operation *op);

  /// Insert a ToMemoryConfigOp right after `result`'s defining op to spill
  /// `result` to DRAM. ALL uses of `result` (past and future) get rewired
  /// to read from the spill. Snapshot/replay-safe: pair with
  /// `markEvictedAndRebuild` to keep the address simulator consistent.
  void spillToDram(Value result);

  /// Insert a ToMemoryConfigOp just BEFORE `triggerOp` to spill `result` to
  /// DRAM. Only uses at/after `triggerOp` get rewired; earlier uses keep
  /// reading from L1.
  ///
  /// INVARIANT: callers MUST follow with a full tracker reset (e.g.
  /// `memoryTracker.init(l1BudgetPerCore)`) because the past-of-trigger
  /// uses still occupy L1 in the IR. The default `markEvictedAndRebuild`
  /// snapshot/replay path is NOT compatible with this overload — it would
  /// mark `result`'s alloc as skipped at the producer's event index, which
  /// disagrees with the IR (`result` is still allocated between defOp and
  /// triggerOp).
  ///
  /// Currently the only caller is `evictAllFromL1`.
  void spillToDramBeforeTrigger(Value result, Operation *triggerOp);

  /// Shared implementation of the two spill overloads. Not part of the
  /// public API.
  void spillToDramImpl(Value result, Operation *insertBefore);

  /// Bundled schedule data built once at the start of run().
  struct ScheduleData {
    llvm::SmallVector<Operation *> schedule;
    llvm::DenseMap<Value, int64_t> lastUsePositions;
    llvm::DenseMap<int64_t, llvm::SmallVector<Value>> deathSchedule;
    llvm::DenseMap<Operation *, int64_t> positionMap;
  };

  /// Build schedule, last-use positions, death schedule, and position map.
  ScheduleData buildScheduleData();

  /// Insert a reshard op into the schedule at consumerPos (shifting the
  /// consumer and all later ops by +1), updating positionMap, lastUsePositions,
  /// and deathSchedule so the forward sweep processes the reshard naturally.
  void insertReshardIntoSchedule(Operation *reshardOp, Value reshardResult,
                                 uint64_t reshardSizePerCore,
                                 int64_t consumerPos, ScheduleData &data);

  /// Remove result tensors whose last use was the previous position.
  void processDeadTensors(int64_t pos, const ScheduleData &data);

protected:
  // --- Strategy hooks. Implemented by AddressSimSpillManagement (address-sim)
  //     and StatefulL1SpillManagement (captured allocator state). ---

  /// Place a successfully-validated op's output. Returns the L1 size to add to
  /// the live set (0 = demoted to DRAM).
  virtual uint64_t placeValidatedOutput(
      Operation *op, int64_t pos, ScheduleData &data,
      const op_constraint_validation::ValidationResult &result) = 0;

  /// Recover from an OOM validation result (evict live tensors or demote self
  /// to DRAM).
  virtual void
  recoverFromOOM(Operation *op, int64_t pos,
                 llvm::ArrayRef<OpResult> tensorResults, ScheduleData &data,
                 std::function<void(uint64_t)> addResultsToLiveSet) = 0;

  /// Commit one result tensor into the live set: log the alloc event, add it to
  /// the tracker (alias-or-fresh), and push it onto the FLU heap.
  virtual void commitAllocation(Value val, uint64_t perResultL1,
                                ScheduleData &data) = 0;

  /// Handle an L1-output op that cannot be validated (a pre-decomposition
  /// ToLayoutOp). Address-sim runs a fit check; the stateful path defers to the
  /// consuming op's query.
  virtual void handleUnvalidatedL1Output(Operation *op, int64_t pos,
                                         ScheduleData &data,
                                         uint64_t derivedL1) = 0;

  /// Rebuild live-set state after an eviction marks events skipped, replaying
  /// from `startIdx`. Returns true iff every still-live tensor was placed.
  virtual bool replayFrom(size_t startIdx) = 0;

protected:
  /// Evict all live L1 tensors. Used when encountering ops without OpModel
  /// support — since we cannot know their L1 requirements, the only safe
  /// choice is a full flush. Spill ops are inserted right before triggerOp
  /// (the CCL op) so earlier consumers can still read from L1.
  void evictAllFromL1(int64_t pos, ScheduleData &data,
                      Operation *triggerOp = nullptr);

  /// Evict tensors (farthest-last-use first) until shouldStop() returns true
  /// or the live set is empty. Returns true if shouldStop was satisfied.
  /// After each eviction, rebuilds tracker state (via the replayFrom hook) and
  /// inserts reshards for already-processed consumers.
  bool evictUntil(int64_t pos, ScheduleData &data,
                  std::function<bool()> shouldStop);

  /// Evict a specific live value: spill to DRAM, update tracker, and insert
  /// reshards for consumers that still require the L1 layout. If
  /// skipReshardConsumer is non-null, that one consumer is left reading the
  /// DRAM spill (no reshard inserted for it); all other consumers are still
  /// restored.
  /// `campaignMin` accumulates the smallest alloc-event index evicted across
  /// the enclosing campaign (caller inits it to SIZE_MAX) so every rebuild
  /// replays a self-consistent suffix; see `markEvictedAndRebuild`.
  /// Returns markEvictedAndRebuild's result: true iff the post-eviction replay
  /// placed every still-live tensor.
  bool evictValue(Value victim, int64_t pos, ScheduleData &data,
                  size_t &campaignMin,
                  Operation *skipReshardConsumer = nullptr);

  /// Insert a ToMemoryConfigOp before an already-processed consumer to
  /// convert the DRAM spill output back to the consumer's expected L1 layout.
  void insertReshardForConsumer(Operation *consumer, unsigned operandIdx,
                                TTNNLayoutAttr originalL1Layout);

  /// Collect downstream consumers of an op, following through spill ops.
  static llvm::SmallVector<Operation *>
  collectDownstreamConsumers(Operation *changed);
};

// Explicit instantiation declarations (definitions in .cpp).
extern template class L1SpillManagementBase<SumL1MemoryTracker>;
extern template class L1SpillManagementBase<MockAllocatorL1Tracker>;

/// Address-sim strategy: implements the spill hooks with the simulated top-down
/// first-fit allocator (SumL1MemoryTracker's address model) and the
/// contiguous-fit / CB-overlap fragmentation checks. Shared by the legacy Sum
/// path and (transitionally) the Mock path until the stateful path lands.
template <typename MemoryTracker>
class AddressSimSpillManagement : public L1SpillManagementBase<MemoryTracker> {
public:
  using L1SpillManagementBase<MemoryTracker>::L1SpillManagementBase;

protected:
  using Base = L1SpillManagementBase<MemoryTracker>;
  using typename Base::L1Event;
  using typename Base::ScheduleData;
  // Base state + generic helpers the address-sim strategy reuses. (Dependent
  // base: member names must be brought into scope explicitly.)
  using Base::addressSnapshots;
  using Base::allocEventIndex;
  using Base::applyOutputConfig;
  using Base::cbFragCushion;
  using Base::collectDownstreamConsumers;
  using Base::compilationFailed;
  using Base::demoteToDram;
  using Base::evictFarthestUse;
  using Base::evictUntil;
  using Base::evictValue;
  using Base::extractOpConfigFromIR;
  using Base::insertedReshardValues;
  using Base::insertEventIntoLog;
  using Base::insertReshardForConsumer;
  using Base::insertReshardIntoSchedule;
  using Base::l1BudgetPerCore;
  using Base::l1EventLog;
  using Base::liveSet;
  using Base::liveValues;
  using Base::markEvictedAndRebuild;
  using Base::memoryTracker;
  using Base::observer_;
  using Base::spillToDram;

  // Strategy hooks (address-sim implementations).
  uint64_t placeValidatedOutput(
      Operation *op, int64_t pos, ScheduleData &data,
      const op_constraint_validation::ValidationResult &result) override;
  void
  recoverFromOOM(Operation *op, int64_t pos,
                 llvm::ArrayRef<OpResult> tensorResults, ScheduleData &data,
                 std::function<void(uint64_t)> addResultsToLiveSet) override;
  void commitAllocation(Value val, uint64_t perResultL1,
                        ScheduleData &data) override;
  void handleUnvalidatedL1Output(Operation *op, int64_t pos, ScheduleData &data,
                                 uint64_t derivedL1) override;
  bool replayFrom(size_t startIdx) override;

  // Address-simulation fit / fragmentation / CB-overlap helpers.
  uint64_t ensureFitsL1(Operation *op, int64_t pos, ScheduleData &data,
                        uint64_t cbPeakUsage, uint64_t l1Size);
  uint64_t handleNoFit(Operation *op, int64_t pos, ScheduleData &data,
                       uint64_t outputL1Size);
  uint64_t handleFragmentation(Operation *op, int64_t pos, ScheduleData &data,
                               uint64_t cbPeakUsage, uint64_t outputL1Size);
  bool wouldCBsOverlapTensors(Operation *op, int64_t pos, uint64_t cbPeakUsage,
                              uint64_t speculativeOutputAddr);
  bool willAliasSourceInL1(Operation *op) const;
  void handleOOM(Operation *op, int64_t pos,
                 llvm::ArrayRef<OpResult> tensorResults, ScheduleData &data,
                 std::function<void(uint64_t)> addResultsToLiveSet);
  void evictForCBOverlap(uint64_t cushionedCBUsage, int64_t pos,
                         ScheduleData &data);
  void evictForDramCBGrowth(Operation *op, int64_t pos, ScheduleData &data);
};

extern template class AddressSimSpillManagement<SumL1MemoryTracker>;

/// Legacy spill manager: scalar-sum tracker + simulated top-down first-fit
/// address model. Selected when `use-mock-allocator-state=false`.
class SumL1SpillManagement final
    : public AddressSimSpillManagement<SumL1MemoryTracker> {
public:
  using AddressSimSpillManagement<
      SumL1MemoryTracker>::AddressSimSpillManagement;
};

/// Stateful spill manager: fit / fragmentation / placement / CB-clash are all
/// answered by tt-metal's captured allocator state (no address simulation).
/// Selected by default (`use-mock-allocator-state=true`). Eviction drops the
/// victim's records and replays the surviving allocation history through the
/// op-model, so the allocator re-flows addresses under the victim-free history
/// (a stored record cannot simply be dropped: with_allocations pins buffers at
/// their recorded addresses). The base is a non-dependent type here, so base
/// members are reachable without using-declarations.
class StatefulL1SpillManagement final
    : public L1SpillManagementBase<MockAllocatorL1Tracker> {
public:
  using L1SpillManagementBase<MockAllocatorL1Tracker>::L1SpillManagementBase;

protected:
  bool usesRewindEviction() const override { return true; }

  uint64_t placeValidatedOutput(
      Operation *op, int64_t pos, ScheduleData &data,
      const op_constraint_validation::ValidationResult &result) override;
  void
  recoverFromOOM(Operation *op, int64_t pos,
                 llvm::ArrayRef<OpResult> tensorResults, ScheduleData &data,
                 std::function<void(uint64_t)> addResultsToLiveSet) override;
  void commitAllocation(Value val, uint64_t perResultL1,
                        ScheduleData &data) override;
  void handleUnvalidatedL1Output(Operation *op, int64_t pos, ScheduleData &data,
                                 uint64_t derivedL1) override;
  bool replayFrom(size_t startIdx) override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H
