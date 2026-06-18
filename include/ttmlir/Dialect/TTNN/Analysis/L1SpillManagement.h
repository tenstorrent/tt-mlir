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
  /// CB re-queries inside L1SpillManagement — those call sites already know
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

/// L1SpillManagement enforces L1 budget constraints using farthest-last-use
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
class L1SpillManagement {
public:
  L1SpillManagement(func::FuncOp func, ttcore::GridAttr deviceGrid,
                    uint64_t l1BudgetPerCore,
                    std::unique_ptr<L1SpillObserver> observer = nullptr);

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

private:
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
  std::priority_queue<LiveEntry, std::vector<LiveEntry>, LiveEntryCompare>
      liveSet;
  llvm::DenseSet<Value> liveValues;

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

  /// Mark victim's alloc/dealloc events as skipped, restore the snapshot
  /// taken before victim's alloc, then replay all subsequent non-skipped
  /// events to rebuild a consistent address-simulation state. Reshard
  /// kAlloc/kDealloc pairs in the log are replayed naturally.
  void markEvictedAndRebuild(Value victim);

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

  /// Check if the op's CB region (growing bottom-up) would overlap with any
  /// live tensor or the speculative output tensor based on simulated L1
  /// addresses. Uses min(speculativeOutputAddr, lowestOccupiedAddress) as
  /// the effective lowest tensor address.
  /// See: https://github.com/tenstorrent/tt-mlir/issues/7396
  bool wouldCBsOverlapTensors(Operation *op, int64_t pos, uint64_t cbPeakUsage,
                              uint64_t speculativeOutputAddr);

  /// Output cannot fit contiguously in the free list. Evict farthest-last-use
  /// tensors until the output fits, then re-validate. Returns L1 bytes to add
  /// to live set (0 if demoted to DRAM).
  uint64_t handleNoFit(Operation *op, int64_t pos, ScheduleData &data,
                       uint64_t outputL1Size);

  /// CB fragmentation recovery: evict tensors in the CB danger zone,
  /// re-validate, or demote output to DRAM. Returns L1 bytes to add to live
  /// set (0 if demoted).
  uint64_t handleFragmentation(Operation *op, int64_t pos, ScheduleData &data,
                               uint64_t cbPeakUsage, uint64_t outputL1Size);

  /// Run contiguous-fit and CB-fragmentation checks on a validated op's
  /// output. Returns the (possibly updated) L1 size to add to the live set,
  /// or 0 if the output was demoted to DRAM.
  uint64_t ensureFitsL1(Operation *op, int64_t pos, ScheduleData &data,
                        uint64_t cbPeakUsage, uint64_t l1Size);

  /// True when `op` is a view-eligible reshape whose source operand is still
  /// resident in L1. Its output will alias the source's existing slot
  /// (addTensorAtAddress) instead of carving a fresh one, so it consumes no
  /// additional L1 and the fit / CB-overlap checks must not treat it as a
  /// new allocation.
  bool willAliasSourceInL1(Operation *op) const;

  /// OOM recovery: evict farthest-last-use tensors or demote self to DRAM.
  void handleOOM(Operation *op, int64_t pos,
                 llvm::ArrayRef<OpResult> tensorResults, ScheduleData &data,
                 std::function<void(uint64_t)> addResultsToLiveSet);

  /// Evict all live L1 tensors. Used when encountering ops without OpModel
  /// support — since we cannot know their L1 requirements, the only safe
  /// choice is a full flush. Spill ops are inserted right before triggerOp
  /// (the CCL op) so earlier consumers can still read from L1.
  void evictAllFromL1(int64_t pos, ScheduleData &data,
                      Operation *triggerOp = nullptr);

  /// Evict live tensors (farthest-last-use first) until no tensor's simulated
  /// address falls below the cushioned CB threshold. Replaces the former
  /// evictTensorsBelow, which was incorrect after address rebuild (addresses
  /// shift, making the threshold check stale).
  void evictForCBOverlap(uint64_t cushionedCBUsage, int64_t pos,
                         ScheduleData &data);

  /// Evict tensors (farthest-last-use first) until shouldStop() returns true
  /// or the live set is empty. Returns true if shouldStop was satisfied.
  /// After each eviction, rebuilds address simulation and inserts reshards
  /// for already-processed consumers.
  bool evictUntil(int64_t pos, ScheduleData &data,
                  std::function<bool()> shouldStop);

  /// Evict a specific live value: spill to DRAM, update tracker, and insert
  /// reshards for consumers that still require the L1 layout. If
  /// skipReshardConsumer is non-null, that one consumer is left reading the
  /// DRAM spill (no reshard inserted for it); all other consumers are still
  /// restored.
  void evictValue(Value victim, int64_t pos, ScheduleData &data,
                  Operation *skipReshardConsumer = nullptr);

  /// Insert a ToMemoryConfigOp before an already-processed consumer to
  /// convert the DRAM spill output back to the consumer's expected L1 layout.
  void insertReshardForConsumer(Operation *consumer, unsigned operandIdx,
                                TTNNLayoutAttr originalL1Layout);

  /// After demoting an op's output to DRAM, re-query op_model for the DRAM
  /// config's cbPeakUsage and evict any live tensors that fall within the
  /// (potentially larger) static CB region.  When output switches from
  /// L1-sharded to DRAM, the output CB flips from globally_allocated (aliased
  /// to the shard, not in the static CB region) to locally_allocated (bottom-up
  /// in the static CB region), which can significantly increase the CB
  /// footprint.
  void evictForDramCBGrowth(Operation *op, int64_t pos, ScheduleData &data);

  /// Collect downstream consumers of an op, following through spill ops.
  static llvm::SmallVector<Operation *>
  collectDownstreamConsumers(Operation *changed);
};

// Explicit instantiation declaration (definition in .cpp).
extern template class L1SpillManagement<SumL1MemoryTracker>;

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H
