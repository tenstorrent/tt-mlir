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
  op_constraint_validation::ValidationResult
  validate(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
           const OpConfig &config) const;

  /// Initialize address simulation with the L1 budget. Must be called before
  /// addTensor/removeTensor.
  void init(uint64_t l1BudgetPerCore);

  uint64_t getOccupiedL1() const;
  void addTensor(Value result, uint64_t l1SizePerCore);
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

  /// Snapshot of the address simulation state (freeList + tensorAddresses).
  /// Captured before each allocation event so that eviction can replay from
  /// any point in the schedule.
  struct Snapshot {
    llvm::SmallVector<FreeBlock> freeList;
    llvm::DenseMap<Value, std::pair<uint64_t, uint64_t>> tensorAddresses;
  };

  /// Take a snapshot of the current address simulation state.
  Snapshot takeSnapshot() const;

  /// Restore the address simulation state from a snapshot (freeList +
  /// tensorAddresses). Does NOT change tensorSizes or currentOccupied.
  void restoreSnapshot(const Snapshot &snapshot);

  /// Allocate an address block for a tensor (top-down first-fit, aligned).
  /// Address-only: does not update tensorSizes/currentOccupied.
  void allocateAddress(Value result, uint64_t l1SizePerCore);

  /// Free a tensor's address block and merge with adjacent free blocks.
  /// Address-only: does not update tensorSizes/currentOccupied.
  void freeAddress(Value result);

private:
  uint64_t currentOccupied = 0;
  llvm::DenseMap<Value, uint64_t> tensorSizes;

  // --- Address simulation state ---
  static constexpr uint64_t kL1Alignment = 32;
  uint64_t l1Budget = 0;

  llvm::SmallVector<FreeBlock> freeList;

  // Allocated tensor addresses: Value -> (start, alignedSize).
  llvm::DenseMap<Value, std::pair<uint64_t, uint64_t>> tensorAddresses;
};

/// L1SpillManagement enforces L1 budget constraints using Belady's optimal
/// page replacement algorithm with validation-based budget enforcement.
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
/// - Removes "ttnn.output_l1_usage" attributes (cleanup)
template <typename MemoryTracker = SumL1MemoryTracker>
class L1SpillManagement {
public:
  L1SpillManagement(func::FuncOp func, ttcore::GridAttr deviceGrid,
                    uint64_t l1BudgetPerCore,
                    std::unique_ptr<L1SpillObserver> observer = nullptr);

  /// Run Belady's algorithm with validation-based eviction and apply spills
  /// directly to the IR.
  void run();

  /// Access the observer (always non-null; NullObject when tracing disabled).
  L1SpillObserver *getObserver() { return observer_.get(); }

private:
  func::FuncOp func;
  ttcore::GridAttr deviceGrid;
  uint64_t l1BudgetPerCore;

  /// Precomputed CB fragmentation cushion (bytes). See kCBFragCushionFraction.
  uint64_t cbFragCushion;

  /// Observer (NullObject pattern: always non-null).
  std::unique_ptr<L1SpillObserver> observer_;

  /// Pluggable memory state tracker.
  MemoryTracker memoryTracker;

  /// Belady eviction ordering: max-heap by lastUsePosition, keyed by Value.
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
  /// event-log index. Used as starting points for replay after eviction.
  llvm::DenseMap<size_t, typename MemoryTracker::Snapshot> addressSnapshots;

  /// Maps each tensor Value to its alloc event index in l1EventLog.
  /// Provides O(1) lookup in markEvictedAndRebuild instead of linear scan.
  llvm::DenseMap<Value, size_t> allocEventIndex;

  /// Mark all events for a tensor as skipped and rebuild from its alloc
  /// snapshot. Updates snapshots during replay so future evictions start
  /// from accurate state.
  void markEvictedAndRebuild(Value victim);

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

  /// Insert ToMemoryConfigOp to spill a single result value to DRAM
  /// interleaved.
  void spillToDram(Value result);

  /// Remove all "ttnn.output_l1_usage" attributes from ops in the function.
  void cleanupL1UsageAttrs();

  /// Bundled schedule data built once at the start of run().
  struct ScheduleData {
    llvm::SmallVector<Operation *> schedule;
    llvm::DenseMap<Value, int64_t> lastUsePositions;
    llvm::DenseMap<int64_t, llvm::SmallVector<Value>> deathSchedule;
    llvm::DenseMap<Operation *, int64_t> positionMap;
  };

  /// Build schedule, last-use positions, death schedule, and position map.
  ScheduleData buildScheduleData();

  /// Remove result tensors whose last use was the previous position.
  void processDeadTensors(int64_t pos, const ScheduleData &data);

  /// Check if the op's CB region (growing bottom-up) would overlap with any
  /// live tensor or the speculative output tensor based on simulated L1
  /// addresses. Uses min(speculativeOutputAddr, lowestOccupiedAddress) as
  /// the effective lowest tensor address.
  /// See: https://github.com/tenstorrent/tt-mlir/issues/7396
  bool wouldCBsOverlapTensors(Operation *op, int64_t pos, uint64_t cbPeakUsage,
                              uint64_t speculativeOutputAddr);

  /// Output cannot fit contiguously in the free list. Evict using Belady's
  /// algorithm until the output fits, then re-validate. Returns L1 bytes to
  /// add to live set (0 if demoted to DRAM).
  uint64_t handleNoFit(Operation *op, int64_t pos, const ScheduleData &data,
                       uint64_t opL1Usage, uint64_t outputL1Size);

  /// CB fragmentation recovery: evict tensors in the CB danger zone,
  /// re-validate, or demote output to DRAM. Returns L1 bytes to add to live
  /// set (0 if demoted).
  uint64_t handleFragmentation(Operation *op, int64_t pos,
                               const ScheduleData &data, uint64_t opL1Usage,
                               uint64_t cbPeakUsage, uint64_t outputL1Size);

  /// Run contiguous-fit and CB-fragmentation checks on a validated op's
  /// output. Returns the (possibly updated) L1 size to add to the live set,
  /// or 0 if the output was demoted to DRAM.
  uint64_t ensureFitsL1(Operation *op, int64_t pos, const ScheduleData &data,
                        uint64_t opL1Usage, uint64_t cbPeakUsage,
                        uint64_t l1Size);

  /// OOM recovery: demote to L1-interleaved, evict farthest-use, or spill self.
  void handleOOM(Operation *op, int64_t pos,
                 llvm::ArrayRef<OpResult> tensorResults,
                 const ScheduleData &data, uint64_t opL1Usage,
                 std::function<void(uint64_t)> addResultsToLiveSet);

  /// Evict all live L1 tensors. Used when encountering ops without OpModel
  /// support — since we cannot know their L1 requirements, the only safe
  /// choice is a full flush.
  void evictAllFromL1(int64_t pos, const ScheduleData &data);

  /// Evict live tensors using Belady's algorithm until no tensor's simulated
  /// address falls below the cushioned CB threshold. Replaces the former
  /// evictTensorsBelow, which was incorrect after address rebuild (addresses
  /// shift, making the threshold check stale).
  void evictForCBOverlap(uint64_t cushionedCBUsage, int64_t pos,
                         const ScheduleData &data);

  /// Evict tensors using Belady's algorithm until shouldStop() returns true
  /// or the live set is empty. Returns true if shouldStop was satisfied.
  /// After each eviction, rebuilds address simulation and inserts reshards
  /// for already-processed consumers.
  bool evictUntil(int64_t pos, const ScheduleData &data,
                  std::function<bool()> shouldStop);

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
  void evictForDramCBGrowth(Operation *op, int64_t pos,
                            const ScheduleData &data);

  /// Collect downstream consumers of an op, following through spill ops.
  static llvm::SmallVector<Operation *>
  collectDownstreamConsumers(Operation *changed);
};

// Explicit instantiation declaration (definition in .cpp).
extern template class L1SpillManagement<SumL1MemoryTracker>;

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H
