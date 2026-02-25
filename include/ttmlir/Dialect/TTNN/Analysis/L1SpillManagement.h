// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <queue>

namespace mlir::tt::ttnn {

/// Simple sum-based L1 memory tracker. Tracks total L1 as sum of per-tensor
/// sizes. Validates by passing the sum as additionalL1Usage to
/// validateOperation.
struct SumL1MemoryTracker {
  op_constraint_validation::ValidationResult
  validate(Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
           const OpConfig &config) const;

  uint64_t getOccupiedL1() const;
  void addTensor(Operation *op, uint64_t l1SizePerCore);
  void removeTensor(Operation *op);
  bool hasTensor(Operation *op) const;
  uint64_t getTensorSize(Operation *op) const;

private:
  uint64_t currentOccupied = 0;
  llvm::DenseMap<Operation *, uint64_t> tensorSizes;
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
                    uint64_t l1BudgetPerCore);

  /// Run Belady's algorithm with validation-based eviction and apply spills
  /// directly to the IR.
  void run();

private:
  func::FuncOp func;
  ttcore::GridAttr deviceGrid;
  uint64_t l1BudgetPerCore;

  /// Pluggable memory state tracker.
  MemoryTracker memoryTracker;

  /// Belady eviction ordering: max-heap by lastUsePosition.
  using LiveEntry = std::pair<int64_t, Operation *>;
  std::priority_queue<LiveEntry> liveSet;
  llvm::DenseSet<Operation *> liveOps;

  /// Extract OpConfig from op's current IR state (result type + op-specific
  /// attrs like Conv2dConfig, MatmulProgramConfig).
  static OpConfig extractOpConfigFromIR(Operation *op);

  /// Evict one tensor (farthest last-use). Returns evicted op, or null.
  Operation *evictFarthestUse();

  /// Build L1 interleaved OpConfig from op's current config.
  static OpConfig makeL1InterleavedConfig(Operation *op);

  /// Update op's IR to reflect demoted layout (result type, DPS operand,
  /// L1 usage attr).
  void
  applyDemotedConfig(Operation *op,
                     const op_constraint_validation::ValidationResult &result);

  /// After spilling a victim to DRAM, re-validate consumers via worklist.
  /// Cascades through already-processed ops (pos < currentPos) until no more
  /// IR changes occur. Ops after currentPos are handled by the main loop.
  void
  revalidateConsumers(Operation *changedOp, int64_t currentPos,
                      const llvm::DenseMap<Operation *, int64_t> &positionMap);

  /// For each op output that has an L1 usage annotation, compute the schedule
  /// position of its last consumer.
  llvm::DenseMap<Operation *, int64_t>
  computeLastUsePositions(const llvm::SmallVector<Operation *> &schedule);

  /// Insert ToMemoryConfigOp to spill a tensor to DRAM interleaved.
  void spillToDram(Operation *op);

  /// Remove all "ttnn.output_l1_usage" attributes from ops in the function.
  void cleanupL1UsageAttrs();
};

// Explicit instantiation declaration (definition in .cpp).
extern template class L1SpillManagement<SumL1MemoryTracker>;

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLMANAGEMENT_H
