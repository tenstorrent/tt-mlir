// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLOBSERVER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLOBSERVER_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>

namespace mlir::tt::ttnn {

class DecisionTrace;

/// Observer interface for L1SpillManagement decisions.
/// NullObject pattern: all methods have empty default implementations.
/// Concrete observers (DecisionTraceObserver) override the methods they care
/// about.
class L1SpillObserver {
public:
  virtual ~L1SpillObserver() = default;

  /// Called at the start of spill management.
  virtual void onSpillStart(llvm::StringRef funcName, uint64_t budget,
                            size_t scheduleSize) {}

  /// Dead tensor removed from live set.
  virtual void onDeadRemoval(Operation *op, int64_t pos,
                             uint64_t occupiedAfter) {}

  /// Op processed: validation success, added to live set.
  virtual void onLiveAdded(Operation *op, int64_t pos, uint64_t opL1Usage,
                           int64_t lastUse, uint64_t occupiedAfter) {}

  /// OOM detected for op.
  virtual void onOOM(Operation *op, int64_t pos, uint64_t occupiedL1) {}

  /// Demotion attempt (L1 sharded -> L1 interleaved).
  virtual void onDemotion(Operation *op, int64_t pos, bool success,
                          uint64_t newL1Usage) {}

  /// Belady eviction: victim tensor spilled to DRAM.
  virtual void onEviction(Operation *victim, int64_t pos,
                          uint64_t freedBytes) {}

  /// Op exceeds budget alone -- self-spill to DRAM.
  virtual void onSelfSpill(Operation *op, int64_t pos) {}

  /// Consumer revalidation cascade after eviction.
  virtual void onRevalidationCascade(Operation *changed, Operation *consumer,
                                     bool outputChanged) {}

  /// Called at the end of spill management.
  virtual void onSpillEnd(size_t totalSpills, uint64_t finalOccupied,
                          size_t liveTensors) {}

  /// Access trace data (returns null for NullObject base).
  virtual const DecisionTrace *getDecisionTrace() const { return nullptr; }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1SPILLOBSERVER_H
