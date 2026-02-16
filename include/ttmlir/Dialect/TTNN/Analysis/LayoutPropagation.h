// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATION_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATION_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace mlir::tt::ttnn {

/// LayoutPropagation performs greedy (K=1) or beam search (K>1) layout
/// propagation over a function's operations in topological order. It determines
/// the best layout for each op's output by evaluating cross-products of input
/// candidate layouts and output hints, scoring via backend validation. After
/// propagation, it applies all decisions directly to the IR.
class LayoutPropagation {
public:
  LayoutPropagation(
      func::FuncOp func, ttcore::GridAttr deviceGrid,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts = nullptr,
      size_t beamWidth = 8);

  /// Run layout propagation and apply all changes directly to the IR.
  void run();

  /// Accessors for testing and debugging.
  const llvm::DenseMap<Operation *, llvm::SmallVector<BeamCandidate>> &
  getBeamState() const {
    return beamState;
  }
  const llvm::DenseMap<Operation *, size_t> &getFinalChoice() const {
    return finalChoice;
  }
  size_t getBeamWidth() const { return beamWidth; }

private:
  func::FuncOp func;
  ttcore::GridAttr deviceGrid;
  const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs;
  const TensorTypeLayoutsMap *tensorTypePossibleLayouts;

  /// Per-op beam state: K candidates per op (K=1 for greedy).
  llvm::DenseMap<Operation *, llvm::SmallVector<BeamCandidate>> beamState;

  /// Beam width (K=1 for greedy, K>1 for beam search).
  size_t beamWidth = 8;

  /// Final candidate choice per op (set by backward pass, used by applyToIR).
  /// Maps op -> index into beamState[op]. For K=1, always 0.
  llvm::DenseMap<Operation *, size_t> finalChoice;

  /// Process a single op. Returns top-K candidates sorted by score descending.
  llvm::SmallVector<BeamCandidate> processOp(Operation *op);

  /// Build per-operand input candidate sets.
  /// For greedy (K=1): each operand gets {resolved_producer_layout}
  ///   + reshard candidates (if shouldExploreReshards).
  /// Returns one vector per operand.
  struct InputCandidate {
    TTNNLayoutAttr layout;
    size_t producerCandidateIndex = 0;
    bool isReshard = false;
  };
  std::vector<std::vector<InputCandidate>> getInputCandidateSets(
      Operation *op);

  /// Generate reshard candidate layouts for a tensor type: DRAM interleaved
  /// and L1 interleaved variants derived from the current layout.
  std::vector<TTNNLayoutAttr> generateReshardCandidates(
      RankedTensorType tensorType, TTNNLayoutAttr currentLayout);

  /// Create a DRAM interleaved fallback layout for an op.
  TTNNLayoutAttr getDRAMInterleavedFallback(Operation *op);

  /// Apply all resolved configs to IR.
  void applyToIR();

  /// Apply a single op's chosen config to its result type, DPS operand,
  /// interfaces, and op-specific attrs.
  void applyOpConfig(Operation *op, const BeamCandidate &candidate);

  /// Insert ToLayoutOp for a reshard edge.
  void insertReshardOp(Operation *consumerOp, size_t operandIndex,
                        TTNNLayoutAttr reshardLayout);

  /// Update function return types to match modified IR.
  void updateFunctionReturnTypes();

  /// Backward pass: consolidate beam at fork points.
  /// Only runs when beamWidth > 1.
  void consolidateBeam();

  /// Resolve fork point: pick the producer candidate that minimizes
  /// total reshard count across all consumers.
  size_t resolveForForkPoint(Operation *forkOp);

  /// Map from tensor-operand index (used in producerCandidateIndices) back to
  /// the actual defining op. Skips non-tensor operands.
  Operation *getProducerForOperandIdx(Operation *op, size_t tensorOperandIdx);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATION_H
