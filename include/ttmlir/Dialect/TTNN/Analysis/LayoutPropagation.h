// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATION_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATION_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagationObserver.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <vector>

namespace mlir::tt::ttnn {

/// LayoutPropagation performs greedy (K=1) or beam search (K>1) layout
/// propagation over a function's operations in topological order. It determines
/// the best layout for each op's output by evaluating cross-products of input
/// candidate layouts and output hints, scoring via backend validation. After
/// propagation, it applies all decisions directly to the IR.
class LayoutPropagation {
public:
  /// An input candidate for one operand (public for observer access).
  using InputCandidate = LayoutPropagationObserver::InputCandidate;

  /// Construct a layout propagation instance for the given function.
  /// Uses the provided legal configs and device grid to evaluate candidates.
  /// When tensorTypePossibleLayouts is provided, reshard candidates are
  /// generated from all possible sharded layouts for each tensor type.
  LayoutPropagation(
      func::FuncOp func, ttcore::GridAttr deviceGrid,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts = nullptr,
      size_t beamWidth = 8,
      std::unique_ptr<LayoutPropagationObserver> observer = nullptr);

  /// Destructor defined in .cpp (observer is forward-declared).
  ~LayoutPropagation();

  /// Run layout propagation and apply all changes directly to the IR.
  void run();

  /// Accessors for testing and debugging.
  const llvm::DenseMap<Operation *, llvm::SmallVector<BeamCandidate, 0>> &
  getBeamState() const {
    return beamState;
  }
  const llvm::DenseMap<Operation *, size_t> &getFinalChoice() const {
    return finalChoice;
  }
  size_t getBeamWidth() const { return beamWidth; }

  /// Access the observer (always non-null; NullObject when tracing disabled).
  LayoutPropagationObserver *getObserver() { return observer.get(); }

private:
  func::FuncOp func;
  ttcore::GridAttr deviceGrid;
  const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs;
  const TensorTypeLayoutsMap *tensorTypePossibleLayouts;

  /// Per-op beam state: K candidates per op (K=1 for greedy).
  llvm::DenseMap<Operation *, llvm::SmallVector<BeamCandidate, 0>> beamState;

  /// Beam width (K=1 for greedy, K>1 for beam search).
  size_t beamWidth = 8;

  /// Final candidate choice per op (set by backward pass, used by applyToIR).
  /// Maps op -> index into beamState[op]. For K=1, always 0.
  llvm::DenseMap<Operation *, size_t> finalChoice;

  /// Observer (NullObject pattern: always non-null, no-op when tracing
  /// disabled).
  std::unique_ptr<LayoutPropagationObserver> observer;

  /// Process a single op. Returns top-K candidates sorted by score descending.
  llvm::SmallVector<BeamCandidate, 0> processOp(Operation *op);

  /// Build per-operand input candidate sets.
  /// For greedy (K=1): each operand gets {resolved_producer_layout}
  ///   + reshard candidates (if shouldExploreReshards).
  /// Returns one vector per operand.
  std::vector<std::vector<InputCandidate>> getInputCandidateSets(Operation *op);

  /// Generate reshard candidate layouts for a tensor type: sharded-to-sharded
  /// variants derived from the current layout.
  std::vector<TTNNLayoutAttr>
  generateReshardCandidates(RankedTensorType tensorType,
                            TTNNLayoutAttr currentLayout);

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
  size_t resolveForForkPoint(Operation *forkOp,
                             llvm::ArrayRef<Operation *> consumers);

  /// Validate that a reshard (ToMemoryConfigOp) from producerOutputLayout to
  /// reshardLayout is feasible via backend constraint validation.
  /// producerResultIdx specifies which result of the producer to use for the
  /// input shape (relevant for multi-output producers).
  bool validateReshard(Operation *consumerOp, Operation *producerOp,
                       TTNNLayoutAttr producerOutputLayout,
                       TTNNLayoutAttr reshardLayout,
                       size_t producerResultIdx = 0);

  /// Map from tensor-operand index (used in producerCandidateIndices) back to
  /// the actual defining op. Skips non-tensor operands.
  Operation *getProducerForOperandIdx(Operation *op, size_t tensorOperandIdx);

  /// Look up the chosen BeamCandidate for an op (using finalChoice index into
  /// beamState). Returns nullptr if op is not in beamState or beam is empty.
  const BeamCandidate *getChosenCandidate(Operation *op) const;

  /// Evaluate a single output hint against an input combination via backend
  /// validation. Returns a scored BeamCandidate on success, nullopt on failure.
  std::optional<BeamCandidate>
  evaluateHint(Operation *op, const OpConfig &hint, size_t hintIdx,
               const std::vector<TTNNLayoutAttr> &inputLayouts, bool anyReshard,
               const llvm::SmallVector<size_t> &producerCandidateIndices,
               const llvm::DenseMap<size_t, TTNNLayoutAttr> &reshardLayouts);

  /// Add L1-interleaved reshard fallbacks when the producer beam has sharded
  /// but no interleaved L1 candidates.
  void addL1InterleavedFallbacks(
      std::vector<InputCandidate> &candidates, Operation *op,
      const llvm::SmallVector<BeamCandidate, 0> *producerBeam,
      Operation *producerOp, TTNNLayoutAttr currentLayout, size_t resultIdx,
      size_t maxCandidates);

  /// Apply per-op input layout filters, removing candidates that the op
  /// cannot consume efficiently.
  void applyInputLayoutFilter(std::vector<InputCandidate> &candidates,
                              Operation *op, TTNNLayoutAttr currentLayout);

  /// Generate and add reshard candidates for one operand.
  void addReshardCandidates(
      std::vector<InputCandidate> &candidates, Operation *op, Value operand,
      TTNNLayoutAttr currentLayout, RankedTensorType tensorType,
      const llvm::SmallVector<BeamCandidate, 0> *producerBeam,
      Operation *producerOp, size_t resultIdx, size_t maxCandidates);

  /// Disable deallocate_activation for conv ops with multi-use inputs.
  void fixupConvDeallocate();

  /// Insert to_memory_config to DRAM for func.return operands that are in L1.
  void insertReturnDramSpills();
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATION_H
