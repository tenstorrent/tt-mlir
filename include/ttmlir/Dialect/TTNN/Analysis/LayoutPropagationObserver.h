// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATIONOBSERVER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATIONOBSERVER_H

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <vector>

namespace mlir::tt::ttnn {

class DecisionTrace;

/// Observer interface for LayoutPropagation decisions.
/// NullObject pattern: all methods have empty default implementations.
/// Concrete observers (DecisionTraceObserver) override the methods they care
/// about. This eliminates all `if (trace) { ... }` conditionals in
/// LayoutPropagation.cpp.
class LayoutPropagationObserver {
public:
  virtual ~LayoutPropagationObserver() = default;

  /// Called at the start of layout propagation.
  virtual void onStart(llvm::StringRef funcName, size_t beamWidth) {}

  /// An input candidate for one operand of an op.
  struct InputCandidate {
    TTNNLayoutAttr layout;
    size_t producerCandidateIndex = 0;
    bool isReshard = false;
  };

  /// After building input candidates and output hints for an op.
  virtual void
  onOpSetup(Operation *op,
            const std::vector<std::vector<InputCandidate>> &inputSets,
            const OutputHints &hints, size_t crossProductSize) {}

  /// Each evaluation attempt (valid or failed). Hot path.
  virtual void onEvaluation(Operation *op, const OpConfig &hint,
                            size_t hintIdx,
                            llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                            bool valid, const BeamCandidate *candidate,
                            llvm::StringRef failureReason) {}

  /// After beam sort+trim for an op.
  virtual void onBeamResult(Operation *op,
                            llvm::ArrayRef<BeamCandidate> beam,
                            bool usedDramFallback) {}

  /// Fork resolution in backward pass.
  virtual void onForkResolved(Operation *producer, size_t chosenIdx,
                              llvm::ArrayRef<Operation *> consumers) {}

  /// Producer->consumer edge after final choices.
  virtual void onEdge(Operation *producer, Operation *consumer,
                      size_t operandIdx, bool hasReshard,
                      TTNNLayoutAttr reshardLayout) {}

  /// Final chosen candidate per op.
  virtual void onFinalChoice(Operation *op, size_t opIndex,
                             const BeamCandidate &chosen) {}

  /// Called at the end of layout propagation.
  virtual void onEnd(size_t totalOps) {}

  /// Info bundle for an in-place (zero-result) op.
  struct InplaceOpInfo {
    Operation *op;
    /// Per tensor operand: (operandIdx, currentLayout, producerOp).
    /// producerOp may be null for func args.
    struct OperandInfo {
      size_t operandIdx;
      TTNNLayoutAttr layout; // from IR encoding
      Operation *producerOp; // may be null
    };
    llvm::SmallVector<OperandInfo> operands;
  };

  /// Called for zero-result (in-place) ops that consume tracked tensors.
  virtual void onInplaceOp(const InplaceOpInfo &info) {}

  /// Access trace data (returns null for NullObject base).
  virtual const DecisionTrace *getDecisionTrace() const { return nullptr; }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTPROPAGATIONOBSERVER_H
