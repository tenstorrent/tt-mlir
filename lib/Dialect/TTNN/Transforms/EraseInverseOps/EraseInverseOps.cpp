// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNERASEINVERSEOPS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

uint64_t countTms(Operation *op) {
  uint64_t tmCount = 0;
  op->walk([&](Operation *innerOp) {
    if (isa<ttnn::PermuteOp, ttnn::ReshapeOp, ttnn::TransposeOp>(innerOp)) {
      tmCount++;
    }
  });
  return tmCount;
}

class TTNNEraseInverseOps
    : public impl::TTNNEraseInverseOpsBase<TTNNEraseInverseOps> {
public:
  using impl::TTNNEraseInverseOpsBase<
      TTNNEraseInverseOps>::TTNNEraseInverseOpsBase;

  void runOnOperation() final {

    SmallVector<mlir::func::FuncOp> funcOps;
    getOperation().walk([&](mlir::func::FuncOp op) {
      if (!op.isDeclaration()) {
        funcOps.push_back(op);
      }
    });

    if (funcOps.empty()) {
      return;
    }

    commuteAbovePatterns =
        getCommuteRewritePatternSet<CommuteDirection::UPWARDS>();
    commuteBelowPatterns =
        getCommuteRewritePatternSet<CommuteDirection::DOWNWARDS>();
    for (auto funcOp : funcOps) {

#ifdef TTMLIR_ENABLE_DEBUG_LOGS
      const int64_t nonConstevalableTMsBefore = countTms(funcOp);
#endif

      uint64_t maxIterationsValue = maxIterations.getValue();
      uint64_t previousAfterCommuteAboveTMCount =
          std::numeric_limits<uint64_t>::max();
      uint64_t previousAfterCommuteBelowTMCount =
          std::numeric_limits<uint64_t>::max();

      uint64_t iter = 0;
      // The number of TM is expected to converge before maxIterations (default:
      // 100) is reached.
      for (; iter < maxIterationsValue; ++iter) {
        // We do not yet have a way of returning the beginning state of the
        // graph So we will return after we have commuted the TMs above at least
        // once
        applyCommuteAbovePatterns(funcOp);
        uint64_t afterCommuteAboveTMCount = countTms(funcOp);

        applyCommuteBelowPatterns(funcOp);
        uint64_t afterCommuteBelowTMCount = countTms(funcOp);

        // If the number of TM is the same as in the previous iteration, we have
        // converged.
        if (afterCommuteAboveTMCount == previousAfterCommuteAboveTMCount &&
            afterCommuteBelowTMCount == previousAfterCommuteBelowTMCount) {

          // If the number of TM was smaller before commuting below, commute
          // above one more time.
          if (afterCommuteAboveTMCount < afterCommuteBelowTMCount) {
            applyCommuteAbovePatterns(funcOp);
          }
          break;
        }
        previousAfterCommuteAboveTMCount = afterCommuteAboveTMCount;
        previousAfterCommuteBelowTMCount = afterCommuteBelowTMCount;
      }
      if (iter == maxIterationsValue) {
        emitError(funcOp.getLoc())
            << "EraseInverseOps: TM count has not converged after "
            << maxIterationsValue << " iterations.";
        signalPassFailure();
        return;
      }

#ifdef TTMLIR_ENABLE_DEBUG_LOGS
      const int64_t nonConstevalableTMsAfter = countTms(funcOp);
#endif
      TTMLIR_DEBUG(ttmlir::LogComponent::General,
                   "Function: {} | Num TMs on the activation paths before "
                   "EraseInverseOps: {}, "
                   "num TMs on the activation paths after EraseInverseOps: {}, "
                   "total removed: {}",
                   funcOp.getName(), nonConstevalableTMsBefore,
                   nonConstevalableTMsAfter,
                   nonConstevalableTMsBefore - nonConstevalableTMsAfter);
    }
  }

private:
  FrozenRewritePatternSet commuteAbovePatterns;
  FrozenRewritePatternSet commuteBelowPatterns;

  template <CommuteDirection commuteDirection>
  RewritePatternSet getCommuteRewritePatternSet() {
    RewritePatternSet patterns(&getContext());
    populateElementwiseCommutePatterns<commuteDirection>(&getContext(),
                                                         patterns);
    // // Elementwise downwards can move reshapes onto consteval paths, creating
    // // broadcast->reshape matches; keep broadcast-upwards in both sets so
    // // elementwise-upwards does not race and pull those reshapes back first.
    // populateBroadcastCommutePatterns<CommuteDirection::UPWARDS>(&getContext(),
    //                                                             patterns);
    // populateConcatCommutePatterns<commuteDirection>(&getContext(), patterns);
    // populateSliceCommutePatterns<commuteDirection>(&getContext(), patterns);
    // populateReduceCommutePatterns<commuteDirection>(&getContext(), patterns);
    // populateRMSNormCommutePatterns<commuteDirection>(&getContext(),
    // patterns);
    // populateSoftmaxCommutePatterns<commuteDirection>(&getContext(),
    // patterns);
    return patterns;
  }

  void applyCommuteAbovePatterns(Operation *op) {
    if (enableCommuteUpwards.getValue()) {
      if (failed(applyPatternsGreedily(op, commuteAbovePatterns))) {
        signalPassFailure();
      }
    }
  }

  void applyCommuteBelowPatterns(Operation *op) {
    if (enableCommuteDownwards.getValue()) {
      if (failed(applyPatternsGreedily(op, commuteBelowPatterns))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace mlir::tt::ttnn
