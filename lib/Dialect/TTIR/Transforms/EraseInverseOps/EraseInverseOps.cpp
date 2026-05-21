// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/ConstevalForwardAnalysis.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// Check if any op in the funcOp has TTIR_FlattenedCompatInfoAttr
bool hasFlattenedCompatInfoAttr(Operation *op) {
  bool hasAttr = false;
  op->walk([&](Operation *innerOp) {
    for (auto attr : innerOp->getAttrs()) {
      if (llvm::isa<ttir::FlattenedCompatInfoAttr>(attr.getValue())) {
        hasAttr = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return hasAttr;
}

uint64_t countTms(Operation *op, ConstevalForwardAnalysis &analysis) {
  uint64_t tmCount = 0;
  op->walk([&](Operation *op) {
    if (op->hasTrait<tt::ttir::TensorManipulation::Trait>()) {
      // If the TM lies on a constevalable subgraph then we will not count it
      // as it will be removed from the main graph.
      if (!analysis.valueTracesToConstantArgs(op->getResult(0))) {
        tmCount++;
      }
    }
  });
  return tmCount;
}

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;

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

    for (auto funcOp : funcOps) {
      // If no ops were flattened, we don't expect any inverse TMs.
      // TTIR_FlattenedCompatInfoAttr (unless force flag is set)
      if (!force.getValue() && !hasFlattenedCompatInfoAttr(funcOp)) {
        continue;
      }

      ConstevalForwardAnalysis analysis(funcOp);
      FrozenRewritePatternSet commuteAbovePatterns(
          getCommuteRewritePatternSet<CommuteDirection::UPWARDS>(&analysis));
      FrozenRewritePatternSet commuteBelowPatterns(
          getCommuteRewritePatternSet<CommuteDirection::DOWNWARDS>(&analysis));

#ifdef TTMLIR_ENABLE_DEBUG_LOGS
      const int64_t nonConstevalableTMsBefore = countTms(funcOp, analysis);
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
        applyCommuteAbovePatterns(funcOp, commuteAbovePatterns, analysis);
        uint64_t afterCommuteAboveTMCount = countTms(funcOp, analysis);

        applyCommuteBelowPatterns(funcOp, commuteBelowPatterns, analysis);
        uint64_t afterCommuteBelowTMCount = countTms(funcOp, analysis);

        // If the number of TM is the same as in the previous iteration, we have
        // converged.
        if (afterCommuteAboveTMCount == previousAfterCommuteAboveTMCount &&
            afterCommuteBelowTMCount == previousAfterCommuteBelowTMCount) {

          // If the number of TM was smaller before commuting below, commute
          // above one more time.
          if (afterCommuteAboveTMCount < afterCommuteBelowTMCount) {
            applyCommuteAbovePatterns(funcOp, commuteAbovePatterns, analysis);
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
      const int64_t nonConstevalableTMsAfter = countTms(funcOp, analysis);
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
  template <CommuteDirection commuteDirection>
  RewritePatternSet
  getCommuteRewritePatternSet(ConstevalForwardAnalysis *analysis) {
    RewritePatternSet patterns(&getContext());
    populateElementwiseCommutePatterns<commuteDirection>(&getContext(),
                                                         patterns, analysis);
    // Elementwise downwards can move reshapes onto consteval paths, creating
    // broadcast->reshape matches; keep broadcast-upwards in both sets so
    // elementwise-upwards does not race and pull those reshapes back first.
    populateBroadcastCommutePatterns<CommuteDirection::UPWARDS>(
        &getContext(), patterns, analysis);
    populateConcatCommutePatterns<commuteDirection>(&getContext(), patterns,
                                                    analysis);
    populateSliceCommutePatterns<commuteDirection>(&getContext(), patterns,
                                                   analysis);
    populateReduceCommutePatterns<commuteDirection>(&getContext(), patterns,
                                                    analysis);
    populateRMSNormCommutePatterns<commuteDirection>(&getContext(), patterns,
                                                     analysis);
    populateSoftmaxCommutePatterns<commuteDirection>(&getContext(), patterns,
                                                     analysis);

    populateTTIRTMFusionPatterns(&getContext(), patterns);
    return patterns;
  }

  void applyCommuteAbovePatterns(Operation *op,
                                 const FrozenRewritePatternSet &patterns,
                                 ConstevalForwardAnalysis &analysis) {
    if (enableCommuteUpwards.getValue()) {
      GreedyRewriteConfig config;
      config.setListener(&analysis);
      if (failed(applyPatternsGreedily(op, patterns, config))) {
        signalPassFailure();
      }
    }
  }

  void applyCommuteBelowPatterns(Operation *op,
                                 const FrozenRewritePatternSet &patterns,
                                 ConstevalForwardAnalysis &analysis) {
    if (enableCommuteDownwards.getValue()) {
      GreedyRewriteConfig config;
      config.setListener(&analysis);
      if (failed(applyPatternsGreedily(op, patterns, config))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace mlir::tt::ttir
