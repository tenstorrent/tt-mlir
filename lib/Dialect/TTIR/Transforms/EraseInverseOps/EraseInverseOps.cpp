// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

uint64_t
countTms(Operation *op,
         const llvm::SmallPtrSet<mlir::BlockArgument, 4> &constParams) {
  uint64_t tmCount = 0;
  op->walk([&](Operation *op) {
    if (isa<TransposeOp, PermuteOp, ReshapeOp>(op)) {
      // If the TM lies on a constevalable subgraph then we will not count it
      // as it will be removed from the main graph.
      if (!valueTracesToConstantArgs(op->getResult(0))) {
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

    auto constParams = ttcore::getConstsAndParams(funcOps[0]);
    commuteAbovePatterns =
        getCommuteRewritePatternSet<CommuteDirection::UPWARDS>(constParams);
    commuteBelowPatterns =
        getCommuteRewritePatternSet<CommuteDirection::DOWNWARDS>(constParams);
    for (auto funcOp : funcOps) {
      // Surround with ifdef as this would otherwise unnecessarily count TMs
#ifdef TTMLIR_ENABLE_DEBUG_LOGS
      const int64_t nonConstevalableTMsBefore = countTms(funcOp, constParams);
#endif

      // If the maxIterations is 0, then the loop will run until no more TMs
      // will be removed from the activation paths, or indefinetly if there
      // is a bug in the algorithm and/or halting condition.
      // If the maxIterations > 0 then the loop will run until no more TMs
      // will be removed from the activation paths, or until the maxIterations
      // is reached.
      uint64_t maxIterationsValue = maxIterations.getValue();
      uint64_t numConsecutiveCountIncreases = 0;
      auto loopCondition = [maxIterationsValue](uint64_t iter) {
        return maxIterationsValue == 0 ? true : iter < maxIterationsValue;
      };

      uint64_t previousAfterCommuteAboveTMCount =
          std::numeric_limits<uint64_t>::max();
      uint64_t previousAfterCommuteBelowTMCount =
          std::numeric_limits<uint64_t>::max();
      for (uint64_t iter = 0; loopCondition(iter); ++iter) {
        // We do not yet have a way of returning the beginning state of the
        // graph So we will return after we have commuted the TMs above at least
        // once
        applyCommuteAbovePatterns(funcOp);
        uint64_t afterCommuteAboveTMCount = countTms(funcOp, constParams);

        applyCommuteBelowPatterns(funcOp);
        uint64_t afterCommuteBelowTMCount = countTms(funcOp, constParams);

        if (afterCommuteAboveTMCount == previousAfterCommuteAboveTMCount &&
            afterCommuteBelowTMCount == previousAfterCommuteBelowTMCount) {

          if (afterCommuteAboveTMCount < afterCommuteBelowTMCount) {
            applyCommuteAbovePatterns(funcOp);
          }
          break;
        }
        if (afterCommuteAboveTMCount > previousAfterCommuteAboveTMCount ||
            afterCommuteBelowTMCount > previousAfterCommuteBelowTMCount) {
          numConsecutiveCountIncreases++;
        } else {
          numConsecutiveCountIncreases = 0;
        }

        if (numConsecutiveCountIncreases == 10) {
          emitError(funcOp.getLoc())
              << "TM count has increased for 10 consecutive iterations.";
          signalPassFailure();
          return;
        }

        previousAfterCommuteAboveTMCount = afterCommuteAboveTMCount;
        previousAfterCommuteBelowTMCount = afterCommuteBelowTMCount;
      }
      // Surround with ifdef as this would otherwise unnecessarily count TMs
#ifdef TTMLIR_ENABLE_DEBUG_LOGS
      const int64_t nonConstevalableTMsAfter = countTms(funcOp, constParams);
      TTMLIR_DEBUG(ttmlir::LogComponent::General,
                   "Function: {} | Num TMs on the activation paths before "
                   "EraseInverseOps: {}, "
                   "num TMs on the activation paths after EraseInverseOps: {}, "
                   "total removed: {}",
                   funcOp.getName(), nonConstevalableTMsBefore,
                   nonConstevalableTMsAfter,
                   nonConstevalableTMsBefore - nonConstevalableTMsAfter);
#endif
    }
  }

private:
  FrozenRewritePatternSet commuteAbovePatterns;
  FrozenRewritePatternSet commuteBelowPatterns;

  template <CommuteDirection commuteDirection>
  RewritePatternSet getCommuteRewritePatternSet(
      llvm::SmallPtrSet<mlir::BlockArgument, 4> &constParams) {
    RewritePatternSet patterns(&getContext());
    populateElementwiseCommutePatterns<commuteDirection>(&getContext(),
                                                         patterns);
    populateBroadcastCommutePatterns<commuteDirection>(&getContext(), patterns);
    populateConcatCommutePatterns<commuteDirection>(&getContext(), patterns);

    populateTTIRTMFusionPatterns(&getContext(), patterns);
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
} // namespace mlir::tt::ttir
