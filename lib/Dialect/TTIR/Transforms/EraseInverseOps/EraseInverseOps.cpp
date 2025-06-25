// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/MLIRContext.h"
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
      if (!valueTracesToConstantArgs(op->getResult(0), constParams)) {
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

    auto constParams = mlir::tt::getConstsAndParams(funcOps[0]);
    frozenCommuteAbovePatterns =
        getCommuteRewritePatternSet<CommuteDirection::UPWARDS>(constParams);
    frozenCommuteBelowPatterns =
        getCommuteRewritePatternSet<CommuteDirection::DOWNWARDS>(constParams);
    for (auto funcOp : funcOps) {
      while (true) {
        // We do not yet have a way of returning the beginning state of the
        // graph So we will return after we have commuted the TMs above at least
        // once
        uint64_t startingTMCount = countTms(funcOp, constParams);
        uint64_t minTmCount = startingTMCount;
        applyCommuteAbovePatterns(funcOp);

        uint64_t afterCommuteAboveTMCount = countTms(funcOp, constParams);
        if (afterCommuteAboveTMCount < minTmCount) {
          minTmCount = afterCommuteAboveTMCount;
        }

        applyCommuteBelowPatterns(funcOp);

        uint64_t afterCommuteBelowTMCount = countTms(funcOp, constParams);
        if (afterCommuteBelowTMCount < minTmCount) {
          minTmCount = afterCommuteBelowTMCount;
        }
        // If this is true then the minimal TM state has been reached at some
        // point during these two pattern applications
        if (minTmCount <= afterCommuteAboveTMCount &&
            minTmCount <= afterCommuteBelowTMCount) {
          // At this point the TMs are as far down the graph as they can be
          // If there were less TMs when they were as far up as they could be
          // then we must commute them up again.
          if (afterCommuteAboveTMCount <= afterCommuteBelowTMCount) {
            applyCommuteAbovePatterns(funcOp);
          }
          break;
        }
      }
    }
  }

private:
  FrozenRewritePatternSet frozenCommuteAbovePatterns;
  FrozenRewritePatternSet frozenCommuteBelowPatterns;

  template <CommuteDirection commuteDirection>
  RewritePatternSet getCommuteRewritePatternSet(
      llvm::SmallPtrSet<mlir::BlockArgument, 4> &constParams) {
    RewritePatternSet patterns(&getContext());
    populateElementwiseCommutePatterns<commuteDirection>(&getContext(),
                                                         patterns, constParams);
    populateBroadcastCommutePatterns<commuteDirection>(&getContext(), patterns,
                                                       constParams);
    populateTTIRTMFusionPatterns(&getContext(), patterns);
    return patterns;
  }

  void applyCommuteAbovePatterns(Operation *op) {
    if (failed(applyPatternsGreedily(op, frozenCommuteAbovePatterns))) {
      signalPassFailure();
    }
  }

  void applyCommuteBelowPatterns(Operation *op) {
    if (failed(applyPatternsGreedily(op, frozenCommuteBelowPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::tt::ttir
