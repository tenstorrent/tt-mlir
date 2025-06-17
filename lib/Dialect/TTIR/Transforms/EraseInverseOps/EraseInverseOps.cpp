// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;

  uint64_t
  countTms(ModuleOp module,
           const llvm::SmallPtrSet<mlir::BlockArgument, 4> &constParams) {
    uint64_t tmCount = 0;
    module.walk([&](Operation *op) {
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

  void runOnOperation() final {
    mlir::func::FuncOp funcOp;
    getOperation().walk([&](mlir::func::FuncOp op) {
      if (!op.isDeclaration()) {
        funcOp = op;
      }
    });

    if (!funcOp) {
      return;
    }
    auto constParams = ttmlir::utils::populateConstParams(funcOp);
    RewritePatternSet commuteAbovePatterns(&getContext());
    populateElementwiseCommuteAbovePatterns(&getContext(), commuteAbovePatterns,
                                            constParams);
    populateBroadcastCommuteAbovePatterns(&getContext(), commuteAbovePatterns,
                                          constParams);
    mlir::tt::ttir::PermuteOp::getCanonicalizationPatterns(commuteAbovePatterns,
                                                           &getContext());

    frozenCommuteAbovePatterns = std::move(commuteAbovePatterns);

    RewritePatternSet commuteBelowPatterns(&getContext());
    populateElementwiseCommuteBelowPatterns(&getContext(), commuteBelowPatterns,
                                            constParams);
    populateBroadcastCommuteBelowPatterns(&getContext(), commuteBelowPatterns,
                                          constParams);
    mlir::tt::ttir::PermuteOp::getCanonicalizationPatterns(commuteBelowPatterns,
                                                           &getContext());

    frozenCommuteBelowPatterns = std::move(commuteBelowPatterns);

    // We want to continue to run all the commute patterns until we have reached
    // a minumum number of TMs We know if we have reached a minimum if we
    // execute all possible downward patterns, and all possible upward patterns
    // and the number of TMs does not decrease When this happens the minimal
    // graph is at some state in between the graph before executing the downward
    // patterns and the graph after executing the upward patterns For now we
    // will pick whichever graph between the two that has the least number of
    // TMs - even though the true minimal graph might be in between

    while (true) {
      // We do not yet have a way of returning the beginning state of the graph
      // So we will return after we have commuted the TMs above at least once
      uint64_t startingTMCount = countTms(getOperation(), constParams);
      uint64_t minTmCount = startingTMCount;
      if (failed(applyCommuteAbovePatterns(getOperation()))) {
        return;
      }

      uint64_t afterCommuteAboveTMCount = countTms(getOperation(), constParams);
      if (afterCommuteAboveTMCount < minTmCount) {
        minTmCount = afterCommuteAboveTMCount;
      }

      if (failed(applyCommuteBelowPatterns(getOperation()))) {
        return;
      }

      uint64_t afterCommuteBelowTMCount = countTms(getOperation(), constParams);
      if (afterCommuteBelowTMCount < minTmCount) {
        minTmCount = afterCommuteBelowTMCount;
      }
      // If this is true then the minimal TM state has been reached at some
      // point during these two pattern applications
      if (minTmCount <= afterCommuteAboveTMCount &&
          minTmCount <= afterCommuteBelowTMCount) {
        // At this point the TMs are as far down the graph as they can be
        // If there were less TMs when they were as far up as they could be then
        // we must commute them up again.
        if (afterCommuteAboveTMCount < afterCommuteBelowTMCount) {
          if (failed(applyCommuteAbovePatterns(getOperation()))) {
            llvm_unreachable("applyCommuteAbovePatterns should not fail here");
          }
        }
        return;
      }
    }
  }

private:
  FrozenRewritePatternSet frozenCommuteAbovePatterns;
  FrozenRewritePatternSet frozenCommuteBelowPatterns;

  LogicalResult applyCommuteAbovePatterns(Operation *op) {
    if (failed(applyPatternsGreedily(op, frozenCommuteAbovePatterns))) {
      signalPassFailure();
      return failure();
    }
    return success();
  }

  LogicalResult applyCommuteBelowPatterns(Operation *op) {
    if (failed(applyPatternsGreedily(op, frozenCommuteBelowPatterns))) {
      signalPassFailure();
      return failure();
    }
    return success();
  }
};
} // namespace mlir::tt::ttir
