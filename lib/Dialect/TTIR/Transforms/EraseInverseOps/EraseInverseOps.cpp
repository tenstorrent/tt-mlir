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

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;

  uint64_t countTms(ModuleOp module) {
    uint64_t tmCount = 0;
    module.walk([&](Operation *op) {
      if (isa<TransposeOp, PermuteOp, ReshapeOp>(op)) {
        tmCount++;
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

    RewritePatternSet commuteAbovePatterns(&getContext());
    populateElementwiseCommuteAbovePatterns(&getContext(), commuteAbovePatterns,
                                            funcOp);
    populateBroadcastCommuteAbovePatterns(&getContext(), commuteAbovePatterns,
                                          funcOp);
    mlir::tt::ttir::PermuteOp::getCanonicalizationPatterns(commuteAbovePatterns,
                                                           &getContext());

    frozenCommuteAbovePatterns = std::move(commuteAbovePatterns);

    RewritePatternSet commuteBelowPatterns(&getContext());
    populateElementwiseCommuteBelowPatterns(&getContext(), commuteBelowPatterns,
                                            funcOp);
    populateBroadcastCommuteBelowPatterns(&getContext(), commuteBelowPatterns,
                                          funcOp);
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

    // uint64_t startingTMCount = countTms(getOperation());
    // uint64_t currentTMCount = startingTMCount;
    // uint64_t lastTMCount = std::numeric_limits<uint64_t>::max();

    enum MinimalTMState { STARTING, AFTER_COMMUTE_ABOVE, AFTER_COMMUTE_BELOW };

    while (true) {
      MinimalTMState minTmState = STARTING;
      uint64_t startingTMCount = countTms(getOperation());
      if (failed(applyCommuteAbovePatterns(getOperation()))) {
        return;
      }

      uint64_t afterCommuteAboveTMCount = countTms(getOperation());
      if (afterCommuteAboveTMCount < startingTMCount) {
        minTmState = AFTER_COMMUTE_ABOVE;
      }

      if (failed(applyCommuteBelowPatterns(getOperation()))) {
        return;
      }

      uint64_t afterCommuteBelowTMCount = countTms(getOperation());
      if (afterCommuteBelowTMCount < startingTMCount) {
        minTmState = AFTER_COMMUTE_BELOW;
      }
      // If this is true then the minimal TM state has been reached at some
      // point during these two pattern applications
      if (startingTMCount <= afterCommuteBelowTMCount) {
        if (minTmState == STARTING) {
          // We are already at the minimum TM count graph
          return;
        }
        (void)applyCommuteBelowPatterns(getOperation());
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
