// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include <llvm/Support/LogicalResult.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;
  void runOnOperation() final {
    RewritePatternSet commutePatterns(&getContext());
    populateElementwiseCommutePatterns(&getContext(), commutePatterns);
    populateBroadcastCommutePatterns(&getContext(), commutePatterns);
    FrozenRewritePatternSet commutePatternSet(std::move(commutePatterns));

    // We want to commute all TMs upwards as much as possible so they are are
    // placed back to back Then we can erase back to back inverses. The
    // implemented folding patterns for TransposeOp, PermuteOp. and ReshapeOp
    // will automatically erase back to back inverses during the pass.
    // see TTIROps.cpp for the folding patterns.
    //
    // Because there are multiple TMs we wish to commute and erase, we must
    // continuously run the commute and erase patterns until the graph stops
    // changing. This is because erasing a pair of TMs may free up a path
    // for another pair of TMs to be erased.
    GreedyRewriteConfig rewriteConfig = GreedyRewriteConfig();
    bool changed = false;
    do {
      if (failed(applyPatternsGreedily(getOperation(), commutePatternSet,
                                       rewriteConfig, &changed))) {
        signalPassFailure();
        return;
      }
    } while (changed);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace mlir::tt::ttir
