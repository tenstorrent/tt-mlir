// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRBROADCASTFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Broadcast Folding pass
//===----------------------------------------------------------------------===//

class TTIRBroadcastFoldRewriter : public OpRewritePattern<BroadcastOp> {
public:
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const final {

    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  }
};

class TTIRBroadcastFold
    : public impl::TTIRBroadcastFoldBase<TTIRBroadcastFold> {
public:
  using impl::TTIRBroadcastFoldBase<TTIRBroadcastFold>::TTIRBroadcastFoldBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRBroadcastFoldRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir
