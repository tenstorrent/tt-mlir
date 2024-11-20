// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRRESHAPEFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Constant as fill pass
//===----------------------------------------------------------------------===//

class TTIRReshapeFoldingRewriter : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getType() == op->getOperand(0).getType()) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    return failure();
  }
};

class TTIRReshapeFold : public impl::TTIRReshapeFoldBase<TTIRReshapeFold> {
public:
  using impl::TTIRReshapeFoldBase<TTIRReshapeFold>::TTIRReshapeFoldBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRReshapeFoldingRewriter>(&getContext());
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
