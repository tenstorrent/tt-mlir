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
// Our backend supports implicit broadcast of operands, so explicit broadcast
// instructions are folded.
//
// For Example:
//
// %0 = tensor.empty() : tensor<512xf32>
// %1 = "ttir.broadcast"(%arg0, %0) (tensor<1xf32>, tensor<512xf32>) ->
// tensor<512xf32> %2 = tensor.empty() : tensor<512xf32> %3 = "ttir.maximum"(%1,
// %arg1, %2) (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) ->
// tensor<512xf32>
//
// After folding:
//
// %0 = tensor.empty() : tensor<512xf32>
// %1 = "ttir.maximum"(%arg0, %arg1, %0) (tensor<1xf32>, tensor<512xf32>,
// tensor<512xf32>) -> tensor<512xf32>
//===----------------------------------------------------------------------===//

class TTIRBroadcastFoldRewriter : public OpRewritePattern<BroadcastOp> {
public:
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const final {

    if (op->hasTrait<ImplicitBroadcastable::Trait>()) {
      // Only folding when the operation is marked to support implicit
      // broadcasting.
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    return failure();
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
