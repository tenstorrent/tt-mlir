// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRCONSTANTASFILL
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Constant as fill pass
//===----------------------------------------------------------------------===//

class TTIRConstantAsFillRewriter : public OpRewritePattern<ConstantOp> {
public:
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const final {
    auto resultTy = op.getResult().getType();
    auto empty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultTy.getShape(), resultTy.getElementType(),
        resultTy.getEncoding());
    auto operandConstraints = rewriter.getArrayAttr(SmallVector<Attribute>(
        1,
        rewriter.getAttr<OperandConstraintAttr>(OperandConstraint::AnyDevice)));
    rewriter.replaceOpWithNewOp<ttir::FillOp>(
        op, resultTy, empty, op.getValue(), operandConstraints);
    return success();
  }
};

class TTIRConstantAsFill
    : public impl::TTIRConstantAsFillBase<TTIRConstantAsFill> {
public:
  using impl::TTIRConstantAsFillBase<
      TTIRConstantAsFill>::TTIRConstantAsFillBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRConstantAsFillRewriter>(&getContext());
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
