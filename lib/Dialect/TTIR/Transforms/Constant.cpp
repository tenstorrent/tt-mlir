// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

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
    ttmlir::utils::replaceOpWithNewDPSOp<FillOp>(rewriter, op, op.getType(),
                                                 op.getValue());

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
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

} // namespace mlir::tt::ttir
