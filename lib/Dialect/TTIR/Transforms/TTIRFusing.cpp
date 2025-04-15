// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Common/FusingCommon.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFUSING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

class TTIRConv2dWithBias
    : public ttmlir::utils::fusing::Conv2dAddPattern<Conv2dOp, AddOp> {
  using TTIRConv2dWithBias::Conv2dAddPattern::Conv2dAddPattern;

public:
  mlir::Value replaceConv2d(mlir::PatternRewriter &rewriter, Conv2dOp srcOp,
                            mlir::Value bias) const final {
    return rewriter.replaceOpWithNewOp<Conv2dOp>(
        srcOp, srcOp.getResult().getType(), srcOp.getInput(), srcOp.getWeight(),
        bias, srcOp.getOutput(), srcOp.getStride(), srcOp.getPadding(),
        srcOp.getDilation(), srcOp.getGroupsAttr(),
        srcOp.getFlattenedCompatInfo());
  }

  mlir::Value getAddResult(AddOp addOp) const final {
    return addOp.getResult(0);
  }
};

class TTIRFusingPass : public impl::TTIRFusingBase<TTIRFusingPass> {
public:
  using impl::TTIRFusingBase<TTIRFusingPass>::TTIRFusingBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRConv2dWithBias>(&getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttir
