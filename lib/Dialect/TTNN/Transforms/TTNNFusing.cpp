// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Common/FusingCommon.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNConv2dWithBias
    : public ttmlir::utils::fusing::Conv2dAddPattern<Conv2dOp, AddOp> {
  using TTNNConv2dWithBias::Conv2dAddPattern::Conv2dAddPattern;

public:
  mlir::Value replaceConv2d(mlir::PatternRewriter &rewriter, Conv2dOp srcOp,
                            mlir::Value bias) const final {
    return rewriter.replaceOpWithNewOp<Conv2dOp>(
        srcOp, srcOp.getResult().getType(), srcOp.getInput(), srcOp.getWeight(),
        bias, srcOp.getDevice(), srcOp.getInChannels(), srcOp.getOutChannels(),
        srcOp.getBatchSize(), srcOp.getInputHeight(), srcOp.getInputWidth(),
        srcOp.getKernelSize(), srcOp.getStride(), srcOp.getPadding(),
        srcOp.getDilation(), srcOp.getGroups(), srcOp.getConv2dConfigAttr());
  }
};

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNConv2dWithBias>(&getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
