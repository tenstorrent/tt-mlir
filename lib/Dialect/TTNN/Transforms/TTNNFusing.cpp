// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNConv2dWithActivation : public mlir::OpRewritePattern<Conv2dOp> {
  using TTNNConv2dWithActivation::OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    auto [reluOp, reluInput] = getReluOpAndReluInput(srcOp);

    mlir::StringAttr activation = rewriter.getStringAttr("relu");
    Conv2dConfigAttr conv2dConfigAttr =
        srcOp.getConv2dConfigAttr()
            ? srcOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());
    conv2dConfigAttr = conv2dConfigAttr.withActivation(activation);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(conv2dConfigAttr); });

    // Replace the relu op uses with either conv2d or reshape
    // depending on if reshape was present.
    rewriter.replaceAllUsesWith(reluOp, reluInput);

    return mlir::success();
  }

private:
  std::pair<ReluOp, mlir::Value> getReluOpAndReluInput(Conv2dOp srcOp) const {
    assert((ttmlir::utils::allUsers<ReshapeOp, ReluOp>(srcOp)) &&
           "Conv2d should have either Relu or Reshape as user.");

    if (ttmlir::utils::allUsers<ReluOp>(srcOp)) {
      return {mlir::cast<ReluOp>(*srcOp.getResult().getUsers().begin()),
              srcOp.getResult()};
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());
    ReluOp reluOp =
        mlir::cast<ReluOp>(*reshapeOp.getResult().getUsers().begin());

    return {reluOp, reshapeOp.getResult()};
  }

  bool isFusable(Conv2dOp srcOp) const {
    if (srcOp.getConv2dConfig() && srcOp.getConv2dConfig()->hasActivation()) {
      return false;
    }

    // Conv2d has multiple uses so we cannot fuse.
    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    // Conv2d only user is ReLU so we can fuse.
    if (ttmlir::utils::allUsers<ReluOp>(srcOp)) {
      return true;
    }

    // Since window flattening will add rehape after conv we need to check
    // if there is reshape right after conv2d.
    if (!ttmlir::utils::allUsers<ReshapeOp>(srcOp)) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    // If we want to fuse relu to conv we need to make sure that reshape
    // has only one user and that user is relu.
    return reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsers<ReluOp>(reshapeOp);
  }
};

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNConv2dWithActivation>(&getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
