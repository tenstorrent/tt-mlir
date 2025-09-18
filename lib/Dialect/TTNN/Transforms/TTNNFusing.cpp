// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

template <typename ActivationOp>
class TTNNConv2dWithActivation : public mlir::OpRewritePattern<Conv2dOp> {
  using TTNNConv2dWithActivation::OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp = getActivationOp(srcOp);
    Value activationInput = activationOp.getInput();

    auto activation = getActivationOpType(rewriter);
    ttcore::DataType weightDtype = ttcore::elementTypeToDataType(
        srcOp.getWeight().getType().getElementType());
    Conv2dConfigAttr conv2dConfigAttr =
        srcOp.getConv2dConfigAttr()
            ? srcOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());
    conv2dConfigAttr = conv2dConfigAttr.withActivation(UnaryOpType::Relu)
                           .withWeightsDtype(weightDtype);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(conv2dConfigAttr); });

    // Replace the activation op uses with either conv2d or reshape
    // depending on if reshape was present.
    rewriter.replaceAllUsesWith(activationOp, activationInput);

    return mlir::success();
  }

private:
  ActivationOp getActivationOp(Conv2dOp srcOp) const {
    assert((ttmlir::utils::allUsersOfType<ReshapeOp, ActivationOp>(srcOp)) &&
           "Conv2d should have either activation or Reshape as user.");

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    assert(reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp) &&
           "Reshape should have only one user and that user should be "
           "activation.");
    return mlir::cast<ActivationOp>(*reshapeOp.getResult().getUsers().begin());
  }

  ttnn::UnaryOpType getActivationOpType(mlir::PatternRewriter &rewriter) const {
    if constexpr (std::is_same_v<ActivationOp, ReluOp>) {
      return ttnn::UnaryOpType::Relu;
    } else if constexpr (std::is_same_v<ActivationOp, Relu6Op>) {
      return ttnn::UnaryOpType::Relu6;
    } else if constexpr (std::is_same_v<ActivationOp, SiluOp>) {
      return ttnn::UnaryOpType::Silu;
    } else {
      static_assert(ttmlir::utils::always_false<ActivationOp>(),
                    "Unsupported activation op");
    }

    bool isFusable(Conv2dOp srcOp) const {
      if (srcOp.getConv2dConfig() && srcOp.getConv2dConfig()->hasActivation()) {
        return false;
      }

      // Conv2d has multiple uses so we cannot fuse.
      if (!srcOp.getResult().hasOneUse()) {
        return false;
      }

      // Conv2d only user is activation so we can fuse.
      if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
        return true;
      }

      // Since window flattening will add rehape after conv we need to check
      // if there is reshape right after conv2d.
      if (!ttmlir::utils::allUsersOfType<ReshapeOp>(srcOp)) {
        return false;
      }

      ReshapeOp reshapeOp =
          mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

      // If we want to fuse activation to conv we need to make sure that reshape
      // has only one user and that user is activation.
      return reshapeOp.getResult().hasOneUse() &&
             ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp);
    }
  };

  class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
  public:
    using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

    void runOnOperation() final {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNConv2dWithActivation<ReluOp>,
                   TTNNConv2dWithActivation<Relu6Op>,
                   TTNNConv2dWithActivation<SiluOp>>(&getContext());
      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
  };
} // namespace

} // namespace mlir::tt::ttnn
