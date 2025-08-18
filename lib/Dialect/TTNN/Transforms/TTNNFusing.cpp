// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsInterfaces.h"
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

    ReluOp reluOp = getReluOp(srcOp);
    Value reluInput = reluOp.getInput();

    mlir::StringAttr activation = rewriter.getStringAttr("relu");
    ttcore::DataType weightDtype = ttcore::elementTypeToDataType(
        srcOp.getWeight().getType().getElementType());
    Conv2dConfigAttr conv2dConfigAttr =
        srcOp.getConv2dConfigAttr()
            ? srcOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::getEmpty(rewriter.getContext());
    conv2dConfigAttr = conv2dConfigAttr.withActivation(activation)
                           .withWeightsDtype(weightDtype);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(conv2dConfigAttr); });

    // Replace the relu op uses with either conv2d or reshape
    // depending on if reshape was present.
    rewriter.replaceAllUsesWith(reluOp, reluInput);

    return mlir::success();
  }

private:
  ReluOp getReluOp(Conv2dOp srcOp) const {
    assert((ttmlir::utils::allUsersOfType<ReshapeOp, ReluOp>(srcOp)) &&
           "Conv2d should have either Relu or Reshape as user.");

    if (ttmlir::utils::allUsersOfType<ReluOp>(srcOp)) {
      return mlir::cast<ReluOp>(*srcOp.getResult().getUsers().begin());
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    assert(reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ReluOp>(reshapeOp) &&
           "Reshape should have only one user and that user should be relu.");
    return mlir::cast<ReluOp>(*reshapeOp.getResult().getUsers().begin());
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
    if (ttmlir::utils::allUsersOfType<ReluOp>(srcOp)) {
      return true;
    }

    // Since window flattening will add rehape after conv we need to check
    // if there is reshape right after conv2d.
    if (!ttmlir::utils::allUsersOfType<ReshapeOp>(srcOp)) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    // If we want to fuse relu to conv we need to make sure that reshape
    // has only one user and that user is relu.
    return reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ReluOp>(reshapeOp);
  }
};
} // namespace

namespace {
class TTNNBinaryOpInputsActivation
    : public mlir::OpInterfaceRewritePattern<ElementwiseBinary> {
  using TTNNBinaryOpInputsActivation::OpInterfaceRewritePattern<
      ElementwiseBinary>::OpInterfaceRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ElementwiseBinary binaryOp,
                  mlir::PatternRewriter &rewriter) const final {
    bool isFused = false;

    if (auto lhsUnaryOp = getFusableUnaryOp(binaryOp.getLhs())) {
      fuseInputActivation(lhsUnaryOp, binaryOp, rewriter, /*isLhs=*/true);
      isFused = true;
    }

    if (auto rhsUnaryOp = getFusableUnaryOp(binaryOp.getRhs())) {
      fuseInputActivation(rhsUnaryOp, binaryOp, rewriter, /*isLhs=*/false);
      isFused = true;
    }

    return mlir::success(isFused);
  }

private:
  ElementwiseUnary getFusableUnaryOp(Value operand) const {
    if (!operand.hasOneUse()) {
      return {};
    }

    auto unaryOp = operand.getDefiningOp<ElementwiseUnary>();
    if (unaryOp && unaryOp.getUnaryOpType() != UnaryOpType::Unknown) {
      return unaryOp;
    }

    return {};
  }

  void fuseInputActivation(ElementwiseUnary unaryOp, ElementwiseBinary binaryOp,
                           mlir::PatternRewriter &rewriter, bool isLhs) const {
    rewriter.modifyOpInPlace(binaryOp, [&]() {
      if (isLhs) {
        binaryOp.addLhsActivation(unaryOp.getUnaryOpType(),
                                  unaryOp.getParams());
      } else {
        binaryOp.addRhsActivation(unaryOp.getUnaryOpType(),
                                  unaryOp.getParams());
      }
      rewriter.replaceOp(unaryOp, unaryOp.getInput());
    });
  }
};
} // namespace

namespace {
class TTNNBinaryOpOutputActivation
    : public mlir::OpInterfaceRewritePattern<ElementwiseUnary> {
  using TTNNBinaryOpOutputActivation::OpInterfaceRewritePattern<
      ElementwiseUnary>::OpInterfaceRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ElementwiseUnary unaryOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(unaryOp)) {
      return failure();
    }

    auto binaryOp = getFusableBinaryOp(unaryOp.getInput());
    if (!binaryOp) {
      return failure();
    }

    rewriter.modifyOpInPlace(binaryOp, [&]() {
      binaryOp.addPostActivation(unaryOp.getUnaryOpType(), unaryOp.getParams());
      binaryOp->getResult(0).setType(unaryOp->getResult(0).getType());
    });
    rewriter.replaceOp(unaryOp, unaryOp.getInput());

    return mlir::success();
  }

private:
  bool isFusable(ElementwiseUnary unaryOp) const {
    return unaryOp.getUnaryOpType() != UnaryOpType::Unknown;
  }

  ElementwiseBinary getFusableBinaryOp(Value operand) const {
    if (!operand.hasOneUse()) {
      return {};
    }

    return operand.getDefiningOp<ElementwiseBinary>();
  }
};
} // namespace

namespace {
class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNConv2dWithActivation, TTNNBinaryOpInputsActivation,
                 TTNNBinaryOpOutputActivation>(&getContext());
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
