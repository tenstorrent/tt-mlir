// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsInterfaces.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSwitch.h"

#include "llvm/ADT/TypeSwitch.h"
#include <optional>

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

class TTNNBinaryOpActivationFusing
    : public mlir::OpInterfaceRewritePattern<ElementwiseBinary> {
  using OpInterfaceRewritePattern<ElementwiseBinary>::OpInterfaceRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ElementwiseBinary binaryOp,
                  mlir::PatternRewriter &rewriter) const final {
    bool fused = false;

    if (auto *lhsUnary = findFusableUnaryProducer(binaryOp.getLhs())) {
      binaryOp.addLhsActivation(lhsUnary);
      rewriter.modifyOpInPlace(binaryOp, [&]() { binaryOp.consumeLhs(); });
      fused = true;
    }

    if (auto *rhsUnary = findFusableUnaryProducer(binaryOp.getRhs())) {
      binaryOp.addRhsActivation(rhsUnary);
      rewriter.modifyOpInPlace(binaryOp, [&]() { binaryOp.consumeRhs(); });
      fused = true;
    }

    if (auto *postUnary =
            findFusableUnaryConsumer(binaryOp.getOperation()->getResult(0))) {
      binaryOp.addPostActivation(postUnary);
      assert(postUnary->hasOneUse() &&
             "post-activation should have only one use");
      rewriter.replaceOp(postUnary, binaryOp->getResult(0));
      fused = true;
    }

    return mlir::success(fused);
  }

private:
  ElementwiseUnary findFusableUnaryProducer(Value operand) const {
    Operation *producer = operand.getDefiningOp();
    if (!producer || !producer->hasOneUse()) {
      return {};
    }

    return mlir::dyn_cast<ElementwiseUnary>(producer);
  }

  Operation *findFusableUnaryConsumer(Value result) const {
    if (!result.hasOneUse()) {
      return nullptr;
    }

    Operation *consumer = *result.getUsers().begin();

    return mlir::dyn_cast<ElementwiseUnary>(consumer);
  }

  bool fuseLhsActivation(ElementwiseBinary binaryOp, Operation *unaryOp,
                         mlir::PatternRewriter &rewriter) const {
    auto activationAttr = createUnaryActivationAttr(unaryOp, rewriter);
    if (!activationAttr) {
      return false;
    }

    // Update the binary op using interface methods
    rewriter.modifyOpInPlace(binaryOp.getOperation(), [&]() {
      binaryOp.addLhsActivation(activationAttr);
      binaryOp.getOperation()->replaceUsesOfWith(binaryOp.getLhs(),
                                                 unaryOp->getOperand(0));
    });

    // Remove the unary op if it has no other uses
    if (unaryOp->use_empty()) {
      rewriter.eraseOp(unaryOp);
    }

    return true;
  }

  bool fuseRhsActivation(ElementwiseBinary binaryOp, Operation *unaryOp,
                         mlir::PatternRewriter &rewriter) const {
    auto activationAttr = createUnaryActivationAttr(unaryOp, rewriter);
    if (!activationAttr) {
      return false;
    }

    // Update the binary op using interface methods
    rewriter.modifyOpInPlace(binaryOp.getOperation(), [&]() {
      binaryOp.addRhsActivation(activationAttr);
      binaryOp.getOperation()->replaceUsesOfWith(binaryOp.getRhs(),
                                                 unaryOp->getOperand(0));
    });

    // Remove the unary op if it has no other uses
    if (unaryOp->use_empty()) {
      rewriter.eraseOp(unaryOp);
    }

    return true;
  }

  bool fusePostActivation(ElementwiseBinary binaryOp, Operation *unaryOp,
                          mlir::PatternRewriter &rewriter) const {
    auto activationAttr = createUnaryActivationAttr(unaryOp, rewriter);
    if (!activationAttr) {
      return false;
    }

    // Update the binary op using interface methods
    rewriter.modifyOpInPlace(binaryOp.getOperation(), [&]() {
      binaryOp.addPostActivation(activationAttr);
    });

    // Replace uses of unary op with binary op result
    rewriter.replaceOp(unaryOp, binaryOp.getOperation()->getResult(0));

    return true;
  }

  UnaryWithParamAttr createUnaryWithParam(ElementwiseUnary op) const {
    auto unaryOpType =
        llvm::TypeSwitch<ElementwiseUnary, std::optional<UnaryOpType>>(op)
            .Case<ReluOp>([](auto &&) { return UnaryOpType::Relu; })
            .Case<AbsOp>([](auto &&) { return UnaryOpType::Abs; })
            .Case<NegOp>([](auto &&) { return UnaryOpType::Neg; })
            .Case<RsqrtOp>([](auto &&) { return UnaryOpType::Rsqrt; })
            .Case<SigmoidOp>([](auto &&) { return UnaryOpType::Sigmoid; })
            .Case<SqrtOp>([](auto &&) { return UnaryOpType::Sqrt; })
            .Case<TanhOp>([](auto &&) { return UnaryOpType::Tanh; })
            .Case<SinOp>([](auto &&) { return UnaryOpType::Sin; })
            .Case<CosOp>([](auto &&) { return UnaryOpType::Cos; })
            .Case<AtanOp>([](auto &&) { return UnaryOpType::Atan; })
            .Case<ErfOp>([](auto &&) { return UnaryOpType::Erf; })
            .Case<ErfcOp>([](auto &&) { return UnaryOpType::Erfc; })
            .Case<Expm1Op>([](auto &&) { return UnaryOpType::Expm1; })
            .Case<Log1pOp>([](auto &&) { return UnaryOpType::Log10; })
            .Case<LeakyReluOp>([](auto &&) { return UnaryOpType::LeakyRelu; })
            .Case<FloorOp>([](auto &&) { return UnaryOpType::Floor; })
            .Case<CeilOp>([](auto &&) { return UnaryOpType::Ceil; })
            .Case<SignOp>([](auto &&) { return UnaryOpType::Sign; })
            .Case<IsFiniteOp>([](auto &&) { return UnaryOpType::IsFinite; })
            .Case<LogicalNotOp>(
                [](auto &&) { return UnaryOpType::LogicalNotUnary; })
            .Case<BitwiseNotOp>([](auto &&) { return UnaryOpType::BitwiseNot; })
            .Default([](auto &&) { return std::nullopt; });

    if (!unaryOpType) {
      return nullptr;
    }

    return UnaryWithParamAttr::get(op.getContext(), *unaryOpType, {});
  }

  UnaryWithParamAttr
  createUnaryActivationAttr(Operation *unaryOp,
                            mlir::PatternRewriter &rewriter) const {
    StringRef opTypeName = unaryOp->getName().getStringRef();
    // Remove "ttnn." prefix if present
    if (opTypeName.starts_with("ttnn.")) {
      opTypeName = opTypeName.drop_front(5);
    }

    // Convert operation name to UnaryOpType enum
    std::optional<UnaryOpType> unaryOpType = convertToUnaryOpType(opTypeName);
    if (!unaryOpType) {
      return nullptr; // Unsupported operation type
    }

    // Get parameters from the unary op if it has any
    ArrayRef<FloatAttr> params;
    if (auto unaryInterface = dyn_cast<ElementwiseUnary>(unaryOp)) {
      params = unaryInterface.getParams();
    }

    return UnaryWithParamAttr::get(rewriter.getContext(), *unaryOpType, params);
  }

private:
  std::optional<UnaryOpType> convertToUnaryOpType(StringRef opName) const {
    // Map TTNN operation names to UnaryOpType enum values
    return llvm::StringSwitch<std::optional<UnaryOpType>>(opName)
        .Case("abs", UnaryOpType::Abs)
        .Case("exp", UnaryOpType::Exp)
        .Case("gelu", UnaryOpType::Gelu)
        .Case("log", UnaryOpType::Log)
        .Case("neg", UnaryOpType::Neg)
        .Case("reciprocal", UnaryOpType::Recip)
        .Case("relu", UnaryOpType::Relu)
        .Case("rsqrt", UnaryOpType::Rsqrt)
        .Case("sigmoid", UnaryOpType::Sigmoid)
        .Case("sqrt", UnaryOpType::Sqrt)
        .Case("tanh", UnaryOpType::Tanh)
        .Case("sin", UnaryOpType::Sin)
        .Case("cos", UnaryOpType::Cos)
        .Case("atan", UnaryOpType::Atan)
        .Case("erf", UnaryOpType::Erf)
        .Case("erfc", UnaryOpType::Erfc)
        .Case("expm1", UnaryOpType::Expm1)
        .Case("log1p", UnaryOpType::Log10) // Note: may need adjustment based on
                                           // actual mapping
        .Case("leaky_relu", UnaryOpType::LeakyRelu)
        .Case("silu", UnaryOpType::Silu)
        .Case("floor", UnaryOpType::Floor)
        .Case("ceil", UnaryOpType::Ceil)
        .Case("sign", UnaryOpType::Sign)
        .Case("isfinite", UnaryOpType::IsFinite)
        .Case("logical_not", UnaryOpType::LogicalNotUnary)
        .Case("bitwise_not", UnaryOpType::BitwiseNot)
        // Add more mappings as needed for other fusable operations
        .Default(std::nullopt);
  }
};

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNConv2dWithActivation>(&getContext());
    patterns.add<TTNNBinaryOpActivationFusing>(&getContext());
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
