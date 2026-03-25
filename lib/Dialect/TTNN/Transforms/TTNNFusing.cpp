// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/TopKFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/NLPConcatHeadsDecodeInputRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#endif

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
    conv2dConfigAttr = conv2dConfigAttr.withActivation(activation)
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
    // Extract op name from full operation name (e.g., "ttnn.relu" -> "relu")
    // and convert to enum
    llvm::StringLiteral fullOpName = ActivationOp::getOperationName();
    llvm::StringRef opName = fullOpName.rsplit('.').second;
    auto activation = ttnn::symbolizeUnaryOpType(opName);
    assert(activation.has_value() && "Unsupported activation op");
    return activation.value();
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

    // Since window flattening will add reshape after conv we need to check
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

template <typename SrcOp, typename ActivationOp>
class TTNNMatmulAndLinearWithActivation : public mlir::OpRewritePattern<SrcOp> {
  using TTNNMatmulAndLinearWithActivation::template OpRewritePattern<
      SrcOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(SrcOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp =
        mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    Value activationInput = activationOp.getInput();
    auto activationStr = getActivationString();

    rewriter.modifyOpInPlace(srcOp, [&]() {
      srcOp.setActivationAttr(rewriter.getStringAttr(activationStr));
    });

    rewriter.replaceAllUsesWith(activationOp, activationInput);
    return mlir::success();
  }

private:
  // After tt-metal resolves this issue:
  // https://github.com/tenstorrent/tt-metal/issues/31393, we can use the
  // UnaryWithParam enum directly instead of string.
  std::string getActivationString() const {
    llvm::StringLiteral fullOpName = ActivationOp::getOperationName();
    llvm::StringRef opName = fullOpName.rsplit('.').second;
    return opName.str();
  }

  bool isFusable(SrcOp srcOp) const {
    if (srcOp.getActivation()) {
      return false;
    }

    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }

    return false;
  }
};

#ifdef TTMLIR_ENABLE_OPMODEL

// ============================================================================
// NLP Concat Heads Decode Fusing
// ============================================================================
//
// Matches the decode-phase concat-heads pattern that appears after
// scaled_dot_product_attention_decode in LLMs:
//
//   permute([1, 2, 0, 3])  :  [S, B, H, D] -> [B, H, S, D]
//   reshape                 :  [B, H, S, D] -> [B, H*D]  (or similar collapse)
//
// This sequence shuffles the multi-head attention output back into a single
// hidden dimension. It is replaced by the optimized hardware op
// nlp_concat_heads_decode which performs:
//
//   [S, B, H_padded, D] -> [S, 1, B, num_heads * D]
//
// followed by a reshape to match the original output shape.
//
class NLPConcatHeadsDecodeFusing : public mlir::OpRewritePattern<ReshapeOp> {
  using NLPConcatHeadsDecodeFusing::OpRewritePattern<
      ReshapeOp>::OpRewritePattern;

  // Permutation that converts [S, B, H, D] -> [B, H, S, D].
  static constexpr std::array<int64_t, 4> kConcatHeadsDecodePermutation = {
      1, 2, 0, 3};

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto permuteOp = reshapeOp.getInput().getDefiningOp<PermuteOp>();
    if (!permuteOp) {
      return failure();
    }

    // Check permutation is [1, 2, 0, 3].
    auto permutation = permuteOp.getPermutation();
    if (!llvm::equal(permutation,
                     ArrayRef<int64_t>(kConcatHeadsDecodePermutation))) {
      return failure();
    }

    Value input = permuteOp.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());

    auto inputShape = inputType.getShape();
    int64_t seqLen = inputShape[0];
    int64_t batchSize = inputShape[1];
    int64_t numHeads = inputShape[2];
    int64_t headDim = inputShape[3];

    // NLP concat heads decode is specifically for decode phase (seq_len == 1).
    if (seqLen != 1) {
      return failure();
    }

    // TODO(vkovacevic): https://github.com/tenstorrent/tt-metal/issues/38992
    // The tt-metal nlp_concat_heads_decode op computes its output logical shape
    // from the input's padded shape. If head_dim or batch aren't tile-aligned,
    // the output logical shape will differ from what our IR expects, causing a
    // volume mismatch in the subsequent reshape at runtime.
    constexpr int64_t kTileSize = 32;
    if (headDim % kTileSize != 0 || batchSize % kTileSize != 0) {
      return failure();
    }

    SmallVector<int64_t> concatHeadsOutputShape = {seqLen, 1, batchSize,
                                                   numHeads * headDim};
    auto concatHeadsResultType = utils::RankedTensorTypeFactory::create(
        inputType, concatHeadsOutputShape);

    op_model::ScopedSingletonDeviceGuard deviceGuard(reshapeOp);

    auto nlpConcatHeadsDecodeOp = rewriter.create<NLPConcatHeadsDecodeOp>(
        reshapeOp.getLoc(), concatHeadsResultType, input,
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)),
        /*memory_config=*/MemoryConfigAttr());

    // Validate the fused op. The op requires height-sharded L1 input, so
    // try the workaround-sharded version since the workaround pass hasn't
    // run yet.
    auto workaround = workarounds::decomposition::getWorkaroundedInput(
        nlpConcatHeadsDecodeOp, rewriter);
    if (workaround) {
      auto shardedInputType =
          mlir::cast<RankedTensorType>(workaround->getType());
      auto shardedResultType = utils::RankedTensorTypeFactory::create(
          shardedInputType, concatHeadsOutputShape);

      auto validationOp = rewriter.create<NLPConcatHeadsDecodeOp>(
          reshapeOp.getLoc(), shardedResultType, workaround->getResult(),
          rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)),
          /*memory_config=*/MemoryConfigAttr());

      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(validationOp.getOperation());
      OpConfig config(
          mlir::cast<TTNNLayoutAttr>(shardedResultType.getEncoding()));
      auto validationResult = op_constraint_validation::validateOperation(
          validationOp.getOperation(), inputLayouts, config);

      rewriter.eraseOp(validationOp);
      rewriter.eraseOp(*workaround);

      if (!validationResult.isSuccess()) {
        rewriter.eraseOp(nlpConcatHeadsDecodeOp);
        return failure();
      }
    }

    rewriter.setInsertionPointAfter(nlpConcatHeadsDecodeOp);

    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getType(),
        nlpConcatHeadsDecodeOp.getResult(), reshapeOp.getShapeAttr(),
        /*memory_config=*/MemoryConfigAttr());

    rewriter.replaceOp(reshapeOp, newReshapeOp.getResult());
    return mlir::success();
  }
};

#endif // TTMLIR_ENABLE_OPMODEL

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    // TODO(mvasiljevic): Add HardsigmoidOp once tt-metal issue is resolved
    // https://github.com/tenstorrent/tt-metal/issues/30973
    patterns.add<
        TTNNConv2dWithActivation<ReluOp>, TTNNConv2dWithActivation<Relu6Op>,
        TTNNConv2dWithActivation<SiluOp>, TTNNConv2dWithActivation<SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, GeluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, GeluOp>>(&getContext());

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      FusionValidationConfig validationConfig;
      validationConfig.maxFallbackAttempts = maxFallbackAttempts;

      patterns.add<fusing::RoPEFusing>(&getContext());
      patterns.add<fusing::RoPEDecodeFusing>(&getContext());
      patterns.add<fusing::TopKFusing>(&getContext(), validationConfig);
      patterns.add<fusing::SDPAFusing>(&getContext(), validationConfig);
      patterns.add<NLPConcatHeadsDecodeFusing>(&getContext());
    }
#endif // TTMLIR_ENABLE_OPMODEL

    // Add TypecastOp canonicalization patterns to fold consecutive typecasts
    // (e.g. bf16->f32->bf16) that appear after SDPA fusing, enabling
    // patterns like NLPConcatHeadsDecodeFusing to match cleanly.
    TypecastOp::getCanonicalizationPatterns(patterns, &getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
