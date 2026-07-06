// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SplitQKVFusingPatterns.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"
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

// Build nlp_concat_heads_decode for a [S=1, B, H, D] input and confirm it can
// run: the op needs height-sharded L1 input, but the workaround pass that would
// provide it hasn't run yet, so probe the sharded form through the op model.
// Returns the op, or nullptr (leaving no ops behind) when it can't run. `anchor`
// supplies the insertion location and the device for the probe.
static NLPConcatHeadsDecodeOp
buildValidatedNLPConcatHeadsDecode(Value input, int64_t numHeads,
                                   Operation *anchor,
                                   mlir::PatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto inputShape = inputType.getShape();
  SmallVector<int64_t> outputShape = {inputShape[0], 1, inputShape[1],
                                      numHeads * inputShape[3]};
  auto resultType =
      utils::RankedTensorTypeFactory::create(inputType, outputShape);

  op_model::ScopedSingletonDeviceGuard deviceGuard(anchor);
  auto decodeOp = rewriter.create<NLPConcatHeadsDecodeOp>(
      anchor->getLoc(), resultType, input,
      rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)));

  auto workaround =
      workarounds::decomposition::getWorkaroundedInput(decodeOp, rewriter);
  if (workaround) {
    auto shardedInputType = mlir::cast<RankedTensorType>(workaround->getType());
    auto shardedResultType =
        utils::RankedTensorTypeFactory::create(shardedInputType, outputShape);
    auto validationOp = rewriter.create<NLPConcatHeadsDecodeOp>(
        anchor->getLoc(), shardedResultType, workaround->getResult(),
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)));
    std::vector<TTNNLayoutAttr> inputLayouts =
        utils::extractInputLayouts(validationOp.getOperation());
    OpConfig config(
        mlir::cast<TTNNLayoutAttr>(shardedResultType.getEncoding()));
    auto validationResult = op_constraint_validation::validateOperation(
        validationOp.getOperation(), inputLayouts, config);
    rewriter.eraseOp(validationOp);
    rewriter.eraseOp(*workaround);
    if (!validationResult.isSuccess()) {
      rewriter.eraseOp(decodeOp);
      return nullptr;
    }
  }
  return decodeOp;
}

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
// Canonicalization may fold the permute into a reshape when S==1 (since only
// size-1 dims move). In that case the pattern becomes:
//
//   reshape                 :  [S=1, B, H, D] -> [B, H, 1, D]
//   reshape                 :  [B, H, 1, D]   -> [B, H*D]
//
// Both variants are matched. The sequence is replaced by the optimized
// hardware op nlp_concat_heads_decode which performs:
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

  // Try to match the concat-heads-decode input pattern.
  // Returns the original [S, B, H, D] input value, or nullptr on failure.
  //
  // Matches two cases:
  //   1. permute([1,2,0,3]) on [S, B, H, D] — the un-canonicalized pattern.
  //   2. Direct reshape input with shape [S=1, B, H, D] — the canonicalized
  //      pattern where permute([1,2,0,3]) was folded into a reshape (since
  //      only size-1 dims moved) and then the two reshapes were merged into
  //      one. We detect this by checking the reshape's input is 4D with S==1
  //      and its output collapses the last three dims (B*H*D or similar).
  static Value matchConcatHeadsInput(ReshapeOp reshapeOp) {
    Value reshapeInput = reshapeOp.getInput();

    // Case 1: explicit permute op.
    if (auto permuteOp = reshapeInput.getDefiningOp<PermuteOp>()) {
      auto permutation = permuteOp.getPermutation();
      if (llvm::equal(permutation,
                      ArrayRef<int64_t>(kConcatHeadsDecodePermutation))) {
        return permuteOp.getInput();
      }
      return nullptr;
    }

    // Case 2: canonicalized — the permute was folded away because S==1,
    // leaving a single reshape from [1, B, H, D] -> [B, H*D].
    // We recognise this by checking the reshape's own input is 4D with
    // dim-0 == 1, and the output is a 2D collapse of [B, H*D].
    auto inputType = mlir::cast<RankedTensorType>(reshapeInput.getType());
    auto inputShape = inputType.getShape();

    if (inputShape.size() == 4 && inputShape[0] == 1) {
      auto outShape =
          mlir::cast<RankedTensorType>(reshapeOp.getResult().getType())
              .getShape();
      // Output must be 2D: [B, H*D].
      if (outShape.size() == 2 && outShape[0] == inputShape[1] &&
          outShape[1] == inputShape[2] * inputShape[3]) {
        return reshapeInput;
      }
    }

    return nullptr;
  }

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    Value input = matchConcatHeadsInput(reshapeOp);
    if (!input) {
      return failure();
    }

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

    auto decodeOp =
        buildValidatedNLPConcatHeadsDecode(input, numHeads, reshapeOp, rewriter);
    if (!decodeOp) {
      return failure();
    }

    rewriter.setInsertionPointAfter(decodeOp);
    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getType(), decodeOp.getResult(),
        reshapeOp.getShapeAttr());
    rewriter.replaceOp(reshapeOp, newReshapeOp.getResult());
    return mlir::success();
  }
};

// TTIR-level fusing already folds merge-heads into a batch-first
// concatenate_heads op, so the reshape/permute-rooted fusion above never matches
// the decode case. Match a decode-shaped concatenate_heads ([B, H, S=1, D]) and
// upgrade it to nlp_concat_heads_decode so the KV stays in the decode sharded
// layout rather than going through the generic head concat.
class ConcatHeadsToNLPConcatHeadsDecodeFusing
    : public mlir::OpRewritePattern<ConcatenateHeadsOp> {
  using mlir::OpRewritePattern<ConcatenateHeadsOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ConcatenateHeadsOp concatOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputType = mlir::cast<RankedTensorType>(concatOp.getInput().getType());
    auto inputShape = inputType.getShape(); // [B, H, S, D]
    // Decode phase only (seq_len == 1).
    if (inputShape.size() != 4 || inputShape[2] != 1) {
      return failure();
    }
    int64_t batchSize = inputShape[0];
    int64_t numHeads = inputShape[1];
    int64_t headDim = inputShape[3];

    // nlp_concat_heads_decode derives its output shape from the input's padded
    // shape, so head_dim and batch must be tile-aligned (see the sibling
    // fusion's note on tt-metal #38992).
    constexpr int64_t kTileSize = 32;
    if (headDim % kTileSize != 0 || batchSize % kTileSize != 0) {
      return failure();
    }

    // nlp_concat_heads_decode consumes [S=1, B, H, D]; concatenate_heads is
    // batch-first. With S==1 this reshape only repositions the unit dim, so it
    // preserves data order (and cancels the frontend's [S,B,H,D]->[B,H,S,D]).
    SmallVector<int64_t> sbhdShape = {1, batchSize, numHeads, headDim};
    auto sbhdType = utils::RankedTensorTypeFactory::create(inputType, sbhdShape);
    SmallVector<int32_t> sbhdShape32(sbhdShape.begin(), sbhdShape.end());
    auto sbhdInput = rewriter.create<ReshapeOp>(
        concatOp.getLoc(), sbhdType, concatOp.getInput(),
        rewriter.getI32ArrayAttr(sbhdShape32));

    auto decodeOp = buildValidatedNLPConcatHeadsDecode(
        sbhdInput.getResult(), numHeads, concatOp, rewriter);
    if (!decodeOp) {
      return failure();
    }

    // The [S, 1, B, H*D] decode output collapses back to concatenate_heads'
    // [B, S, H*D] result (again a unit-dim-only reshape).
    rewriter.setInsertionPointAfter(decodeOp);
    auto outType = mlir::cast<RankedTensorType>(concatOp.getResult().getType());
    SmallVector<int32_t> outShape32(outType.getShape().begin(),
                                    outType.getShape().end());
    rewriter.replaceOpWithNewOp<ReshapeOp>(concatOp, outType,
                                           decodeOp.getResult(),
                                           rewriter.getI32ArrayAttr(outShape32));
    return success();
  }
};

#endif // TTMLIR_ENABLE_OPMODEL

// Folds the "expand-style" repeat_interleave — reshape (insert a unit dim) ->
// repeat (tile that unit dim by n) -> reshape (merge it into the preceding
// dim) — into a single repeat_interleave on the preceding dim. Tiling a size-1
// dim and merging it into its neighbor is exactly repeat_interleave of that
// neighbor, so the rewrite is value-preserving. Frontends that lower
// torch.repeat_interleave via unsqueeze/expand/reshape (e.g. HF's repeat_kv for
// GQA) produce this shape; the StableHLO frontend emits repeat_interleave
// directly. Normalizing to the canonical op lets downstream fusions that match
// repeat_interleave (notably SDPAFusing's analyzeK/analyzeV, which strip a
// repeat_interleave on the num-heads dim so KV feeds GQA-native into
// scaled_dot_product_attention_decode) fire regardless of frontend.
class RepeatToRepeatInterleave : public mlir::OpRewritePattern<ReshapeOp> {
  using mlir::OpRewritePattern<ReshapeOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp mergeReshape,
                  mlir::PatternRewriter &rewriter) const final {
    auto repeatOp = mergeReshape.getInput().getDefiningOp<RepeatOp>();
    if (!repeatOp || !repeatOp.getResult().hasOneUse()) {
      return failure();
    }

    // Match a unit dim being expanded: exactly one tiled dim, size 1 in input.
    auto inType = mlir::cast<RankedTensorType>(repeatOp.getInput().getType());
    llvm::ArrayRef<int64_t> inShape = inType.getShape();
    llvm::ArrayRef<int64_t> repeatDims = repeatOp.getRepeatDims().getShape();
    if (repeatDims.size() != inShape.size()) {
      return failure();
    }
    int64_t tileDim = -1;
    int64_t tileFactor = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(repeatDims.size()); ++i) {
      if (repeatDims[i] == 1) {
        continue;
      }
      if (tileDim != -1) {
        return failure(); // more than one tiled dim
      }
      tileDim = i;
      tileFactor = repeatDims[i];
    }
    // Need a preceding dim to merge into (tileDim >= 1) and a genuine unit-dim
    // expansion (tileFactor > 1, input size 1 at tileDim).
    if (tileDim < 1 || tileFactor <= 1 || inShape[tileDim] != 1) {
      return failure();
    }

    // The outer reshape must merge (tileDim-1, tileDim) into one dim and leave
    // every other dim intact — that is exactly what makes tile+merge equal to
    // repeat_interleave on (tileDim-1). Merging (tileDim, tileDim+1) instead
    // would be plain tiling, not interleave, so reject anything whose output
    // shape does not match the (tileDim-1, tileDim) merge.
    auto outType =
        mlir::cast<RankedTensorType>(mergeReshape.getResult().getType());
    llvm::SmallVector<int64_t> mergedShape;
    for (int64_t i = 0; i < static_cast<int64_t>(inShape.size()); ++i) {
      if (i == tileDim) {
        continue; // folded into the preceding dim
      }
      mergedShape.push_back(i == tileDim - 1 ? inShape[i] * tileFactor
                                             : inShape[i]);
    }
    if (outType.getShape() != llvm::ArrayRef<int64_t>(mergedShape)) {
      return failure();
    }

    llvm::SmallVector<int64_t> droppedShape;
    for (int64_t i = 0; i < static_cast<int64_t>(inShape.size()); ++i) {
      if (i != tileDim) {
        droppedShape.push_back(inShape[i]);
      }
    }
    RankedTensorType droppedType =
        utils::RankedTensorTypeFactory::create(inType, droppedShape);
    llvm::SmallVector<int32_t> droppedShape32(droppedShape.begin(),
                                              droppedShape.end());
    auto dropped = rewriter.create<ReshapeOp>(
        mergeReshape.getLoc(), droppedType, repeatOp.getInput(),
        rewriter.getI32ArrayAttr(droppedShape32));

    rewriter.replaceOpWithNewOp<RepeatInterleaveOp>(
        mergeReshape, outType, dropped.getResult(),
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(tileFactor)),
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(tileDim - 1)));
    return success();
  }
};

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    // Normalize expand-style repeats (reshape/repeat/reshape) into
    // repeat_interleave in a separate sweep that runs to fixpoint before the
    // fusion patterns below. This cannot be co-scheduled with SDPAFusing: that
    // pattern anchors on the attention matmul and commits to the pre-rewrite
    // (materialized, full-head) KV before this rewrite would fire, so the
    // canonical repeat_interleave must already exist when SDPAFusing runs its
    // analyzeK/analyzeV to strip the repeat off the num-heads dim.
    {
      RewritePatternSet prePatterns(&getContext());
      prePatterns.add<RepeatToRepeatInterleave>(&getContext());
      (void)applyPatternsGreedily(getOperation(), std::move(prePatterns));
    }

    RewritePatternSet patterns(&getContext());
    // TODO(mvasiljevic): Add HardsigmoidOp once tt-metal issue is resolved
    // https://github.com/tenstorrent/tt-metal/issues/30973
    patterns.add<
        TTNNConv2dWithActivation<ReluOp>,
        TTNNConv2dWithActivation<Relu6Op>, TTNNConv2dWithActivation<SiluOp>,
        TTNNConv2dWithActivation<SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, GeluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, GeluOp>>(&getContext());

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      OpValidationConfig validationConfig;
      validationConfig.maxFallbackAttempts = maxFallbackAttempts;

      if (enableRoPEFusion) {
        patterns.add<fusing::RoPERotateHalfFusing>(&getContext(),
                                                   validationConfig);
        patterns.add<fusing::RoPEExpandedFusing>(&getContext(),
                                                 validationConfig);
      }
      // RoPEDecodeFusing must always run (not gated by enableRoPEFusion)
      // because it rearranges the [2,0,1,3] permutes that
      // NLPCreateQKVHeadsDecodeFusing depends on. When TTIR-level RoPE fusion
      // is active (enableRoPEFusion=false), the RotaryEmbeddingOp already
      // exists from TTIR lowering — RoPEDecodeFusing detects the decode
      // signature and sets token_index, enabling the decode QKV upgrade.
      // TODO(sdjordjevic): #8598 Decouple NLPCreateQKVHeadsDecodeFusing from
      // RoPEDecodeFusing
      patterns.add<fusing::RoPEDecodeFusing>(&getContext());
      patterns.add<fusing::SDPAFusing>(&getContext(), validationConfig);
      patterns.add<NLPConcatHeadsDecodeFusing>(&getContext());
      patterns.add<ConcatHeadsToNLPConcatHeadsDecodeFusing>(&getContext());
      patterns.add<fusing::SplitQueryKeyValueAndSplitHeadsFusing<MatmulOp>>(
          &getContext(), validationConfig);
      patterns.add<fusing::SplitQueryKeyValueAndSplitHeadsFusing<LinearOp>>(
          &getContext(), validationConfig);
      patterns.add<fusing::NLPCreateQKVHeadsDecodeFusing>(&getContext(),
                                                          validationConfig);
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
