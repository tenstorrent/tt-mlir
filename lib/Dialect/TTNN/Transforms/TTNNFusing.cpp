// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
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

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    // `reshapeOp` is the head-collapse reshape ([B, H, 1, D] -> [B, H*D]). Walk
    // back through any dtype typecasts and an optional NaN-safe row-zeroing
    // `where` to the head reorder. The reorder is [S, B, H, D] -> [B, H, S, D],
    // expressed either as ttnn.permute([1, 2, 0, 3]) or — for decode (S == 1) —
    // the shape-only ttnn.reshape it canonicalizes to (a [1,2,0,3] permute that
    // only relocates the unit seq dim is rewritten to a reshape, see
    // PermuteOp::getCanonicalizationPatterns).
    Value beforeCollapse =
        ttmlir::utils::lookThrough<TypecastOp>(reshapeOp.getInput());

    // Optional NaN-safe scrub: where(cond, replacement, data). The attention
    // output flows through getThird(); record the op so it can be re-applied
    // after the concat.
    WhereOp scrub = beforeCollapse.getDefiningOp<WhereOp>();
    Value reorderResult =
        scrub ? ttmlir::utils::lookThrough<TypecastOp>(scrub.getThird())
              : beforeCollapse;

    // Match the reorder and recover its [S, B, H, D] input.
    Value input;
    if (auto permuteOp = reorderResult.getDefiningOp<PermuteOp>()) {
      if (!llvm::equal(permuteOp.getPermutation(),
                       ArrayRef<int64_t>(kConcatHeadsDecodePermutation))) {
        return failure();
      }
      input = permuteOp.getInput();
    } else if (auto reorderReshape = reorderResult.getDefiningOp<ReshapeOp>()) {
      auto inTy =
          mlir::dyn_cast<RankedTensorType>(reorderReshape.getInput().getType());
      auto outTy = mlir::dyn_cast<RankedTensorType>(reorderReshape.getType());
      if (!inTy || !outTy || inTy.getRank() != 4 || outTy.getRank() != 4) {
        return failure();
      }
      // Equivalent to permute([1, 2, 0, 3]) with S == 1: in [S,B,H,D] maps to
      // out [B,H,S,D].
      ArrayRef<int64_t> in = inTy.getShape();
      ArrayRef<int64_t> out = outTy.getShape();
      if (in[0] != 1 || out[0] != in[1] || out[1] != in[2] || out[2] != in[0] ||
          out[3] != in[3]) {
        return failure();
      }
      input = reorderReshape.getInput();
    } else {
      // Merged form: a [1, 2, 0, 3] permute that only relocates the unit seq
      // dim is canonicalized to a reshape and then folded into this collapse
      // (PermuteOp::getCanonicalizationPatterns + foldConsecutiveReshape), so
      // there is no separate reorder op — `reshapeOp` itself collapses
      // [1, B, H, D] -> [B, H*D] and `reorderResult` is the [S, B, H, D] input.
      auto inTy = mlir::dyn_cast<RankedTensorType>(reorderResult.getType());
      auto outTy = mlir::dyn_cast<RankedTensorType>(reshapeOp.getType());
      if (!inTy || !outTy || inTy.getRank() != 4 || outTy.getRank() != 2 ||
          inTy.getShape()[0] != 1 ||
          outTy.getShape()[0] != inTy.getShape()[1] ||
          outTy.getShape()[1] != inTy.getShape()[2] * inTy.getShape()[3]) {
        return failure();
      }
      input = reorderResult;
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

    // The op height-shards the batch one shard per core, so batchSize must fit
    // the worker grid; beyond that the sharded layout built below asserts in
    // deriveCanonicalL1CoreRangeSet.
    ttcore::DeviceAttr device = ttcore::lookupDevice(reshapeOp);
    int64_t workerGridVolume =
        ttmlir::utils::volume(device.getWorkerGrid().getShape());
    if (batchSize > workerGridVolume) {
      return failure();
    }

    // Ops created below are rolled back if op-model validation declines the
    // fused op (the greedy rewriter does not auto-revert on failure()).
    SmallVector<Operation *> createdOps;
    Value opInput = input;

    // Re-apply the NaN-safe scrub on the op's [S, B, H, D] input. The scrub ran
    // on the reordered [B, H, 1, D] tensor with a [B, H, 1, 1] condition
    // (broadcast over head_dim) and a splat replacement, so it zeroes whole
    // (B, H) heads — which commutes with both the reorder and the head concat.
    // The condition is reshaped to [S, B, H, 1] to match the pre-reorder
    // layout.
    if (scrub) {
      Value cond = scrub.getFirst();
      Value replacement = scrub.getSecond();
      auto condType = mlir::dyn_cast<RankedTensorType>(cond.getType());
      auto replType = mlir::dyn_cast<RankedTensorType>(replacement.getType());
      if (!condType || !replType || condType.getRank() != 4) {
        return failure();
      }
      // Soundness guard: condition per-batch and broadcast over the seq and
      // head_dim axes, with a splat replacement. The head axis may be either
      // per-head (numHeads) or broadcast over heads (1) — broadcasting is sound
      // too, as it zeroes all of a batch's heads uniformly, which still
      // commutes with the reorder and head concat. Otherwise the commute is
      // invalid.
      ArrayRef<int64_t> condShape = condType.getShape();
      int64_t condHeads = condShape[1];
      if (condShape[0] != batchSize ||
          (condHeads != numHeads && condHeads != 1) || condShape[2] != 1 ||
          condShape[3] != 1 ||
          !llvm::all_of(replType.getShape(),
                        [](int64_t d) { return d == 1; })) {
        return failure();
      }
      // Re-materialize the condition on the pre-reorder [S, B, H, D] layout,
      // preserving its head axis (numHeads or the broadcast 1).
      auto reCondType = utils::RankedTensorTypeFactory::create(
          condType, SmallVector<int64_t>{seqLen, batchSize, condHeads, 1});
      auto reCondOp = rewriter.create<ReshapeOp>(
          scrub.getLoc(), reCondType, cond,
          rewriter.getI32ArrayAttr({static_cast<int32_t>(seqLen),
                                    static_cast<int32_t>(batchSize),
                                    static_cast<int32_t>(condHeads), 1}));
      createdOps.push_back(reCondOp);
      auto scrubbed = rewriter.create<WhereOp>(
          scrub.getLoc(), inputType, reCondOp.getResult(), replacement, input);
      createdOps.push_back(scrubbed);
      opInput = scrubbed.getResult();
    }

    // nlp_concat_heads_decode runs in the collapse's element type; insert a
    // typecast if the (possibly higher-precision) scrub/input dtype differs.
    auto collapseType = reshapeOp.getType();
    if (inputType.getElementType() != collapseType.getElementType()) {
      ttcore::DataType collapseDataType =
          mlir::cast<TTNNLayoutAttr>(collapseType.getEncoding()).getDataType();
      auto castType = utils::RankedTensorTypeFactory::create(
          mlir::cast<RankedTensorType>(opInput.getType()), collapseDataType);
      auto castOp =
          rewriter.create<TypecastOp>(reshapeOp.getLoc(), castType, opInput);
      createdOps.push_back(castOp);
      opInput = castOp.getResult();
    }

    SmallVector<int64_t> concatHeadsOutputShape = {seqLen, 1, batchSize,
                                                   numHeads * headDim};
    auto concatHeadsResultType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(opInput.getType()),
        concatHeadsOutputShape);

    op_model::ScopedSingletonDeviceGuard deviceGuard(reshapeOp);

    auto nlpConcatHeadsDecodeOp = rewriter.create<NLPConcatHeadsDecodeOp>(
        reshapeOp.getLoc(), concatHeadsResultType, opInput,
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)));
    createdOps.push_back(nlpConcatHeadsDecodeOp);

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
        for (Operation *op : llvm::reverse(createdOps)) {
          rewriter.eraseOp(op);
        }
        return failure();
      }
    }

    rewriter.setInsertionPointAfter(nlpConcatHeadsDecodeOp);

    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), collapseType, nlpConcatHeadsDecodeOp.getResult(),
        reshapeOp.getShapeAttr());

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
      // Sibling that matches the canonicalized (folded reshape) form of the
      // decode permute the same RoPEDecodeFusing handles.
      patterns.add<fusing::RoPEDecodeReshapeFusing>(&getContext());
      if (enableSDPAFusion) {
        patterns.add<fusing::SDPAFusing>(&getContext(), validationConfig);
      }
      patterns.add<NLPConcatHeadsDecodeFusing>(&getContext());
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
