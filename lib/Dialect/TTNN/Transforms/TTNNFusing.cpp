// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
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

// Shared builder for the two all_gather+matmul fusion patterns below. Replaces
// `toReplace` with an all_gather_minimal_matmul_async that gathers
// `allGatherOp`'s input and multiplies by `weight`, carrying the collective
// attributes from `allGatherOp`. `addcmulInput1`/`addcmulInput2`/`scalar` are
// null for the plain matmul fusion and populated for the gated-residual variant.
// Semaphores are left unbound here and materialized later by
// TTNNAllocateDistributedOpSemaphores via the DistributedOpInterface.
static void replaceWithFusedAllGatherMatmul(
    mlir::PatternRewriter &rewriter, mlir::Operation *toReplace,
    mlir::Type resultType, AllGatherOp allGatherOp, mlir::Value weight,
    mlir::Value bias, mlir::Value addcmulInput1, mlir::Value addcmulInput2,
    mlir::FloatAttr scalar, mlir::Value device) {
  rewriter.replaceOpWithNewOp<AllGatherMinimalMatmulAsyncOp>(
      toReplace, resultType, allGatherOp.getInput(), weight, bias,
      addcmulInput1, addcmulInput2,
      /*multi_device_semaphore=*/mlir::ValueRange{},
      /*barrier_semaphore=*/mlir::Value(), device,
      allGatherOp.getClusterAxisAttr(), scalar, allGatherOp.getTopologyAttr(),
      allGatherOp.getNumLinksAttr(), /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*dtype=*/ttcore::DataTypeAttr(), /*force_transpose=*/true,
      /*num_workers_per_link=*/1u, /*num_buffers_per_channel=*/1u, /*chunks=*/1,
      /*dim=*/allGatherOp.getAllGatherDim());
}

// True if this matmul/linear result flows into a gated-residual epilogue
// (a multiply then an add), which AllGatherMatmulAddcmulFusing folds in whole.
template <typename MatmulLikeOp>
static bool feedsGatedResidualEpilogue(MatmulLikeOp matmulOp) {
  if (!matmulOp.getResult().hasOneUse()) {
    return false;
  }
  auto mulOp =
      mlir::dyn_cast<MultiplyOp>(*matmulOp.getResult().getUsers().begin());
  if (!mulOp || !mulOp.getResult().hasOneUse()) {
    return false;
  }
  return mlir::isa<AddOp>(*mulOp.getResult().getUsers().begin());
}

// Fuses an all_gather feeding a matmul/linear into a single kernel:
//   gathered = all_gather(x)            (x sharded along the contraction dim)
//   proj     = matmul(gathered, W)      (or linear(gathered, W, bias))
// becomes ttnn.all_gather_minimal_matmul_async, which performs the gather and
// the matmul together, overlapping communication with compute. Gated behind the
// `enable-all-gather-matmul-fusion` pass option. Bails on activation/transposed
// operands since the fused op does not model those. The gathered intermediate
// must have a single use (the matmul) so the collective is not duplicated.
template <typename MatmulLikeOp>
class AllGatherMatmulFusing : public mlir::OpRewritePattern<MatmulLikeOp> {
  using AllGatherMatmulFusing::template OpRewritePattern<
      MatmulLikeOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MatmulLikeOp matmulOp,
                  mlir::PatternRewriter &rewriter) const final {
    // The activation operand A must be produced by an all_gather feeding only
    // this matmul (otherwise fusing would duplicate the collective).
    AllGatherOp allGatherOp =
        matmulOp.getA().template getDefiningOp<AllGatherOp>();
    if (!allGatherOp || !allGatherOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // The fused op models neither a fused activation nor transposed operands.
    if (matmulOp.getActivation() || matmulOp.getTransposeA() ||
        matmulOp.getTransposeB()) {
      return mlir::failure();
    }

    // Defer to AllGatherMatmulAddcmulFusing when a gated-residual epilogue
    // consumes this matmul, so the whole epilogue folds into one op instead of
    // leaving a separate multiply/add behind.
    if (feedsGatedResidualEpilogue(matmulOp)) {
      return mlir::failure();
    }

    mlir::Value bias;
    if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
      bias = matmulOp.getBias();
    }

    mlir::Value device =
        ttnn::utils::getOrInsertDevice(rewriter, matmulOp).getResult();

    replaceWithFusedAllGatherMatmul(
        rewriter, matmulOp, matmulOp.getResult().getType(), allGatherOp,
        matmulOp.getB(), bias, /*addcmul_input1=*/mlir::Value(),
        /*addcmul_input2=*/mlir::Value(), /*scalar=*/mlir::FloatAttr(), device);
    return mlir::success();
  }
};

// Fuses an all_gather + matmul/linear + gated-residual epilogue into a single
// all_gather_minimal_matmul_async:
//   gathered = all_gather(x)
//   proj     = matmul(gathered, W)      (or linear(gathered, W, bias))
//   out      = residual + gate * proj
// maps residual -> addcmul_input1, gate -> addcmul_input2, scalar = 1.0. This
// mirrors tt-metal's optional addcmul (gated-residual) epilogue on the same
// kernel. Gated behind `enable-all-gather-matmul-fusion`. Requires single uses
// of the gather, projection, and multiply so nothing is duplicated, and bails
// on activation/transposed operands the fused op cannot model.
template <typename MatmulLikeOp>
class AllGatherMatmulAddcmulFusing : public mlir::OpRewritePattern<AddOp> {
  using AllGatherMatmulAddcmulFusing::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final {
    // add is commutative: one operand is the (gate * proj) multiply, the other
    // is the residual.
    MultiplyOp mulOp = addOp.getLhs().getDefiningOp<MultiplyOp>();
    mlir::Value residual = addOp.getRhs();
    if (!mulOp) {
      mulOp = addOp.getRhs().getDefiningOp<MultiplyOp>();
      residual = addOp.getLhs();
    }
    if (!mulOp || !mulOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // multiply is commutative: one operand is the projection, the other gate.
    MatmulLikeOp projOp = mulOp.getLhs().getDefiningOp<MatmulLikeOp>();
    mlir::Value gate = mulOp.getRhs();
    if (!projOp) {
      projOp = mulOp.getRhs().getDefiningOp<MatmulLikeOp>();
      gate = mulOp.getLhs();
    }
    if (!projOp || !projOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // The projection's activation must come from an all_gather feeding only it.
    AllGatherOp allGatherOp =
        projOp.getA().template getDefiningOp<AllGatherOp>();
    if (!allGatherOp || !allGatherOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // The fused op models neither a fused activation nor transposed operands.
    if (projOp.getActivation() || projOp.getTransposeA() ||
        projOp.getTransposeB()) {
      return mlir::failure();
    }

    mlir::Value bias;
    if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
      bias = projOp.getBias();
    }

    mlir::Value device =
        ttnn::utils::getOrInsertDevice(rewriter, addOp).getResult();

    replaceWithFusedAllGatherMatmul(
        rewriter, addOp, addOp.getType(), allGatherOp, projOp.getB(), bias,
        /*addcmul_input1=*/residual, /*addcmul_input2=*/gate,
        /*scalar=*/rewriter.getF32FloatAttr(1.0), device);
    return mlir::success();
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

    // The op height-shards the batch one shard per core, so batchSize must fit
    // the worker grid; beyond that the sharded layout built below asserts in
    // deriveCanonicalL1CoreRangeSet.
    ttcore::DeviceAttr device = ttcore::lookupDevice(reshapeOp);
    int64_t workerGridVolume =
        ttmlir::utils::volume(device.getWorkerGrid().getShape());
    if (batchSize > workerGridVolume) {
      return failure();
    }

    SmallVector<int64_t> concatHeadsOutputShape = {seqLen, 1, batchSize,
                                                   numHeads * headDim};
    auto concatHeadsResultType = utils::RankedTensorTypeFactory::create(
        inputType, concatHeadsOutputShape);

    op_model::ScopedSingletonDeviceGuard deviceGuard(reshapeOp);

    auto nlpConcatHeadsDecodeOp = rewriter.create<NLPConcatHeadsDecodeOp>(
        reshapeOp.getLoc(), concatHeadsResultType, input,
        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(numHeads)));

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
        rewriter.eraseOp(nlpConcatHeadsDecodeOp);
        return failure();
      }
    }

    rewriter.setInsertionPointAfter(nlpConcatHeadsDecodeOp);

    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getType(),
        nlpConcatHeadsDecodeOp.getResult(), reshapeOp.getShapeAttr());

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

    if (enableAllGatherMatmulFusion) {
      patterns.add<AllGatherMatmulFusing<MatmulOp>,
                   AllGatherMatmulFusing<LinearOp>,
                   AllGatherMatmulAddcmulFusing<MatmulOp>,
                   AllGatherMatmulAddcmulFusing<LinearOp>>(&getContext());
    }

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
      patterns.add<fusing::SDPAFusing>(&getContext(), validationConfig);
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
