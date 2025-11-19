// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
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

// ============================================================================
// SDPA Fusing - Modular Architecture
// ============================================================================
//
// This is organized as a hierarchy:
//   - SDPAFusingBase: Common pattern matching and transformation utilities
//   - SDPADecodeFusing: Specializes for decode case (query seq_len == 1)
//   - SDPAPrefillFusing: Specializes for prefill case (query seq_len > 1)
//
// This pattern identifies Scaled Dot Product Attention (SDPA) by matching its
// mathematical semantics rather than exact IR structure:
//
// Attention(Q, K, V) = softmax(scale * (Q @ K^T) + mask) @ V
//
// The pattern is matched by working backwards from the final matmul:
//
// 1. scores@V matmul (attention output):
//    attention_output = matmul(attention_scores, V)
//
// 2. Softmax normalization (produces attention scores):
//    attention_scores = softmax(scaled_qk_scores)
//
// 3. Optional mask addition (adds attention mask before softmax):
//    masked_qk_scores = add(scaled_qk_scores, attention_mask)
//    where attention_mask has shape [batch, 1, query_seq_len, kv_seq_len]
//
// 4. Optional scaling (scales the QK^T scores):
//    scaled_qk_scores = multiply(qk_scores, scale_constant)
//
// 5. Q@K^T matmul (computes attention scores):
//    qk_scores = matmul(Q, K^T)
//
// The matcher walks through the IR backwards starting from the final matmul,
// skipping through layout operations (reshape, permute, repeat) which are
// transparent to the data flow. It extracts scale and mask from the chain
// between softmax and the Q@K^T matmul, then verifies the Q@K^T matmul exists.
//

//
// Base class: Common SDPA pattern matching and utilities
//
class SDPAFusingBase : public mlir::OpRewritePattern<MatmulOp> {
protected:
  // Named constants for tensor dimension indices
  struct TensorDims {
    // 4D tensor dimensions [batch, num_heads, seq_len, head_size]
    static constexpr int64_t kBatch = 0;
    static constexpr int64_t kNumHeads = 1;
    static constexpr int64_t kSeqLen = 2;
    static constexpr int64_t kHeadSize = 3;

    // 3D tensor dimensions [batch*num_heads, seq_len, head_size]
    static constexpr int64_t k3DBatchHeads = 0;
    static constexpr int64_t k3DSeqLen = 1;
    static constexpr int64_t k3DHeadSize = 2;
  };

  struct AttentionComponents {
    Value query;
    Value key;
    Value value;
    Value mask;
    float scale = 1.0f;
    MatmulOp qkMatmul;
    MatmulOp attentionMatmul;
    SoftmaxOp softmax;
    LogicalResult canFuse() const {
      if (!query || !key || !value) {
        return failure();
      }

      auto queryType = mlir::dyn_cast<RankedTensorType>(query.getType());
      auto keyType = mlir::dyn_cast<RankedTensorType>(key.getType());
      auto valueType = mlir::dyn_cast<RankedTensorType>(value.getType());

      if (!queryType || !keyType || !valueType) {
        return failure();
      }

      int64_t queryRank = queryType.getRank();
      if (queryRank != 3 && queryRank != 4) {
        return failure();
      }

      if (keyType.getRank() != 4 || valueType.getRank() != 4) {
        return failure();
      }

      if (!qkMatmul->hasOneUse() || !softmax->hasOneUse() ||
          !attentionMatmul->hasOneUse()) {
        return failure();
      }

      // If query is 3D, verify it can be reshaped to 4D
      if (queryRank == 3) {
        auto queryShape = queryType.getShape();
        int64_t batchSize = keyType.getShape()[TensorDims::kBatch];

        // Check that batch*num_heads is divisible by batch size
        if (queryShape[TensorDims::k3DBatchHeads] % batchSize != 0) {
          return failure();
        }
      }

      return success();
    }
  };

  using SDPAFusingBase::OpRewritePattern<MatmulOp>::OpRewritePattern;

  bool matchAttentionPattern(Operation *anchor,
                             AttentionComponents &components) const {
    auto matmul = dyn_cast<MatmulOp>(anchor);
    if (!matmul) {
      return false;
    }

    SoftmaxOp softmax = findOpInChain<SoftmaxOp>(matmul.getA());
    if (!softmax) {
      return false;
    }

    components.attentionMatmul = matmul;
    components.softmax = softmax;
    components.value = traceToSourceTensor(matmul.getB());

    Value qkMatmulOutput = extractScaleAndMask(
        softmax.getInput(), components.scale, components.mask);
    if (!qkMatmulOutput) {
      return false;
    }

    MatmulOp qkMatmul = findOpInChain<MatmulOp>(qkMatmulOutput);
    if (!qkMatmul) {
      return false;
    }

    components.qkMatmul = qkMatmul;
    components.query = traceToSourceTensor(qkMatmul.getA());
    components.key = traceToSourceTensor(qkMatmul.getB());

    return components.query && components.key && components.value;
  }

  template <typename OpType>
  OpType findOpInChain(Value v, size_t maxDepth = 10) const {
    size_t depth = 0;
    while (v && depth++ < maxDepth) {
      Operation *defOp = v.getDefiningOp();
      if (!defOp) {
        return nullptr;
      }

      if (auto targetOp = dyn_cast<OpType>(defOp)) {
        return targetOp;
      }

      if (isLayoutOp(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      return nullptr;
    }
    return nullptr;
  }

  Value extractScaleAndMask(Value v, float &scale, Value &mask) const {
    scale = 1.0f;
    mask = nullptr;

    while (auto *defOp = v.getDefiningOp()) {
      // Try to extract scale from multiply
      if (auto mulOp = dyn_cast<MultiplyOp>(defOp)) {
        if (auto scaleVal = extractConstantScale(mulOp.getRhs())) {
          scale = *scaleVal;
          v = mulOp.getLhs();
          continue;
        }
        if (auto scaleVal = extractConstantScale(mulOp.getLhs())) {
          scale = *scaleVal;
          v = mulOp.getRhs();
          continue;
        }
      }

      // Try to extract mask from add
      if (auto addOp = dyn_cast<AddOp>(defOp)) {
        if (looksLikeMask(addOp.getRhs())) {
          mask = addOp.getRhs();
          v = addOp.getLhs();
          continue;
        }
        if (looksLikeMask(addOp.getLhs())) {
          mask = addOp.getLhs();
          v = addOp.getRhs();
          continue;
        }
      }

      // Skip layout ops
      if (isLayoutOp(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      break;
    }

    return v;
  }

  bool isLayoutOp(Operation *op) const {
    return isa<ReshapeOp, PermuteOp, RepeatOp, TypecastOp>(op);
  }

  Value traceToSourceTensor(Value v) const {
    while (v) {
      Operation *defOp = v.getDefiningOp();
      if (!defOp || !isLayoutOp(defOp)) {
        break;
      }

      v = defOp->getOperand(0);
    }
    return v;
  }

  std::optional<float> extractConstantScale(Value scaleVal) const {
    if (!scaleVal) {
      return std::nullopt;
    }

    Operation *defOp = scaleVal.getDefiningOp();
    if (!defOp) {
      return std::nullopt;
    }

    if (auto fullOp = dyn_cast<FullOp>(defOp)) {
      if (auto fillValueAttr = dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        return fillValueAttr.getValue().convertToFloat();
      }
      if (auto fillValueAttr = dyn_cast<IntegerAttr>(fullOp.getFillValue())) {
        return static_cast<float>(fillValueAttr.getValue().getSExtValue());
      }
    }

    return std::nullopt;
  }

  bool looksLikeMask(Value v) const {
    auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
    if (!type || type.getRank() != 4) {
      return false;
    }
    // Attention masks typically broadcast over heads:
    // [batch x 1 x query_seq_len x kv_seq_len]
    return type.getShape()[TensorDims::kNumHeads] == 1;
  }

  // Transforms [batch*num_heads, seq_len, head_size] -> [batch, num_heads,
  // seq_len, head_size]
  Value reshapeQueryTo4D(Value query, Value key,
                         mlir::PatternRewriter &rewriter, Location loc) const {
    auto queryType = mlir::cast<RankedTensorType>(query.getType());
    auto keyType = mlir::cast<RankedTensorType>(key.getType());

    // Only reshape if query is 3D
    if (queryType.getRank() != 3) {
      return query;
    }

    auto queryShape = queryType.getShape();
    auto keyShape = keyType.getShape();
    int64_t batchSize = keyShape[TensorDims::kBatch];

    SmallVector<int64_t> reshapedShape = {
        batchSize, queryShape[TensorDims::k3DBatchHeads] / batchSize,
        queryShape[TensorDims::k3DSeqLen], queryShape[TensorDims::k3DHeadSize]};

    auto typedQuery =
        mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(query);
    return ttir_to_ttnn::utils::generateReshape(typedQuery, reshapedShape,
                                                rewriter, loc)
        .getResult();
  }

  // Common mask broadcasting (used by both prefill and decode)
  Value broadcastMask(Value mask, Value query, Value key,
                      mlir::PatternRewriter &rewriter, Location loc) const {
    if (!mask) {
      return mask;
    }

    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto queryType = mlir::cast<RankedTensorType>(query.getType());
    auto keyType = mlir::cast<RankedTensorType>(key.getType());
    auto maskShape = maskType.getShape();

    if (maskShape.size() != 4) {
      return mask;
    }

    SmallVector<int64_t> targetShape = {
        keyType.getShape()[TensorDims::kBatch], // batch from key
        1, // num_heads (always 1 for masks)
        queryType.getShape()[TensorDims::kSeqLen], // query_seq_len from query
        keyType.getShape()[TensorDims::kSeqLen]    // kv_seq_len from key
    };

    // Check if broadcasting is needed
    if (llvm::equal(maskShape, targetShape)) {
      return mask;
    }

    auto repeatDims =
        ttmlir::utils::getBroadcastDimensions<int64_t>(maskShape, targetShape);
    auto repeatDimsAttr = ShapeAttr::get(rewriter.getContext(), repeatDims);

    auto broadcastType =
        utils::RankedTensorTypeFactory::create(maskType, targetShape);

    auto repeatOp =
        rewriter.create<RepeatOp>(loc, broadcastType, mask, repeatDimsAttr);

    return repeatOp.getResult();
  }

  // Reverts 4D output back to 3D if original query was 3D
  Value revertTo3DIfNeeded(Value output, Value originalQuery,
                           mlir::PatternRewriter &rewriter,
                           Location loc) const {
    auto originalQueryType =
        mlir::cast<RankedTensorType>(originalQuery.getType());

    if (originalQueryType.getRank() == 3) {
      auto originalShape = originalQueryType.getShape();
      auto typedOutput =
          mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(output);
      return ttir_to_ttnn::utils::generateReshape(typedOutput, originalShape,
                                                  rewriter, loc)
          .getResult();
    }

    return output;
  }

  static bool isDecodeCase(RankedTensorType queryType) {
    return queryType.getShape()[TensorDims::kSeqLen] == 1;
  }
};

//
// Decode-specific pattern (query seq_len == 1)
//
class SDPADecodeFusing : public SDPAFusingBase {
public:
  using SDPAFusingBase::SDPAFusingBase;

  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Match the attention pattern
    AttentionComponents components;
    if (!matchAttentionPattern(srcOp.getOperation(), components)) {
      return failure();
    }

    // Validate this is decode case
    auto queryType =
        mlir::dyn_cast<RankedTensorType>(components.query.getType());
    if (!isDecodeCase(queryType)) {
      return failure();
    }

    // Validate components for decode
    if (failed(components.canFuse())) {
      return failure();
    }

    // Perform decode-specific fusion
    return fuseToDecodeSDPA(components, rewriter);
  }

private:
  LogicalResult fuseToDecodeSDPA(AttentionComponents &components,
                                 mlir::PatternRewriter &rewriter) const {
    Location loc = components.attentionMatmul.getLoc();

    // Prepare query: transform to [1, batch, num_heads, head_size]
    Value query =
        prepareQueryForDecode(components.query, components.key, rewriter, loc);
    if (!query) {
      return failure();
    }

    // Broadcast mask if present
    Value mask =
        broadcastMask(components.mask, query, components.key, rewriter, loc);

    // Create decode SDPA operation
    auto queryType = mlir::cast<RankedTensorType>(query.getType());
    auto sdpaResultType =
        utils::RankedTensorTypeFactory::create(queryType, queryType.getShape());

    Value sdpaResult =
        rewriter
            .create<ScaledDotProductAttentionDecodeOp>(
                loc, sdpaResultType, query, components.key, components.value,
                /*is_causal=*/false, mask,
                /*cur_pos_tensor=*/nullptr,
                /*attention_sink=*/nullptr,
                rewriter.getF32FloatAttr(components.scale),
                /*memory_config=*/nullptr)
            .getResult();

    // Revert output to original shape
    sdpaResult =
        revertDecodeOutput(sdpaResult, components.query, rewriter, loc);

    rewriter.replaceOp(components.attentionMatmul, sdpaResult);
    return success();
  }

  Value prepareQueryForDecode(Value query, Value key,
                              mlir::PatternRewriter &rewriter,
                              Location loc) const {
    // Reshape 3D query to 4D if needed
    query = reshapeQueryTo4D(query, key, rewriter, loc);

    auto queryType = mlir::cast<RankedTensorType>(query.getType());
    auto queryShape = queryType.getShape();

    // Permute from [batch, num_heads, 1, head_size] to [1, batch, num_heads,
    // head_size]
    if (queryShape[TensorDims::kSeqLen] == 1) {
      SmallVector<int64_t> permuteIndices = {
          TensorDims::kSeqLen, TensorDims::kBatch, TensorDims::kNumHeads,
          TensorDims::kHeadSize};
      SmallVector<int64_t> permutedShape =
          ttmlir::utils::applyPermutation(queryShape, permuteIndices);
      auto permutedType =
          utils::RankedTensorTypeFactory::create(queryType, permutedShape);

      query =
          rewriter
              .create<PermuteOp>(loc, permutedType, query,
                                 rewriter.getDenseI64ArrayAttr(permuteIndices),
                                 /*memory_config=*/nullptr,
                                 /*pad_value=*/nullptr)
              .getResult();
    }

    return query;
  }

  Value revertDecodeOutput(Value sdpa, Value originalQuery,
                           mlir::PatternRewriter &rewriter,
                           Location loc) const {
    auto originalQueryType =
        mlir::cast<RankedTensorType>(originalQuery.getType());
    auto sdpaOutputType = mlir::cast<RankedTensorType>(sdpa.getType());
    auto originalShape = originalQueryType.getShape();
    auto outputShape = sdpaOutputType.getShape();
    int64_t originalRank = originalQueryType.getRank();

    // Permute from [1, batch, num_heads, head_size] back to
    // [batch, num_heads, 1, head_size]
    if (originalShape[originalRank - 2] == 1) {
      SmallVector<int64_t> permuteIndices = {1, 2, 0, 3};
      SmallVector<int64_t> permutedShape =
          ttmlir::utils::applyPermutation(outputShape, permuteIndices);
      auto permutedType =
          utils::RankedTensorTypeFactory::create(sdpaOutputType, permutedShape);

      sdpa =
          rewriter
              .create<PermuteOp>(loc, permutedType, sdpa,
                                 rewriter.getDenseI64ArrayAttr(permuteIndices),
                                 /*memory_config=*/nullptr,
                                 /*pad_value=*/nullptr)
              .getResult();
    }

    // Revert to 3D if needed
    return revertTo3DIfNeeded(sdpa, originalQuery, rewriter, loc);
  }
};

//
// Prefill-specific pattern (query seq_len > 1)
//
class SDPAPrefillFusing : public SDPAFusingBase {
public:
  using SDPAFusingBase::SDPAFusingBase;

  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Match the attention pattern
    AttentionComponents components;
    if (!matchAttentionPattern(srcOp.getOperation(), components)) {
      return failure();
    }

    // Validate this is prefill case (NOT decode)
    auto queryType =
        mlir::dyn_cast<RankedTensorType>(components.query.getType());
    if (isDecodeCase(queryType)) {
      return failure();
    }

    if (failed(components.canFuse())) {
      return failure();
    }

    return fuseToPrefillSDPA(components, rewriter);
  }

private:
  LogicalResult fuseToPrefillSDPA(AttentionComponents &components,
                                  mlir::PatternRewriter &rewriter) const {
    Location loc = components.attentionMatmul.getLoc();

    // Prepare query: ensure it's 4D [batch, num_heads, seq_len, head_size]
    Value query =
        reshapeQueryTo4D(components.key, components.key, rewriter, loc);
    if (!query) {
      return failure();
    }

    // Broadcast mask if present
    Value mask =
        broadcastMask(components.mask, query, components.key, rewriter, loc);

    // Create prefill SDPA operation
    auto queryType = mlir::cast<RankedTensorType>(query.getType());
    auto sdpaResultType =
        utils::RankedTensorTypeFactory::create(queryType, queryType.getShape());

    Value sdpaResult =
        rewriter
            .create<ScaledDotProductAttentionOp>(
                loc, sdpaResultType, query, components.key, components.value,
                mask,
                /*is_causal=*/false, rewriter.getF32FloatAttr(components.scale),
                /*sliding_window_size=*/nullptr,
                /*memory_config=*/nullptr)
            .getResult();

    // Revert output to original shape if needed
    sdpaResult =
        revertTo3DIfNeeded(sdpaResult, components.query, rewriter, loc);

    rewriter.replaceOp(components.attentionMatmul, sdpaResult);
    return success();
  }
};

// This is rope fusing pattern. Given this formula:
// (x * cos) + (rotate_half(x) * sin)
// into ttnn rotary_embedding op from ttnn.
//
// rotate_half is defined as half rotation of the last dimension:
// rotate_half([x1, x2, x3, x4]) = [-x3, -x4, x1, x2]
//
// This pattern is broken down into two sub-patterns:
// 1. unrotatedProjection: matches x * cos
// 2. rotatedProjection: matches rotate_half(x) * sin
//
// unrotatedProjection comes as following sequence of operations:
// cos_unsqueezed = unsqueeze(cos) - where unsqueeze will be reshape which
// prepends
//  dimensions to cos to match x's rank.
// unrotated_projection = multiply(x, cos_unsqueezed)
//
// rotatedProjections comes as following sequence of operations:
// sin_unsqueezed = unsqueeze(sin) - where unsqueeze will be reshape which
// prepends
//  dimensions to sin to match x's rank.
// rotated_second_half = slice(x, ..., start = last_dim/2, end = last_dim)
// neg_rotated_second_half = negate(rotated_second_half)
// rotated_first_half = slice(x, ..., start = 0, end = last_dim/2)
// rotated_x = concatenate(neg_rotated_second_half, rotated_first_half, axis =
// last_dim) rotated_projection = multiply(rotated_x, sin_unsqueezed)
//
// Finally there is an add operation which adds unrotated_projection and
// rotated_projection. x_embedding = add(unrotated_projection,
// rotated_projection)
//
// This whole pattern is replaced with ttnn.rotary_embedding op.
// This op accepts x, cos, sin and trans_mat. First three are self explanatory.
// Trans matrix is used to perform the half rotation. If we take that we have
// x of shape [batch_size, num_heads, seq_len, head_dim] then trans mat
// will be of shape[head_dim, head_dim] and will look like this (lets take
// head_dim = 4):
// [[0, 0, 1, 0],
//  [0, 0, 0, 1],
//  [-1,0, 0, 0],
//  [0,-1, 0, 0]]
//
// So when we multiply x with this matrix we get the desired half rotation.
class RoPEFusing : public mlir::OpRewritePattern<AddOp> {
  using RoPEFusing::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const override {
    // Match the final add: add(unrotated_projection, rotated_projection)
    Value lhs = srcOp.getOperand(0);
    Value rhs = srcOp.getOperand(1);

    // Try to identify which operand is unrotated and which is rotated
    auto lhsMul = lhs.getDefiningOp<MultiplyOp>();
    auto rhsMul = rhs.getDefiningOp<MultiplyOp>();

    if (!lhsMul || !rhsMul) {
      return failure();
    }

    // Try both orderings
    if (auto result = tryMatch(srcOp, lhsMul, rhsMul, rewriter);
        result.succeeded()) {
      return result;
    }
    if (auto result = tryMatch(srcOp, rhsMul, lhsMul, rewriter);
        result.succeeded()) {
      return result;
    }

    return failure();
  }

private:
  mlir::LogicalResult tryMatch(AddOp srcOp, MultiplyOp unrotatedMul,
                               MultiplyOp rotatedMul,
                               mlir::PatternRewriter &rewriter) const {
    // Match unrotated projection: multiply(x, cos_unsqueezed)
    Value xUnrotated, cos;
    if (!matchUnrotatedProjection(unrotatedMul, xUnrotated, cos)) {
      return failure();
    }

    // Match rotated projection: multiply(rotated_x, sin_unsqueezed)
    Value xRotated, sin;
    if (!matchRotatedProjection(rotatedMul, xRotated, sin)) {
      return failure();
    }

    // Verify that both projections use the same input x
    if (xUnrotated != xRotated) {
      return failure();
    }

    if (!cos || !sin) {
      return failure();
    }

    // Get input tensor shape to determine head_dim
    auto inputType = mlir::cast<RankedTensorType>(xUnrotated.getType());
    if (!inputType || inputType.getRank() < 1) {
      return failure();
    }

    int64_t headDim = inputType.getShape()[inputType.getRank() - 1];
    if (headDim <= 0 || headDim % 2 != 0) {
      return failure();
    }

    // Create rotary_embedding op
    auto resultType = srcOp.getType();
    auto ropeOp = rewriter.create<RotaryEmbeddingOp>(
        srcOp.getLoc(), resultType, xUnrotated, cos, sin,
        /*token_index=*/nullptr,
        /*memory_config=*/nullptr, /*compute_config=*/nullptr);

    rewriter.replaceOp(srcOp, ropeOp.getResult());
    return mlir::success();
  }

  bool matchUnrotatedProjection(MultiplyOp mulOp, Value &x,
                                Value &cosUnsqueezed) const {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    // Check if one operand is the unsqueezed cos
    if (isUnsqueezedTensor(rhs)) {
      x = lhs;
      cosUnsqueezed = rhs;
      return true;
    }
    if (isUnsqueezedTensor(lhs)) {
      x = rhs;
      cosUnsqueezed = lhs;
      return true;
    }
    return false;
  }

  bool matchRotatedProjection(MultiplyOp mulOp, Value &x,
                              Value &sinUnsqueezed) const {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    // One operand should be unsqueezed sin, the other should be rotated x
    Value rotatedX = nullptr;
    if (isUnsqueezedTensor(rhs)) {
      rotatedX = lhs;
      sinUnsqueezed = rhs;
    } else if (isUnsqueezedTensor(lhs)) {
      rotatedX = rhs;
      sinUnsqueezed = lhs;
    } else {
      return false;
    }

    // Match the rotation pattern: concat(neg(second_half), first_half)
    auto concatOp = rotatedX.getDefiningOp<ConcatOp>();
    if (!concatOp || concatOp.getNumOperands() != 2) {
      return false;
    }

    Value negHalf = concatOp.getOperand(0);
    Value firstHalf = concatOp.getOperand(1);

    // Check that first operand is negated second half
    auto negOp = negHalf.getDefiningOp<NegOp>();
    if (!negOp) {
      return false;
    }

    Value secondHalf = negOp.getInput();

    // Both halves should be slices of the same input
    auto firstSlice = firstHalf.getDefiningOp<SliceStaticOp>();
    auto secondSlice = secondHalf.getDefiningOp<SliceStaticOp>();

    if (!firstSlice || !secondSlice) {
      return false;
    }

    Value input1 = firstSlice.getInput();
    Value input2 = secondSlice.getInput();

    if (input1 != input2) {
      return false;
    }

    x = input1;

    // rotary_embedding requires 4D input in form
    // [batch_size, num_heads, seq_len, head_dim]
    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(x.getType()).getShape();
    if (inputShape.size() != 4) {
      return false;
    }

    int64_t lastDim = inputShape.back();
    if (lastDim <= 0 || lastDim % 2 != 0) {
      return false;
    }

    int64_t halfDim = lastDim / 2;

    // Check first_half slice: [0, halfDim)
    auto firstBegins =
        ttmlir::utils::getIntegerVector<int64_t>(firstSlice.getBegins());
    auto firstEnds =
        ttmlir::utils::getIntegerVector<int64_t>(firstSlice.getEnds());
    if (!firstBegins || !firstEnds || firstBegins->back() != 0 ||
        firstEnds->back() != halfDim) {
      return false;
    }

    // Check second_half slice: [halfDim, lastDim)
    auto secondBegins =
        ttmlir::utils::getIntegerVector<int64_t>(secondSlice.getBegins());
    auto secondEnds =
        ttmlir::utils::getIntegerVector<int64_t>(secondSlice.getEnds());
    if (!secondBegins || !secondEnds || secondBegins->back() != halfDim ||
        secondEnds->back() != lastDim) {
      return false;
    }

    return true;
  }

  bool isUnsqueezedTensor(Value val) const {
    // An unsqueezed tensor is typically a reshape that adds dimensions
    auto reshapeOp = val.getDefiningOp<ReshapeOp>();
    if (!reshapeOp) {
      return false;
    }

    auto inputShape =
        mlir::cast<RankedTensorType>(reshapeOp.getInput().getType()).getShape();
    SmallVector<int64_t> outputShape(
        mlir::cast<RankedTensorType>(reshapeOp.getType()).getShape());

    // Check that output shape is input shape with one extra leading dimension
    SmallVector<int64_t> expectedShape(inputShape);
    expectedShape.insert(expectedShape.begin(), 1);
    return outputShape == expectedShape;
  }
};

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
        TTNNMatmulAndLinearWithActivation<LinearOp, SigmoidOp>, RoPEFusing,
        SDPADecodeFusing, SDPAPrefillFusing>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
