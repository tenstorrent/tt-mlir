// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#ifdef TTMLIR_ENABLE_OPMODEL
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

#ifdef TTMLIR_ENABLE_OPMODEL

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
// prepends dimensions to cos to match x's rank.
// unrotated_projection = multiply(x, cos_unsqueezed)
//
// rotatedProjections comes as following sequence of operations:
// sin_unsqueezed = unsqueeze(sin) - where unsqueeze will be reshape which
// prepends dimensions to sin to match x's rank.
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
    Value lhs = srcOp.getLhs();
    Value rhs = srcOp.getRhs();

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

    op_model::ScopedSingletonDeviceGuard deviceGuard;

    // Create rotary_embedding op
    auto resultType = srcOp.getType();
    auto ropeOp = rewriter.create<RotaryEmbeddingOp>(
        srcOp.getLoc(), resultType, xUnrotated, cos, sin,
        /*token_index=*/nullptr,
        /*memory_config=*/nullptr, /*compute_config=*/nullptr);

    // Extract input layouts from the operation
    std::vector<TTNNLayoutAttr> inputLayouts =
        utils::extractInputLayouts(ropeOp.getOperation());

    OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
    auto result = op_constraint_validation::validateOperation(
        ropeOp.getOperation(), inputLayouts, config);

    if (!result.isSuccess()) {
      rewriter.eraseOp(ropeOp);
      return failure();
    }

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

// ============================================================================
// SDPA Fusing
// ============================================================================
//
// Matches Scaled Dot Product Attention:
//   Attention(Q, K, V) = softmax((Q @ K^T) * scale + mask) @ V
//
// Anchors on final matmul (attention @ V) and walks backward through:
//   matmul -> [transparent] -> [where] -> softmax -> [transparent] ->
//   [add(mask)] -> [transparent] -> [multiply(scale)] -> [transparent] ->
//   matmul
//
// Uses a generic skipTransparent() utility to handle type conversions and
// layout ops that don't change semantics, making the pattern robust to
// variations in the IR.
//
class SDPAFusing : public mlir::OpRewritePattern<MatmulOp> {
  using SDPAFusing::OpRewritePattern<MatmulOp>::OpRewritePattern;

  // SDPA Query, Key, Value tensors have shape [B, H, S, D] (Batch, NumHeads,
  // SeqLen, HeadDim).
  static constexpr int64_t kNumHeadsDim = 1;
  static constexpr int64_t kSeqLenDim = 2;

  // Permutation to convert query from [B, H, S, D] -> [S, B, H, D] for SDPA
  // decode op.
  static constexpr std::array<int64_t, 4> kToDecodePermutation = {2, 0, 1, 3};

  // Permutation to un-transpose key from [B, H, D, S] -> [B, H, S, D].
  // Used when key comes from SplitQueryKeyValueAndSplitHeadsOp with
  // transpose_key=true.
  static constexpr std::array<int64_t, 4> kUnTransposeKeyPermutation = {0, 1, 3,
                                                                        2};

public:
  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    SDPAComponents c;
    c.attentionMatmul = srcOp;
    c.value = skipTransparent(srcOp.getB());

    // Match: matmul -> [where] -> softmax -> score
    if (!matchSoftmaxPath(srcOp.getA(), c)) {
      return failure();
    }

    if (!matchScoreComputation(c.softmax.getInput(), c)) {
      return failure();
    }

    if (!validateSemantics(c)) {
      return failure();
    }

    return createSDPAOp(rewriter, c);
  }

private:
  struct SDPAComponents {
    Value query, key, value, mask;
    std::optional<float> scale;
    MatmulOp attentionMatmul;
    SoftmaxOp softmax;
    Operation *scoreOp = nullptr;
  };

  // ============================================================================
  // Transparent Op Utilities
  // ============================================================================

  // Operations that don't change semantic meaning - can be traced through.
  static bool isTransparentOp(Operation *op) {
    return isa<ReshapeOp, RepeatOp, PermuteOp, TypecastOp>(op);
  }

  // Skip through transparent ops to find the semantic operation.
  Value skipTransparent(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (!isTransparentOp(defOp)) {
        break;
      }
      v = defOp->getOperand(0);
    }
    return v;
  }

  // Check if key is transposed by looking at its source operation.
  // Returns true if key came from SplitQueryKeyValueAndSplitHeadsOp with
  // transpose_key=true.
  bool isKeyTransposed(Value key) const {
    Operation *defOp = key.getDefiningOp();
    if (auto splitOp =
            dyn_cast_or_null<SplitQueryKeyValueAndSplitHeadsOp>(defOp)) {
      return splitOp.getTransposeKey();
    }
    return false;
  }

  // ============================================================================
  // Pattern Matching with Backtracking
  // ============================================================================

  // Match: [transparent] -> [where(cond, zeros, softmax)] -> softmax
  //    or: [transparent] -> softmax
  bool matchSoftmaxPath(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Try where(cond, zeros, softmax) pattern first
    if (auto whereOp = v.getDefiningOp<WhereOp>()) {
      Value softmaxCandidate = skipTransparent(whereOp.getThird());
      if (auto softmax = softmaxCandidate.getDefiningOp<SoftmaxOp>()) {
        c.softmax = softmax;
        return true;
      }
    }

    // Direct softmax
    if (auto softmax = v.getDefiningOp<SoftmaxOp>()) {
      c.softmax = softmax;
      return true;
    }

    return false;
  }

  // Match score computation with backtracking for different orderings.
  // Patterns (in order of priority):
  //   1. [transparent] -> linear(Q_scaled, K_scaled, mask)
  //   2. [transparent] -> add(score_chain, mask)
  //   3. [transparent] -> score_chain (no mask)
  bool matchScoreComputation(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Try linear(Q_scaled, K_scaled, mask) first
    if (auto linearOp = v.getDefiningOp<LinearOp>()) {
      c.scoreOp = linearOp;
      if (!extractQKWithScales(linearOp.getA(), linearOp.getB(), c)) {
        return false;
      }
      if (linearOp.getBias()) {
        c.mask = skipTransparent(linearOp.getBias());
      }
      return true;
    }

    // Try add(score, mask) with both operand orderings
    if (auto addOp = v.getDefiningOp<AddOp>()) {
      // Try lhs as score, rhs as mask
      if (matchScoreChain(addOp.getLhs(), c)) {
        c.mask = skipTransparent(addOp.getRhs());
        return true;
      }
      // Try rhs as score, lhs as mask
      if (matchScoreChain(addOp.getRhs(), c)) {
        c.mask = skipTransparent(addOp.getLhs());
        return true;
      }
      return false;
    }

    // No add - try direct score chain (no mask)
    return matchScoreChain(v, c);
  }

  // Match: [transparent] -> [multiply(*, scale)] -> [transparent] -> matmul
  // Extracts scale if present, then matches the Q@K matmul.
  bool matchScoreChain(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Optional multiply for scale (post-matmul scaling)
    if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
      if (auto scale = extractConstant(mulOp.getRhs())) {
        c.scale = scale;
        v = skipTransparent(mulOp.getLhs());
      } else if (auto scale = extractConstant(mulOp.getLhs())) {
        c.scale = scale;
        v = skipTransparent(mulOp.getRhs());
      }
    }

    // Must end with matmul (different from attention matmul)
    if (auto matmul = v.getDefiningOp<MatmulOp>()) {
      if (matmul != c.attentionMatmul) {
        c.scoreOp = matmul;
        if (!extractQKWithScales(matmul.getA(), matmul.getB(), c)) {
          return false;
        }
        return true;
      }
    }

    return false;
  }

  // ============================================================================
  // Q/K Extraction with Scale Handling
  // ============================================================================

  // Extract tensor and its scale, handling TM reordering from
  // EraseInverseOps. Skips transparent ops and extracts scale from multiply
  // if present.
  std::pair<Value, std::optional<float>> extractTensorWithScale(Value v) const {
    std::optional<float> scale;

    v = skipTransparent(v);

    // Check if we hit a multiply (scale applied to tensor)
    if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
      if (auto s = extractConstant(mulOp.getRhs())) {
        scale = s;
        v = skipTransparent(mulOp.getLhs());
      } else if (auto s = extractConstant(mulOp.getLhs())) {
        scale = s;
        v = skipTransparent(mulOp.getRhs());
      }
    }

    return {v, scale};
  }

  // Returns false if we find both post-matmul scaling AND pre-scaling on Q/K,
  // which would indicate this is likely not a standard SDPA pattern.
  bool extractQKWithScales(Value a, Value b, SDPAComponents &c) const {
    auto [query, qScale] = extractTensorWithScale(a);
    auto [key, kScale] = extractTensorWithScale(b);

    // Reject if we found both post-matmul scale and pre-scaling on Q/K.
    // Standard SDPA uses one or the other, not both.
    bool hasPostMatmulScale = c.scale.has_value();
    bool hasPreScale = qScale.has_value() || kScale.has_value();
    if (hasPostMatmulScale && hasPreScale) {
      return false;
    }

    c.query = query;
    c.key = key;

    // Combine pre-scales if present: Q*s and K*s → combined scale = s*s
    if (hasPreScale) {
      float qs = qScale.value_or(1.0f);
      float ks = kScale.value_or(1.0f);
      c.scale = qs * ks;
    }
    return true;
  }

  // ============================================================================
  // Constant Extraction
  // ============================================================================

  std::optional<float> extractConstant(Value v) const {
    // Skip transparent ops to find the actual constant
    v = skipTransparent(v);

    // Direct FullOp
    if (auto fullOp = v.getDefiningOp<FullOp>()) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        return attr.getValue().convertToFloat();
      }
    }

    // Try load_cached - look up the const_eval function and find FullOp
    // inside
    if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
      auto callee = loadCached.getCallee();
      auto moduleOp = loadCached->getParentOfType<ModuleOp>();
      if (!moduleOp) {
        return std::nullopt;
      }

      auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
      if (!funcOp) {
        return std::nullopt;
      }

      // Walk the function body to find a FullOp
      std::optional<float> result;
      funcOp.walk([&](FullOp fullOp) {
        if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
          result = attr.getValue().convertToFloat();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      return result;
    }

    return std::nullopt;
  }

  // ============================================================================
  // Key Un-transpose
  // ============================================================================

  // Check if key is transposed by looking at its source operation.
  // If key came from SplitQueryKeyValueAndSplitHeadsOp with
  // transpose_key=true, generate a permute to restore the expected shape [B,
  // H, S, D] for SDPA.
  Value unTransposeKeyIfNeeded(Value query, Value key,
                               mlir::PatternRewriter &rewriter,
                               Location loc) const {
    if (!isKeyTransposed(key)) {
      return key;
    }

    // Generate permute to un-transpose: [B, H, D, S] -> [B, H, S, D]
    return ttir_to_ttnn::utils::generatePermute(
        mlir::cast<TypedValue<RankedTensorType>>(key),
        llvm::to_vector(kUnTransposeKeyPermutation), rewriter, loc);
  }

  // ============================================================================
  // Validation
  // ============================================================================

  bool validateSemantics(const SDPAComponents &c) const {
    if (!c.query || !c.key || !c.value || !c.softmax || !c.scoreOp) {
      return false;
    }

    auto qType = mlir::dyn_cast<RankedTensorType>(c.query.getType());
    auto kType = mlir::dyn_cast<RankedTensorType>(c.key.getType());
    auto vType = mlir::dyn_cast<RankedTensorType>(c.value.getType());

    if (!qType || !kType || !vType) {
      return false;
    }

    // Validate shapes: Q, K, V should be 4D tensors.
    // Q shape: [batch, num_heads, seq_q, head_dim]
    // K shape: [batch, num_kv_heads, seq_k, head_dim] or
    //          [batch, num_kv_heads, head_dim, seq_k] if transposed
    // V shape: [batch, num_kv_heads, seq_v, head_dim]
    auto qShape = qType.getShape();
    auto kShape = kType.getShape();
    auto vShape = vType.getShape();

    if (qShape.size() != 4 || kShape.size() != 4 || vShape.size() != 4) {
      return false;
    }

    int64_t qHeadDim = qShape[3];
    int64_t vSeqLen = vShape[kSeqLenDim];
    int64_t vHeadDim = vShape[3];

    // Determine if key is transposed by checking if it came from
    // SplitQueryKeyValueAndSplitHeadsOp with transpose_key=true.
    // If K is not transposed: [B, H, S, D] -> kShape[3] == head_dim
    // If K is transposed: [B, H, D, S] -> kShape[2] == head_dim
    bool keyTransposed = isKeyTransposed(c.key);
    int64_t kSeqLen = keyTransposed ? kShape[3] : kShape[kSeqLenDim];
    int64_t kHeadDim = keyTransposed ? kShape[kSeqLenDim] : kShape[3];

    // Key and Value must have the same sequence length.
    if (kSeqLen != vSeqLen) {
      return false;
    }

    // Head dimensions must match across Q, K, V.
    if (qHeadDim != kHeadDim || kHeadDim != vHeadDim) {
      return false;
    }

    // Batch dimensions must match.
    if (qShape[0] != kShape[0] || kShape[0] != vShape[0]) {
      return false;
    }

    // Validate num_heads:
    // - K and V must have the same num_heads (num_kv_heads)
    // - Q's num_heads must be divisible by num_kv_heads (for GQA/MQA support)
    int64_t qNumHeads = qShape[kNumHeadsDim];
    int64_t kNumHeads = kShape[kNumHeadsDim];
    int64_t vNumHeads = vShape[kNumHeadsDim];

    if (kNumHeads != vNumHeads) {
      return false;
    }

    if (qNumHeads % kNumHeads != 0) {
      return false;
    }

    if (!c.softmax->hasOneUse()) {
      return false;
    }

    // If softmax feeds into a typecast, verify the typecast also has one use.
    if (auto *softmaxUser = *c.softmax->getUsers().begin()) {
      if (isa<TypecastOp>(softmaxUser) && !softmaxUser->hasOneUse()) {
        return false;
      }
    }

    if (c.scale.has_value() && (*c.scale <= 0.0f || *c.scale > 1.0f)) {
      return false;
    }

    return true;
  }

  // Broadcast attention mask to the required shape for SDPA operations.
  //
  // For regular SDPA:
  //   Target mask shape: [batch, 1, query_seq, key_seq]
  //   - Dimension 1 (heads) stays as 1
  //
  // For decode SDPA:
  //   Target mask shape: [batch, 1, num_heads, key_seq]
  //   - Dimension 1 is query seq (always 1 for decode)
  //   - Dimension 2 must explicitly match num_heads
  Value broadcastMaskForSDPA(Value mask, RankedTensorType qType,
                             RankedTensorType kType, bool isDecode,
                             mlir::PatternRewriter &rewriter,
                             Location loc) const {
    if (!mask) {
      return mask;
    }

    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto qShape = qType.getShape();
    auto kShape = kType.getShape();

    // Compute target mask shape based on SDPA variant.
    // Q shape: [batch, num_heads, seq_len, head_dim]
    // K shape: [batch, num_heads, key_seq, head_dim]
    SmallVector<int64_t> targetShape;
    if (isDecode) {
      // Decode: [batch, 1, num_heads, key_seq]
      targetShape = {qShape[0], 1, qShape[kNumHeadsDim], kShape[kSeqLenDim]};
    } else {
      // Regular: [batch, 1, query_seq, key_seq]
      targetShape = {qShape[0], 1, qShape[kSeqLenDim], kShape[kSeqLenDim]};
    }

    // Check if broadcast is needed.
    if (llvm::equal(maskType.getShape(), targetShape)) {
      return mask;
    }

    auto broadcastType =
        utils::RankedTensorTypeFactory::create(maskType, targetShape);
    auto broadcastDims = ttmlir::utils::getBroadcastDimensions<int64_t>(
        maskType.getShape(), targetShape);
    auto shapeAttr = ShapeAttr::get(rewriter.getContext(), broadcastDims);

    return rewriter.create<RepeatOp>(loc, broadcastType, mask, shapeAttr);
  }

  mlir::LogicalResult createSDPAOp(mlir::PatternRewriter &rewriter,
                                   SDPAComponents &c) const {
    // When no scale is found in the pattern, explicitly set scale=1.0 to
    // prevent tt-metal from applying the default 1/sqrt(head_dim) scaling.
    float scale = c.scale.value_or(1.0f);
    FloatAttr scaleAttr = rewriter.getF32FloatAttr(scale);

    // If key is transposed (coming from SplitQueryKeyValueAndSplitHeadsOp
    // with transpose_key=true) [B, H, D, S], un-transpose it to restore [B,
    // H, S, D] shape expected by SDPA.
    Value key = unTransposeKeyIfNeeded(c.query, c.key, rewriter,
                                       c.attentionMatmul.getLoc());

    auto qType = mlir::cast<RankedTensorType>(c.query.getType());
    auto qShape = qType.getShape();
    auto kType = mlir::cast<RankedTensorType>(key.getType());

    // Check if this is decode mode (query seq_len == 1)
    // Query shape: [batch x num_heads x seq_len x head_size]
    bool isDecode = qShape.size() == 4 && qShape[kSeqLenDim] == 1;

    // Broadcast mask to the required shape for the SDPA variant.
    Value attentionMask = broadcastMaskForSDPA(
        c.mask, qType, kType, isDecode, rewriter, c.attentionMatmul.getLoc());

    if (isDecode) {
      // Permute query: [B, H, 1, D] -> [1, B, H, D]
      Value permutedQuery = ttir_to_ttnn::utils::generatePermute(
          mlir::cast<TypedValue<RankedTensorType>>(c.query),
          llvm::to_vector(kToDecodePermutation), rewriter,
          c.attentionMatmul.getLoc());

      auto decodeOp = rewriter.create<ScaledDotProductAttentionDecodeOp>(
          c.attentionMatmul.getLoc(), permutedQuery.getType(), permutedQuery,
          key, c.value,
          /*is_causal=*/rewriter.getBoolAttr(false), attentionMask,
          /*cur_pos_tensor=*/Value(),
          /*attention_sink=*/Value(), scaleAttr,
          /*memory_config=*/MemoryConfigAttr(),
          /*program_config=*/SDPAProgramConfigAttr());

      // Permute result back: [1, B, H, D] -> [B, H, 1, D].
      rewriter.replaceOp(
          c.attentionMatmul,
          ttir_to_ttnn::utils::generatePermute(
              decodeOp.getResult(),
              ttmlir::utils::inversePermutation(kToDecodePermutation), rewriter,
              c.attentionMatmul.getLoc()));
    } else {
      auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
          c.attentionMatmul.getLoc(), c.query.getType(), c.query, key, c.value,
          attentionMask,
          /*is_causal=*/rewriter.getBoolAttr(false), scaleAttr,
          /*sliding_window_size=*/IntegerAttr(),
          /*memory_config=*/MemoryConfigAttr());
      rewriter.replaceOp(c.attentionMatmul, sdpaOp.getResult());
    }

    return mlir::success();
  }
};

#endif // TTMLIR_ENABLE_OPMODEL

// ============================================================================
// SplitQKVFromSlices Pattern
// ============================================================================
//
// This pattern fuses a matmul followed by slices and reshapes into a single
// SplitQueryKeyValueAndSplitHeadsOp. It detects the common pattern where
// a fused QKV projection (matmul) is followed by three slices to separate
// Q, K, V components, each followed by a reshape to split heads.
//
// Pattern matched:
//   matmul -> slice[0:q_size] -> reshape[B,H,S,D] (Q)
//          -> slice[q_size:q_size+k_size] -> reshape[B,H,S,D] (K)
//          -> slice[q_size+k_size:total] -> reshape[B,H,S,D] (V)
//
// Replaced with:
//   matmul -> reshape[B,S,total] -> SplitQueryKeyValueAndSplitHeadsOp -> Q, K,
//   V
//
// Role Identification Challenge:
// -----------------------------
// The slices can appear in any order in the concat (e.g., K,V,Q instead of
// Q,K,V), so we need to identify which slice/reshape chain corresponds to
// Query, Key, or Value. We use two approaches:
//
// 1. Name-based identification (fast path):
//    Check weight tensor names from the concat inputs for patterns like
//    "q_proj", "k_proj", "v_proj". This works when weights have standard
//    naming conventions.
//
// 2. Role-based identification (fallback):
//    Trace forward from each reshape output through the IR (including RoPE,
//    permutes, cache updates) to find SDPA ops. The chain's role is determined
//    by which SDPA operand (query, key, or value) it eventually feeds into.
//
// Once roles are identified, we reorder the concat inputs to Q,K,V order
// so the fused SplitQueryKeyValueAndSplitHeadsOp produces outputs correctly.
//
template <typename MatMulOpType>
class SplitQKVFromSlicesPattern : public mlir::OpRewritePattern<MatMulOpType> {
  using mlir::OpRewritePattern<MatMulOpType>::OpRewritePattern;

  static constexpr unsigned kRoleTraceMaxDepth = 20;

  enum OutputDims {
    O_BATCH = 0,
    O_NUM_HEADS = 1,
    O_SEQ_LEN = 2,
    O_HEAD_DIM = 3,
  };

  enum class QKVRole { Query, Key, Value };

  struct SliceReshapeChain {
    SliceStaticOp sliceOp;
    ReshapeOp reshapeOp;
    int64_t numHeads;
    int64_t headDim;
    int64_t seqLen;
    int64_t sliceStart;
    int64_t sliceEnd;
    std::optional<QKVRole> role;
  };

  // Holds information about the concat feeding the matmul RHS,
  // regardless of whether it's direct or via load_cached.
  struct ConcatInfo {
    ConcatOp concatOp;
    // The weight values used for name-based identification.
    // For direct concat: same as concat inputs.
    // For load_cached: the load_cached operands (which map to block args).
    llvm::SmallVector<Value> weightInputs;
  };

public:
  mlir::LogicalResult
  matchAndRewrite(MatMulOpType matmulOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Collect matmul -> slice -> reshape chains for each Q, K, V output.
    llvm::SmallVector<SliceReshapeChain> chains;
    size_t sliceDim;
    if (!collectSliceReshapeChains(matmulOp, chains, sliceDim)) {
      return mlir::failure();
    }
    if (chains.size() != 3) {
      return mlir::failure();
    }
    if (!validateAndSortSlices(matmulOp, chains, sliceDim)) {
      return mlir::failure();
    }

    // Get concat info from matmul RHS (direct or via load_cached).
    std::optional<ConcatInfo> concatInfo = getConcatInfo(matmulOp.getB());
    if (!concatInfo) {
      return mlir::failure();
    }

    // Identify Q, K, V roles by weight names or by tracing to SDPA ops.
    if (!identifyQKVRoles(chains, *concatInfo)) {
      return mlir::failure();
    }

    // Extract chains by role and validate dimensions.
    auto [qChain, kChain, vChain] = getChainsByRole(chains);
    if (!qChain || !kChain || !vChain) {
      return mlir::failure();
    }
    if (!validateQKVDimensions(*qChain, *kChain, *vChain)) {
      return mlir::failure();
    }

    // Reorder concat inputs to Q, K, V order and create fused op.
    reorderConcatInputs(rewriter, concatInfo->concatOp, chains);

    return createFusedOp(rewriter, matmulOp, *qChain, *kChain, *vChain);
  }

private:
  std::tuple<SliceReshapeChain *, SliceReshapeChain *, SliceReshapeChain *>
  getChainsByRole(llvm::SmallVector<SliceReshapeChain> &chains) const {
    SliceReshapeChain *q = nullptr, *k = nullptr, *v = nullptr;
    for (auto &chain : chains) {
      if (chain.role == QKVRole::Query) {
        q = &chain;
      } else if (chain.role == QKVRole::Key) {
        k = &chain;
      } else if (chain.role == QKVRole::Value) {
        v = &chain;
      }
    }
    return {q, k, v};
  }

  bool validateQKVDimensions(const SliceReshapeChain &q,
                             const SliceReshapeChain &k,
                             const SliceReshapeChain &v) const {
    return q.headDim == k.headDim && k.headDim == v.headDim &&
           k.numHeads == v.numHeads;
  }

  // Find which dimension is being sliced and return its index along with
  // bounds. Returns nullopt if multiple dimensions are sliced or no dimension
  // is sliced.
  std::optional<std::tuple<size_t, int64_t, int64_t>>
  findSlicedDimensionWithBounds(SliceStaticOp sliceOp,
                                ArrayRef<int64_t> inputShape) const {
    auto begins = sliceOp.getBegins();
    auto ends = sliceOp.getEnds();

    std::optional<std::tuple<size_t, int64_t, int64_t>> result;
    for (size_t dim = 0; dim < begins.size(); ++dim) {
      int64_t start = mlir::cast<mlir::IntegerAttr>(begins[dim]).getInt();
      int64_t end = mlir::cast<mlir::IntegerAttr>(ends[dim]).getInt();

      if (start != 0 || end != inputShape[dim]) {
        if (result.has_value()) {
          return std::nullopt; // Multiple dimensions being sliced
        }
        result = {dim, start, end};
      }
    }
    return result;
  }

  // Trace a value back through permutes to find the block argument.
  std::optional<BlockArgument> traceToBlockArg(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (auto permuteOp = mlir::dyn_cast<PermuteOp>(defOp)) {
        v = permuteOp.getInput();
      } else {
        break;
      }
    }
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(v)) {
      return blockArg;
    }
    return std::nullopt;
  }

  // Get ttir.name attribute from a block argument.
  std::optional<StringRef> getBlockArgName(BlockArgument blockArg) const {
    if (auto funcOp =
            mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      if (auto nameAttr = funcOp.getArgAttrOfType<StringAttr>(
              blockArg.getArgNumber(), "ttir.name")) {
        return nameAttr.getValue();
      }
    }
    return std::nullopt;
  }

  // Trace forward from a value to find which QKV role it plays in attention.
  // Returns the role if the value reaches an SDPA Q/K/V operand. Traces through
  // intermediate ops (RoPE, permutes, etc.).
  std::optional<QKVRole> findQKVRole(Value start) const {
    llvm::SmallVector<std::pair<Value, unsigned>, 32> queue;
    llvm::DenseSet<Value> visited;
    size_t queueIdx = 0;
    queue.push_back({start, 0});
    visited.insert(start);

    std::optional<QKVRole> foundRole;
    while (queueIdx < queue.size()) {
      auto item = queue[queueIdx++];
      Value v = item.first;
      unsigned depth = item.second;
      if (depth >= kRoleTraceMaxDepth) {
        continue;
      }

      for (Operation *user : v.getUsers()) {
        auto role = llvm::TypeSwitch<Operation *, std::optional<QKVRole>>(user)
                        .template Case<ScaledDotProductAttentionOp,
                                       ScaledDotProductAttentionDecodeOp,
                                       PagedScaledDotProductAttentionDecodeOp>(
                            [&](auto op) -> std::optional<QKVRole> {
                              if (op.getQuery() == v) {
                                return QKVRole::Query;
                              }
                              if (op.getKey() == v) {
                                return QKVRole::Key;
                              }
                              if (op.getValue() == v) {
                                return QKVRole::Value;
                              }
                              return std::nullopt;
                            })
                        .Default([](Operation *) { return std::nullopt; });

        if (role) {
          // A single value should not simultaneously feed multiple SDPA operand
          // roles (Q/K/V). If it does, treat it as ambiguous.
          if (!foundRole) {
            foundRole = *role;
          } else if (*foundRole != *role) {
            return std::nullopt;
          }
        }

        // For cache updates, follow the cache tensor to find SDPA consumption
        if (auto cacheUpdateOp = mlir::dyn_cast<PagedUpdateCacheOp>(user)) {
          Value cache = cacheUpdateOp.getCache();
          unsigned nextDepth = depth + 1;
          if (nextDepth < kRoleTraceMaxDepth && visited.insert(cache).second) {
            queue.push_back({cache, nextDepth});
          }
        }

        // Propagate through all ops to handle RoPE and other intermediate ops
        for (Value res : user->getResults()) {
          unsigned nextDepth = depth + 1;
          if (nextDepth < kRoleTraceMaxDepth && visited.insert(res).second) {
            queue.push_back({res, nextDepth});
          }
        }
      }
    }

    return foundRole;
  }

  // Sets the role field on each chain by tracing forward to SDPA ops.
  // Returns true if all chains have a unique Q/K/V role assigned.
  bool identifyQKVByRoles(llvm::SmallVector<SliceReshapeChain> &chains) const {
    if (chains.size() != 3) {
      return false;
    }

    auto roleBit = [](QKVRole r) -> uint8_t {
      switch (r) {
      case QKVRole::Query:
        return 1u << 0;
      case QKVRole::Key:
        return 1u << 1;
      case QKVRole::Value:
        return 1u << 2;
      }
      llvm_unreachable("Unhandled QKVRole");
    };

    uint8_t seenMask = 0;
    llvm::SmallVector<QKVRole, 3> roles;
    roles.reserve(chains.size());

    for (auto &chain : chains) {
      auto role = findQKVRole(chain.reshapeOp.getResult());
      if (!role) {
        return false;
      }

      // Check for duplicates and track which roles are present.
      uint8_t bit = roleBit(*role);
      if (seenMask & bit) {
        return false;
      }
      seenMask |= bit;
      roles.push_back(*role);
    }

    // Require Query, Key, Value all present.
    if (seenMask != ((1u << 0) | (1u << 1) | (1u << 2))) {
      return false;
    }

    // Only commit roles once we've validated uniqueness/completeness.
    for (size_t i = 0; i < chains.size(); ++i) {
      chains[i].role = roles[i];
    }
    return true;
  }

  // Sets the role field on each chain by matching weight names.
  // rhsInputs are concat inputs corresponding to chains (same order after
  // sorting). Returns true if all chains have a unique Q/K/V role assigned.
  bool identifyQKVByNames(llvm::SmallVector<SliceReshapeChain> &chains,
                          ArrayRef<Value> rhsInputs) const {
    if (chains.size() != rhsInputs.size()) {
      return false;
    }

    bool hasQ = false, hasK = false, hasV = false;
    for (size_t i = 0; i < rhsInputs.size(); ++i) {
      auto blockArg = traceToBlockArg(rhsInputs[i]);
      if (!blockArg) {
        return false;
      }

      auto name = getBlockArgName(*blockArg);
      if (!name) {
        return false;
      }

      if (name->contains("q_proj")) {
        if (hasQ) {
          return false;
        }
        chains[i].role = QKVRole::Query;
        hasQ = true;
      } else if (name->contains("k_proj")) {
        if (hasK) {
          return false;
        }
        chains[i].role = QKVRole::Key;
        hasK = true;
      } else if (name->contains("v_proj")) {
        if (hasV) {
          return false;
        }
        chains[i].role = QKVRole::Value;
        hasV = true;
      }
    }

    return hasQ && hasK && hasV;
  }

  // Returns concat info from matmul RHS, handling both direct concat
  // and load_cached wrapping a const_eval function containing concat.
  std::optional<ConcatInfo> getConcatInfo(Value matmulRHS) const {
    // Case 1: Direct concat
    if (auto concatOp = matmulRHS.getDefiningOp<ConcatOp>()) {
      return ConcatInfo{concatOp,
                        llvm::SmallVector<Value>(concatOp.getInputs())};
    }

    // Case 2: load_cached wrapping concat in const_eval function
    if (auto loadCached = matmulRHS.getDefiningOp<ttcore::LoadCachedOp>()) {
      // Ensure load_cached has only one user so we can safely modify the
      // const_eval function's concat.
      if (!loadCached->hasOneUse()) {
        return std::nullopt;
      }

      auto moduleOp = loadCached->getParentOfType<ModuleOp>();
      if (!moduleOp) {
        return std::nullopt;
      }

      auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(loadCached.getCallee());
      if (!funcOp) {
        return std::nullopt;
      }

      // Find concat inside the const_eval function that feeds the return.
      // We expect the concat to be the last operation before return.
      ConcatOp foundConcat = nullptr;
      funcOp.walk([&](func::ReturnOp returnOp) {
        if (returnOp.getNumOperands() == 1) {
          if (auto concat = returnOp.getOperand(0).getDefiningOp<ConcatOp>()) {
            foundConcat = concat;
          }
        }
        return WalkResult::interrupt();
      });

      if (!foundConcat) {
        return std::nullopt;
      }

      return ConcatInfo{foundConcat,
                        llvm::SmallVector<Value>(loadCached.getInputs())};
    }

    return std::nullopt;
  }

  // Identify which chains are Q, K, V and set their role field.
  // Also sets outConcatInfo with the concat and weight inputs.
  bool identifyQKVRoles(llvm::SmallVector<SliceReshapeChain> &chains,
                        ConcatInfo &concatInfo) const {
    if (concatInfo.weightInputs.size() != 3) {
      return false;
    }

    // Try name-based identification first (faster - simple string check on
    // weight names). Fall back to role-based identification if names don't
    // match expected patterns (traces outputs to SDPA ops).
    if (!identifyQKVByNames(chains, concatInfo.weightInputs) &&
        !identifyQKVByRoles(chains)) {
      return false;
    }

    return true;
  }

  // Reorder concat inputs to match Q, K, V order based on chain roles.
  // Chains are sorted by slice position, so chain index == concat input index.
  void reorderConcatInputs(
      mlir::PatternRewriter &rewriter, ConcatOp concatOp,
      const llvm::SmallVector<SliceReshapeChain> &chains) const {
    auto inputs = concatOp.getInputs();
    SmallVector<Value> reorderedInputs(3);
    for (size_t i = 0; i < chains.size(); ++i) {
      if (chains[i].role == QKVRole::Query) {
        reorderedInputs[0] = inputs[i];
      } else if (chains[i].role == QKVRole::Key) {
        reorderedInputs[1] = inputs[i];
      } else if (chains[i].role == QKVRole::Value) {
        reorderedInputs[2] = inputs[i];
      }
    }
    rewriter.modifyOpInPlace(concatOp, [&]() {
      concatOp.getInputsMutable().assign(reorderedInputs);
    });
  }

  bool collectSliceReshapeChains(MatMulOpType matmulOp,
                                 llvm::SmallVector<SliceReshapeChain> &chains,
                                 size_t &outSliceDim) const {
    Value matmulResult = matmulOp.getResult();
    auto matmulShape = matmulOp.getType().getShape();
    std::optional<size_t> sliceDim;

    for (Operation *user : matmulResult.getUsers()) {
      auto sliceOp = mlir::dyn_cast<SliceStaticOp>(user);
      if (!sliceOp) {
        return false;
      }

      if (!sliceOp.getResult().hasOneUse()) {
        return false;
      }

      auto reshapeOp =
          mlir::dyn_cast<ReshapeOp>(*sliceOp.getResult().getUsers().begin());
      if (!reshapeOp) {
        return false;
      }

      // Validate reshape output is 4D (batch, heads, seq, head_dim)
      auto reshapeShape = reshapeOp.getType().getShape();
      if (reshapeShape.size() != 4) {
        return false;
      }

      // Find sliced dimension and bounds
      auto dimAndBounds = findSlicedDimensionWithBounds(sliceOp, matmulShape);
      if (!dimAndBounds.has_value()) {
        return false;
      }
      auto [dim, sliceStart, sliceEnd] = *dimAndBounds;

      // Verify all slices use the same dimension
      if (!sliceDim.has_value()) {
        sliceDim = dim;
      } else if (*sliceDim != dim) {
        return false;
      }

      chains.push_back({
          sliceOp,
          reshapeOp,
          reshapeShape[O_NUM_HEADS],
          reshapeShape[O_HEAD_DIM],
          reshapeShape[O_SEQ_LEN],
          sliceStart,
          sliceEnd,
          std::nullopt,
      });
    }

    if (!sliceDim.has_value()) {
      return false;
    }

    outSliceDim = *sliceDim;
    return true;
  }

  bool validateAndSortSlices(MatMulOpType matmulOp,
                             llvm::SmallVector<SliceReshapeChain> &chains,
                             size_t sliceDim) const {
    // Sort chains by slice start position
    llvm::sort(chains,
               [](const SliceReshapeChain &a, const SliceReshapeChain &b) {
                 return a.sliceStart < b.sliceStart;
               });

    // Validate slices are contiguous and cover the full dimension
    int64_t prevEnd = 0;
    for (const auto &chain : chains) {
      if (chain.sliceStart != prevEnd) {
        return false;
      }
      prevEnd = chain.sliceEnd;
    }

    return prevEnd == matmulOp.getType().getShape()[sliceDim];
  }

  mlir::LogicalResult createFusedOp(mlir::PatternRewriter &rewriter,
                                    MatMulOpType matmulOp,
                                    SliceReshapeChain &qChain,
                                    SliceReshapeChain &kChain,
                                    SliceReshapeChain &vChain) const {

    // Get dimensions from the reshape outputs
    auto qReshapeShape = qChain.reshapeOp.getType().getShape();
    int64_t batchSize = qReshapeShape[O_BATCH];
    int64_t seqLen = qReshapeShape[O_SEQ_LEN];

    // Calculate total hidden dimension from the matmul output
    auto matmulShape = matmulOp.getType().getShape();
    int64_t totalHiddenSize = matmulShape.back();

    // Determine if this is GQA (different Q and KV heads) or MHA
    bool isGQA = (qChain.numHeads != kChain.numHeads);

    rewriter.setInsertionPointAfter(matmulOp);

    // Both MHA and GQA can use a single fused input tensor.
    // The tt-metal op computes: head_size = total / (num_heads +
    // num_kv_heads*2) For MHA: head_size = total / (num_heads * 3) since
    // num_kv_heads==num_heads For GQA: head_size = total / (num_q_heads +
    // num_kv_heads * 2)
    SmallVector<int64_t> inputReshapeShape = {batchSize, seqLen,
                                              totalHiddenSize};
    SmallVector<int32_t> inputReshapeShapeI32(inputReshapeShape.begin(),
                                              inputReshapeShape.end());

    RankedTensorType reshapeInputTy = utils::RankedTensorTypeFactory::create(
        matmulOp.getType(), inputReshapeShape);

    auto inputReshape = rewriter.create<ReshapeOp>(
        matmulOp.getLoc(), reshapeInputTy, matmulOp.getResult(),
        rewriter.getI32ArrayAttr(inputReshapeShapeI32), MemoryConfigAttr());

    // For GQA, pass num_kv_heads; for MHA, omit it (op defaults to num_heads)
    auto splitOp = rewriter.create<SplitQueryKeyValueAndSplitHeadsOp>(
        matmulOp.getLoc(),
        TypeRange{qChain.reshapeOp.getType(), kChain.reshapeOp.getType(),
                  vChain.reshapeOp.getType()},
        inputReshape.getResult(),
        Value(), // no separate KV input - use single fused tensor
        rewriter.getUI32IntegerAttr(qChain.numHeads),
        isGQA ? rewriter.getUI32IntegerAttr(kChain.numHeads) : IntegerAttr(),
        rewriter.getBoolAttr(false) /*transpose_key*/, MemoryConfigAttr());

    rewriter.replaceOp(qChain.reshapeOp, splitOp.getQuery());
    rewriter.replaceOp(kChain.reshapeOp, splitOp.getKey());
    rewriter.replaceOp(vChain.reshapeOp, splitOp.getValue());

    return mlir::success();
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
        TTNNMatmulAndLinearWithActivation<LinearOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, GeluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, GeluOp>>(&getContext());

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      patterns.add<RoPEFusing>(&getContext());
      patterns.add<SDPAFusing>(&getContext());
      patterns.add<SplitQKVFromSlicesPattern<MatmulOp>>(&getContext());
      patterns.add<SplitQKVFromSlicesPattern<LinearOp>>(&getContext());
    }
#endif // TTMLIR_ENABLE_OPMODEL

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
