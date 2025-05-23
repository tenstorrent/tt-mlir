// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <iostream>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFUSING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
// Check if we can fuse conv2d followed by add into conv2d with bias.
class TTIRConv2dWithBias : public mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const final {
    auto components = getConv2dAndBias(srcOp);
    if (!components) {
      return mlir::failure();
    }

    auto conv2dOp = components->first;
    auto bias = components->second;
    rewriter.modifyOpInPlace(conv2dOp,
                             [&]() { conv2dOp.getBiasMutable().assign(bias); });
    rewriter.replaceAllOpUsesWith(srcOp, conv2dOp);

    // The original conv2d op will be removed by DCE since it's no longer
    // used.
    return mlir::success();
  }

private:
  bool isFusable(ttir::Conv2dOp conv2dOp,
                 mlir::TypedValue<mlir::RankedTensorType> bias) const {
    return conv2dOp && !conv2dOp.getBias() && conv2dOp->hasOneUse() &&
           conv2dOp.isBiasCompatible(bias.getType().getShape()) &&
           (!bias.getDefiningOp() ||
            bias.getDefiningOp()->isBeforeInBlock(conv2dOp));
  }

  std::optional<std::pair<ttir::Conv2dOp, mlir::Value>>
  getConv2dAndBias(AddOp srcOp) const {
    auto lhs = srcOp.getLhs();
    auto rhs = srcOp.getRhs();
    auto lhsConv2dOp = lhs.getDefiningOp<ttir::Conv2dOp>();
    auto rhsConv2dOp = rhs.getDefiningOp<ttir::Conv2dOp>();
    if (isFusable(lhsConv2dOp, rhs)) {
      return std::make_pair(lhsConv2dOp, rhs);
    }
    if (isFusable(rhsConv2dOp, lhs)) {
      return std::make_pair(rhsConv2dOp, lhs);
    }
    return std::nullopt;
  }
};

// This pattern detects when a reduction operation is followed by a reshape
// operation that simply adds back dimensions that were reduced. In such cases,
// we can fuse the reshape into the reduction operation by setting
// keep_dim=true.
template <typename ReductionOpTy>
class ReductionWithReshapePattern
    : public mlir::OpRewritePattern<ReductionOpTy> {
  using mlir::OpRewritePattern<ReductionOpTy>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReductionOpTy reductionOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if the reduction op has exactly one use and it's a reshape op.
    if (!isFusable(reductionOp)) {
      return mlir::failure();
    }

    // Get the reshape op that follows the reduction op.
    auto reshapeOp =
        mlir::cast<ReshapeOp>(*reductionOp.getResult().getUsers().begin());

    // Replce old reduction with new reduction with keep_dim=true.
    utils::replaceOpWithNewDPSOp<ReductionOpTy>(
        rewriter, reductionOp, reshapeOp.getResult().getType(),
        reductionOp.getInput(), /*keep_dim=*/rewriter.getBoolAttr(true),
        reductionOp.getDimArgAttr());

    // Reshape will be folded into the new reduction op.
    return mlir::success();
  }

private:
  bool isFusable(ReductionOpTy reductionOp) const {
    // Reduction should only have one use and that use should be a reshape op.
    if (!reductionOp.getResult().hasOneUse() ||
        !ttmlir::utils::allUsersOfType<ReshapeOp>(reductionOp)) {
      return false;
    }

    // If keep dim is already set, we cannot fuse.
    if (reductionOp.getKeepDim()) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*reductionOp.getResult().getUsers().begin());

    // Get the input and output shapes.
    ArrayRef<int64_t> inputShape = reductionOp.getInput().getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();

    // Reshape output shape should have the same rank as the input shape.
    if (reshapeOutputShape.size() != inputShape.size()) {
      return false;
    }

    // Calculate the expected shape after reduction with keep_dim=true.
    llvm::SmallVector<int64_t> expectedShape(inputShape);

    if (!reductionOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced to 1.
      expectedShape = llvm::SmallVector<int64_t>(inputShape.size(), 1);
    } else {
      // Only specified dimensions are reduced to 1.
      mlir::ArrayAttr reduceDims = *reductionOp.getDimArg();
      for (mlir::Attribute reduceDim : reduceDims) {
        int64_t reduceDimInt =
            mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
        // Handle negative indices.
        reduceDimInt = (reduceDimInt + inputShape.size()) % inputShape.size();
        expectedShape[reduceDimInt] = 1;
      }
    }

    // Check if the reshape output matches the expected shape.
    return expectedShape == reshapeOutputShape;
  }
};

// This pattern detects and fuses a sequence of operations that implement
// softmax:
// 1. exp(x)
// 2. sum(exp(x)) along a dimension with keep_dim=true
// 3. broadcast the sum
// 4. divide exp(x) by the broadcasted sum
class SoftmaxFusionPattern : public mlir::OpRewritePattern<DivOp> {
  using mlir::OpRewritePattern<DivOp>::OpRewritePattern;

public:
  // Pattern: div(exp(x), broadcast(sum(exp(x), keep_dim=true))).
  mlir::LogicalResult
  matchAndRewrite(DivOp divOp, mlir::PatternRewriter &rewriter) const final {

    // Get the numerator (exp) and denominator (broadcast).
    mlir::Value numerator = divOp.getLhs();
    mlir::Value denominator = divOp.getRhs();

    // Check that the numerator is an exp operation.
    auto expOp = numerator.getDefiningOp<ExpOp>();
    if (!expOp) {
      return mlir::failure();
    }

    // Check that the denominator is a broadcast operation.
    auto broadcastOp = denominator.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return mlir::failure();
    }

    // Check that the broadcast input is a sum operation with keep_dim=true.
    auto sumOp = broadcastOp.getInput().getDefiningOp<SumOp>();
    if (!sumOp || !sumOp.getKeepDim()) {
      return mlir::failure();
    }

    // Check that the sum input is the same exp operation as the numerator.
    if (sumOp.getInput() != expOp.getResult()) {
      return mlir::failure();
    }

    // Check correct user counts for each operation in the pattern:
    // - exp should have exactly 2 users (sum and div)
    // - sum should have exactly 1 user (broadcast)
    // - broadcast should have exactly 1 user (div)
    // TODO(milant): Relax requirements for fusing #3183.
    if (ttmlir::utils::countUsers(expOp.getResult()) != 2 ||
        !sumOp.getResult().hasOneUse() ||
        !broadcastOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Get the reduction dimension from the sum operation.
    if (!sumOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced, which is
      // not what we want for softmax.
      return mlir::failure();
    }

    mlir::ArrayAttr reduceDims = *sumOp.getDimArg();
    if (reduceDims.size() != 1) {
      // Softmax reduces along a single dimension.
      return mlir::failure();
    }

    int64_t reduceDim = mlir::cast<mlir::IntegerAttr>(reduceDims[0]).getInt();

    // Replace div op with new softmax op.
    utils::replaceOpWithNewDPSOp<SoftmaxOp>(rewriter, divOp,
                                            divOp.getResult().getType(),
                                            expOp.getInput(), reduceDim);

    return mlir::success();
  }
};

class Conv2dWithMultiply : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  /// Pattern: conv2d(input, weight) * scale
  ///
  /// This pattern detects when a Conv2d operation (with constant weights) is
  /// followed by a multiplication with a constant scale factor. It optimizes
  /// this pattern by pre-multiplying the convolution weights with the scale
  /// factor, which eliminates the runtime multiplication operation.
  ///
  /// Input pattern:
  ///   %conv = conv2d(%input, %weight)  // weight is constant
  ///   %result = multiply(%conv, %scale)  // scale is constant
  ///   (1,1,1,out_channels)
  ///
  /// Output pattern:
  ///   %reshaped_scale = reshape(%scale) to (out_channels,1,1,1)
  ///   %scaled_weight = multiply(%weight, %reshaped_scale)
  ///   %result = conv2d(%input, %scaled_weight)
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if this pattern is applicable.
    auto components = getConv2dAndScale(multiplyOp);
    if (!components) {
      return mlir::failure();
    }

    Conv2dOp conv2dOp = components->first;
    Value scaleValue = components->second;

    // Insert before conv2d op.
    rewriter.setInsertionPoint(conv2dOp);

    // Reshape scale to match weight dimensions and pre-multiply weights.
    Value reshapedScale =
        createReshapedScale(rewriter, conv2dOp.getLoc(), scaleValue);
    Value scaledWeights = createScaledWeights(
        rewriter, conv2dOp.getLoc(), conv2dOp.getWeight(), reshapedScale);

    // Update conv2d to use scaled weights and replace multiply operation
    rewriter.modifyOpInPlace(
        conv2dOp, [&]() { conv2dOp.getWeightMutable().assign(scaledWeights); });
    rewriter.replaceAllOpUsesWith(multiplyOp, conv2dOp);

    return mlir::success();
  }

private:
  static std::optional<std::pair<Conv2dOp, mlir::Value>>
  getConv2dAndScale(MultiplyOp multiplyOp) {
    auto lhs = multiplyOp.getLhs();
    auto rhs = multiplyOp.getRhs();
    Conv2dOp lhsConv2d = lhs.getDefiningOp<Conv2dOp>();
    Conv2dOp rhsConv2d = rhs.getDefiningOp<Conv2dOp>();
    if (isCommutable(lhsConv2d, rhs)) {
      return std::make_pair(lhsConv2d, rhs);
    }
    if (isCommutable(rhsConv2d, lhs)) {
      return std::make_pair(rhsConv2d, lhs);
    }
    return std::nullopt;
  }

  // We can commute only if both scale and weight are constant.
  static bool isCommutable(Conv2dOp conv2dOp,
                           mlir::TypedValue<RankedTensorType> scale) {
    // Conv2d should only have one use and that use should be a multiply op.
    if (!conv2dOp || !conv2dOp.getResult().hasOneUse()) {
      return false;
    }

    mlir::func::FuncOp funcOp = conv2dOp->getParentOfType<mlir::func::FuncOp>();
    llvm::SmallPtrSet<BlockArgument, 4> constParams =
        ttmlir::utils::populateConstParams(funcOp);
    auto isConstant = [&constParams, conv2dOp](mlir::Value value) {
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
        return constParams.contains(blockArg);
      }

      Operation *op = value.getDefiningOp();
      return op->hasTrait<mlir::tt::Trait::TTCreationOpTrait>() &&
             op->isBeforeInBlock(conv2dOp);
    };

    // Both scale and weight must be constant.
    if (!isConstant(scale) || !isConstant(conv2dOp.getWeight())) {
      return false;
    }

    return hasValidScaleShape(conv2dOp, scale.getType());
  }

  // Scale must have rank 4 and shape (1, 1, 1, out_channels).
  static bool hasValidScaleShape(Conv2dOp convOp, RankedTensorType scaleType) {
    return scaleType.getRank() == 4 && scaleType.getDimSize(0) == 1 &&
           scaleType.getDimSize(1) == 1 && scaleType.getDimSize(2) == 1 &&
           scaleType.getDimSize(3) == convOp.getOutputChannelSize();
  }

  static Value createReshapedScale(mlir::PatternRewriter &rewriter,
                                   Location loc, Value scaleValue) {
    // Get the scale's type.
    RankedTensorType scaleType =
        mlir::cast<RankedTensorType>(scaleValue.getType());

    // Create a new shape (out_channels, 1, 1, 1) from (1, 1, 1, out_channels).
    llvm::SmallVector<int64_t> newShape(scaleType.getShape());
    // Swap first and last dimensions.
    assert(newShape.size() == 4 &&
           "Scale tensor must have 4 dimensions for reshaping.");
    std::swap(newShape[0], newShape[3]);
    // Convert to int32 for the reshape operation.
    llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());

    // Create and return the reshape operation.
    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshape"),
        newShape, scaleType.getElementType(), scaleType.getEncoding(),
        scaleValue, rewriter.getI32ArrayAttr(newShapeI32));
  }

  /// Create pre-multiplied weights.
  static Value createScaledWeights(mlir::PatternRewriter &rewriter,
                                   Location loc, Value weightValue,
                                   Value reshapedScale) {
    // Create a multiplication of the weights by the reshaped scale.
    return utils::createDPSOp<MultiplyOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_multiply"),
        mlir::cast<RankedTensorType>(weightValue.getType()), weightValue,
        reshapedScale);
  }
};

// Tag all block arguments which are direct inputs to Conv2dOp with
// discardable attribute. This is used during Layouting to check if
// function argument need to be put to Host/RM. This is temporary
// solution until we complete refactor of TTNNLayout see:
// https://github.com/tenstorrent/tt-mlir/issues/3432.
class Conv2dTagWeights : public mlir::OpRewritePattern<Conv2dOp> {
public:
  Conv2dTagWeights(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  mlir::LogicalResult
  matchAndRewrite(Conv2dOp conv2d,
                  mlir::PatternRewriter &rewriter) const final {
    if (BlockArgument blockArg = dyn_cast<BlockArgument>(conv2d.getWeight())) {
      // Get the function that owns this block argument.
      func::FuncOp owningFunc =
          cast<func::FuncOp>(blockArg.getOwner()->getParentOp());

      // Get the argument index.
      uint32_t argIdx = blockArg.getArgNumber();

      // Check if the argument already has the g_conv2dWeight attribute.
      if (owningFunc.getArgAttr(argIdx,
                                ttmlir::utils::g_conv2dWeightAttrName)) {
        return failure();
      }

      // Create the g_conv2dWeight attribute.
      auto gConv2dWeightAttr = rewriter.getUnitAttr();
      owningFunc.setArgAttr(argIdx, ttmlir::utils::g_conv2dWeightAttrName,
                            gConv2dWeightAttr);

      return success();
    }

    return failure();
  }
};

class KvCacheUpdatePattern : public mlir::OpRewritePattern<ScatterOp> {
  using mlir::OpRewritePattern<ScatterOp>::OpRewritePattern;

public:
  /// Pattern: scatter(input, indices, updates)
  ///
  /// This pattern detects when a Scatter operation is used to update a
  /// key-value cache (kv-cache) in a transformer model. It checks the shape of
  /// the input, indices, and updates tensors to ensure they match the expected
  /// kv-cache update pattern.
  ///
  /// Input pattern:
  ///   %input = scatter(%cache, %indices, %updates)
  ///   - Given a cache with shape (batch, num_heads, max_len, hidden_dim) and a
  ///     an updates tensor with shape (batch, num_heads, sequence_length,
  ///     hidden_dim), the indices tensor should have shape (batch, num_heads,
  ///     sequence, hidden_dim, 4), representingn all the possible indices in
  ///     the kv-cache to be updated.
  /// Output pattern:
  ///   %input = cacheUpdate(%cache, %start_idx, %updates)

  mlir::LogicalResult
  matchAndRewrite(ScatterOp scatterOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (!isKvCacheUpdate(scatterOp)) {
      return mlir::failure();
    }

    std::cout<< "Replacing scatter op with kv-cache update op" << std::endl;

    // Extract the operands of the scatter operation.
    Value cache = scatterOp.getInput();             // Cache tensor
    Value updates = scatterOp.getUpdate();          // Updates tensor

    // Create a constant batch offset (set to 0 for now).
    auto batchOffsetAttr = rewriter.getI32IntegerAttr(0);

    // Replace the scatter operation with the fill_cache operation.
    rewriter.replaceOpWithNewOp<FillCacheOp>(
        scatterOp, scatterOp.getResult().getType(),  // Result type
        cache,                                       // Cache tensor
        updates,                                     // Updates tensor
        batchOffsetAttr                              // Batch offset
    );

    return mlir::success();
  }

private:
  // Check if the scatter op is a kv-cache update.
  //
  // We are looking for a pattern like this (from the bottom up):
  // %187 = "ttir.concat"(%167, %173, %179, %185):
  //        (d0,d1,d2,d3,1), (d0,d1,d2,d3,1), (d0,d1,d2,d3,1), (d0,d1,d2,d3,1))
  //        -> (d0,d1,d2,d3,4)
  // %189 = "ttir.scatter"(%arg6, %187, %95, %188)

  static bool isKvCacheUpdate(ttir::ScatterOp scatterOp) {
    auto scatterIndices = scatterOp.getScatterIndices();

    return true;

    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(scatterOp.getInput().getType()).getShape();
    ArrayRef<int64_t> scatterIdxShape =
        mlir::cast<RankedTensorType>(scatterIndices.getType()).getShape();
    ArrayRef<int64_t> udpateShape =
        mlir::cast<RankedTensorType>(scatterOp.getUpdate().getType())
            .getShape();

    std::cout << "scatter" << std::endl;
    if (inputShape.size() != 4 || scatterIdxShape.size() != 5 ||
        udpateShape.size() != 4) {
      return false;
    }

    for (size_t i = 0; i < udpateShape.size(); ++i) {
      if (scatterIdxShape[i] != udpateShape[i]) {
        return false;
      }
    }

    if (scatterIdxShape[4] != 4) {
      return false;
    }

    if (!(inputShape[0] == udpateShape[0] && inputShape[1] == udpateShape[1] &&
          inputShape[3] == udpateShape[3])) {
      return false;
    }

    // We could stop checking here and return true, and use the rest of the
    // checks/logic to get the cache_position tensor instead,
    // so we can slice it and get the Index for the CacheUpdate op.
    // We would need to change all the functions (isKvScatteredIndeces,
    // isBroadcastedDimIndeces, and get rid of isConstArangedTensor) to track
    // the cache_position tensor instead of checking for validity. But most of
    // the logic would still be the same.

    ConcatOp concatOp = scatterIndices.getDefiningOp<ttir::ConcatOp>();
    if (!concatOp) {
      return false;
    }

    if (!isKvScatteredIndeces(concatOp)) {
      return false;
    }

    return true;
  }

  static bool isKvScatteredIndeces(ConcatOp concatOp) {
    mlir::OperandRange inputs = concatOp.getInputs();
    int32_t dim = concatOp.getDim();
    if (inputs.size() != 4 || dim != 4) {
      return false;
    }

    auto firstInput = inputs[0];
    auto secondInput = inputs[1];
    auto thirdInput = inputs[2];
    auto fourthInput = inputs[3];

    if (isBroadcastedDimIndeces(firstInput, 0, true) ||
        isBroadcastedDimIndeces(secondInput, 1, true) ||
        isBroadcastedDimIndeces(thirdInput, 2,
                                false) || // the cache_position tensor
        isBroadcastedDimIndeces(fourthInput, 3, true)) {
      return false;
    }

    return true;
  }

  // For each input of the concat op we check for a pattern like to this (from
  // bottom up): %137 = "ttir.reshape"(%135, %136) -> (kv-heads, 1) %139 =
  // "ttir.reshape"(%137, %138) -> (kv-heads, 1, 1) %169 = "ttir.reshape"(%139,
  // %168) -> (1, kv-heads, 1, 1) %171 = "ttir.broadcast"(%169, %170) -> (batch,
  // N, seq_len, hidden_dim) %173 = "ttir.reshape"(%171, %172) -> (batch, N,
  // seq_len, hidden_dim, 1)
  static bool isBroadcastedDimIndeces(mlir::Value value, int64_t dimIdx,
                                      bool isConst) {
    // Get the tensor shape of value
    auto tensorType = mlir::cast<RankedTensorType>(value.getType());
    auto tensorShape = tensorType.getShape();
    int64_t dim = tensorShape[dimIdx];

    // Check that input is a reshape op.
    auto reshapeOp = value.getDefiningOp<ttir::ReshapeOp>();
    if (!reshapeOp) {
      return false;
    }
    // check that the reshape input shape is the same as the first output shape
    // dims.
    auto reshapeInput = reshapeOp.getInput();

    auto inputShape =
        mlir::cast<RankedTensorType>(reshapeInput.getType()).getShape();
    auto outputShape =
        mlir::cast<RankedTensorType>(reshapeOp.getOutput().getType())
            .getShape();
    if (inputShape.size() != outputShape.size() - 1) {
      return false;
    }
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != outputShape[i]) {
        return false;
      }
    }

    // Check that input to reshape is a broadcast op.
    auto broadcastOp = reshapeInput.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return false;
    }
    // Check that broadcast input and output rank is the same.
    inputShape = mlir::cast<RankedTensorType>(broadcastOp.getInput().getType())
                     .getShape();
    outputShape =
        mlir::cast<RankedTensorType>(broadcastOp.getOutput().getType())
            .getShape();
    // Check that all dims in broadcast input are 1 except one dim and that dim
    // is the same as the output dim.
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != 1 && inputShape[i] != outputShape[i]) {
        return false;
      }
    }

    // Check that the input to broadcast is a reshape and loop through it until
    // we reach the input to the first reshape.
    auto nextInput = broadcastOp.getInput();
    // Loop through the reshapes until we reach the input to the first reshape.
    while (auto intermediateReshapeOp =
               nextInput.getDefiningOp<ttir::ReshapeOp>()) {
      nextInput = intermediateReshapeOp.getInput();
    }
    // Check that the first input of the reshape chain is a single dim flat
    // tensor.
    auto arangedTensor = nextInput;
    auto arangedTensorShape =
        mlir::cast<RankedTensorType>(arangedTensor.getType()).getShape();
    if (arangedTensorShape.size() != 1) {
      return false;
    }

    if (isConst && !isConstArangedTensor(arangedTensor, dim)) {
      return false;
    } else if (!isConst) {
      if (arangedTensorShape[0] != dim) {
        // This is the cache_position tensor. We want to slice this tensor and
        // keep only the first value as the startIdx for the cache_update.
        return false;
      }
    }

    return true;
  }

  // We are looking for a pattern like to this (from bottom up):
  // %127 = "ttir.arange"() {arange_dimension = 0, end = kv-head, start = 0,
  // step = 1} -> (kv-head, ) %129 = "ttir.broadcast"(%127, %128) -> (kv-head,)
  // %131 = "ttir.multiply"(%129, 1, %130) -> (kv-head,)
  // %133 = "ttir.broadcast"(%131, %132) -> (kv-head,)
  // %135 = "ttir.add"(%133, 0, %134) -> (kv-head,)
  static bool isConstArangedTensor(mlir::Value value, int64_t dim) {
    // Check that the input is an add op.
    auto addOp = value.getDefiningOp<AddOp>();
    if (!addOp) {
      return false;
    }
    // Check that first input and output have the same values.
    auto addInputType = mlir::cast<RankedTensorType>(addOp.getLhs().getType());
    auto addOutputType =
        mlir::cast<RankedTensorType>(addOp.getOutput().getType());
    auto addInputShape = addInputType.getShape();
    auto addOutputShape = addOutputType.getShape();
    if (addInputType.getRank() != addOutputType.getRank()) {
      return false;
    }
    for (size_t i = 0; i < addInputShape.size(); ++i) {
      if (addInputShape[i] != addOutputShape[i]) {
        return false;
      }
    }

    // Check that the add op LHS producer is a broadcast op.
    // addBroadcastOp is the broadcast op, which serves as the input to the add
    // op.
    auto addBroadcastOp = addOp.getLhs().getDefiningOp<BroadcastOp>();
    if (!addBroadcastOp) {
      return false;
    }
    // Check that the broadcast op's input and output shape is the same.
    auto addBroadcastInputType =
        mlir::cast<RankedTensorType>(addBroadcastOp.getInput().getType());
    auto addBroadcastOutputType =
        mlir::cast<RankedTensorType>(addBroadcastOp.getOutput().getType());
    auto addBroadcastInputShape = addBroadcastInputType.getShape();
    auto addBroadcastOutputShape = addBroadcastOutputType.getShape();

    if (addBroadcastInputType.getRank() != addBroadcastOutputType.getRank()) {
      return false;
    }
    for (size_t i = 0; i < addBroadcastInputType.getShape().size(); ++i) {
      if (addBroadcastInputShape[i] != addBroadcastOutputShape[i]) {
        return false;
      }
    }

    // Check that addBroadcastOp's input is a multiply op.
    auto BroadcastInput = addBroadcastOp.getInput();
    auto multiplyOp = BroadcastInput.getDefiningOp<MultiplyOp>();
    if (!multiplyOp) {
      return false;
    }
    // Check that the multiply input is an arange op.
    auto arangeOp = multiplyOp.getLhs().getDefiningOp<ArangeOp>();
    if (!arangeOp) {
      return false;
    }
    int64_t arangeStart = arangeOp.getStart();
    int64_t arangeEnd = arangeOp.getEnd();
    int64_t arangeStep = arangeOp.getStep();
    int64_t arangeDim = arangeOp.getArangeDimension();
    if (arangeStart != 0 || arangeEnd != dim || arangeStep != 1 ||
        arangeDim != 0) {
      return false;
    }

    return true;
  }
};

class TTIRFusingPass : public impl::TTIRFusingBase<TTIRFusingPass> {
  public:
    using impl::TTIRFusingBase<TTIRFusingPass>::TTIRFusingBase;
    void runOnOperation() final {
      {
        RewritePatternSet patterns(&getContext());
        patterns.add<Conv2dTagWeights>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
          signalPassFailure();
          return;
        }
      }
      {
        RewritePatternSet patterns(&getContext());
        patterns.add<TTIRConv2dWithBias>(&getContext());
  
        // Add patterns for each reduction op type.
        patterns.add<ReductionWithReshapePattern<SumOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<MeanOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<MaxOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<MinOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<ProdOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<ReduceAndOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<ReduceOrOp>>(&getContext());
        patterns.add<ReductionWithReshapePattern<ArgMaxOp>>(&getContext());
  
        patterns.add<SoftmaxFusionPattern>(&getContext());
        patterns.add<Conv2dWithMultiply>(&getContext());

        patterns.add<KvCacheUpdatePattern>(&getContext());
  
        GreedyRewriteConfig config;
        config.setUseTopDownTraversal(true);
        (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
      }
    }
  };
  } // namespace
  } // namespace mlir::tt::ttir