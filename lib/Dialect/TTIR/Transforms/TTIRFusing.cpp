// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

// This pattern detects and fuses a sequence of operations that implement
// relu. If a zeros op is consumed by a maximum op, we can fuse the zeros and
// maximum op into a relu op.

// Input pattern:
// %0 = "ttir.zeros"() or %0 = "ttir.full"(fill_value = 0)
// %1 = ttir.empty()
// %2 = "ttir.maximum"(%arg0, %0, %1)
//
// Output pattern:
// %0 = ttir.empty()
// %1 = "ttir.relu"(%arg0, %0)
class ReluFusionPattern : public mlir::OpRewritePattern<MaximumOp> {
  using mlir::OpRewritePattern<MaximumOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MaximumOp maximumOp,
                  mlir::PatternRewriter &rewriter) const final {

    Value input;

    if (isCreatingZeros(maximumOp.getLhs())) {
      input = maximumOp.getRhs();
    } else if (isCreatingZeros(maximumOp.getRhs())) {
      input = maximumOp.getLhs();
    } else {
      return mlir::failure();
    }

    rewriter.setInsertionPoint(maximumOp);
    utils::replaceOpWithNewDPSOp<ReluOp>(
        rewriter, maximumOp, maximumOp.getResult().getType(), input);

    return mlir::success();
  }

private:
  // Check if the fill value of the full op is zero.
  bool isFillValueZero(FullOp op) const {
    mlir::Attribute fillValue = op.getFillValue();
    if (auto integerAttr = dyn_cast<IntegerAttr>(fillValue)) {
      return integerAttr.getValue().isZero();
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(fillValue)) {
      return floatAttr.getValue().isZero();
    }
    llvm_unreachable("Fill value should be integer or float");
  }

  bool isCreatingZeros(mlir::Value value) const {
    if (auto creationOp = value.getDefiningOp<ZerosOp>()) {
      return true;
    }
    if (auto creationOp = value.getDefiningOp<FullOp>()) {
      return isFillValueZero(creationOp);
    }

    return false;
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

    // Reshape scale to match weight dimensions and pre-multiply weights.
    Value reshapedScale =
        createReshapedScale(rewriter, conv2dOp.getLoc(), scaleValue,
                            conv2dOp.getWeight().getType());

    // Get UD chain starting from the reshaped scale. This chain will be
    // moved before the conv2dOp to ensure that weight scale can be
    // const-evaled.
    SetVector<Value> udChain = ttmlir::utils::getUseDefChain(reshapedScale);
    SetVector<Operation *> udChainOps =
        ttmlir::utils::filterOperations(udChain.getArrayRef());
    SetVector<Operation *> udChainSorted = topologicalSort(udChainOps);
    for (auto *op : udChainSorted) {
      op->moveBefore(conv2dOp);
    }

    rewriter.setInsertionPoint(conv2dOp);

    // Create scaled weights by multiplying the original weights with the
    // resshaped scale.
    Value scaledWeights = createScaledWeights(
        rewriter, conv2dOp.getLoc(), conv2dOp.getWeight(), reshapedScale);

    // Update conv2d to use scaled weights and replace multiply operation.
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
        mlir::tt::ttcore::getConstsAndParams(funcOp);
    auto isConstant = [&constParams](mlir::Value value) {
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
        return constParams.contains(blockArg);
      }

      Operation *defOp = value.getDefiningOp();
      return defOp->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>();
    };

    // If weight is not constant, we cannot commute.
    if (!isConstant(conv2dOp.getWeight())) {
      return false;
    }

    RankedTensorType scaleType = scale.getType();
    // If scale is comming from broadcast then we want to use the input type
    // to the broadcast to check the shape.
    if (auto bcastOp =
            mlir::dyn_cast_if_present<BroadcastOp>(scale.getDefiningOp())) {
      scaleType = bcastOp.getInput().getType();
    }

    // Check if scale shape is with conv2d weight.
    if (!hasValidScaleShape(conv2dOp, scaleType)) {
      return false;
    }

    // Now we want to check if operations which produce scale are
    // const-evalable. We do this by getting UD chain of the scale and then
    // checking if all inputs into this chain are constants.
    SetVector<Value> useDefChain = ttmlir::utils::getUseDefChain(scale);
    SetVector<BlockArgument> useDefChainBlockArgs =
        ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
    if (!all_of(useDefChainBlockArgs, isConstant)) {
      return false;
    }

    // Since we want to move the scale chain before conv2dOp we want to make
    // sure that the scale chain does not contain conv2dOp.
    if (useDefChain.contains(conv2dOp)) {
      return false;
    }

    return true;
  }

  // Scale must have rank 4 and shape (1, 1, 1, out_channels).
  static bool hasValidScaleShape(Conv2dOp convOp, RankedTensorType scaleType) {
    return scaleType.getRank() == 4 && scaleType.getDimSize(0) == 1 &&
           scaleType.getDimSize(1) == 1 && scaleType.getDimSize(2) == 1 &&
           scaleType.getDimSize(3) == convOp.getOutputChannelSize();
  }

  // There are two cases we want to handle here:
  // 1. Input scale is a constant tensor that only neeeds reshaping
  // 2. Input scale is a broadcast operation that needs reshaping
  //
  // In case of 1 we just add reshape operation to the scale tensor such that
  // it has shape (out_channels, 1, 1, 1).
  //
  // In case of 2 we need to add reshape operation to the input of the of bcast
  // and then we create new broadcast operation with the new reshaped scale
  // which broadcasts the reshaped scale to the shape of the weight tensor.
  static Value createReshapedScale(mlir::PatternRewriter &rewriter,
                                   Location loc, Value scaleValue,
                                   RankedTensorType weightType) {
    // If scaleValue is broadcast operation we want to reshape its input.
    // Otherwise we reshape the scaleValue itself.
    Value reshapeInput = scaleValue;
    if (auto bcastOp = mlir::dyn_cast_if_present<BroadcastOp>(
            scaleValue.getDefiningOp())) {
      rewriter.setInsertionPoint(bcastOp);
      reshapeInput = bcastOp.getInput();
    }

    // Get the scale's type.
    RankedTensorType scaleType =
        mlir::cast<RankedTensorType>(reshapeInput.getType());

    // Create a new shape (out_channels, 1, 1, 1) from (1, 1, 1, out_channels).
    llvm::SmallVector<int64_t> newShape(scaleType.getShape());
    // Swap first and last dimensions.
    assert(newShape.size() == 4 &&
           "Scale tensor must have 4 dimensions for reshaping.");
    std::swap(newShape[0], newShape[3]);
    // Convert to int32 for the reshape operation.
    llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());

    // Create the reshape operation.
    auto reshapedScale = ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshape"),
        newShape, scaleType.getElementType(), scaleType.getEncoding(),
        reshapeInput, rewriter.getI32ArrayAttr(newShapeI32));

    // If scale value is not a broadcast operation we can return reshapedScale.
    if (!isa_and_present<ttir::BroadcastOp>(scaleValue.getDefiningOp())) {
      return reshapedScale;
    }

    // Otherwise we need to create a new broadcast operation that will take
    // reshaped scale and brroadcast it to the shape of the weight tensor.
    SmallVector<int64_t> broadcastDims =
        ttmlir::utils::getBroadcastDimensions<int64_t>(
            reshapedScale.getType().getShape(), weightType.getShape());
    return ttir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, scaleValue.getLoc(), weightType, reshapedScale,
        broadcastDims);
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

class BatchNormDecomposition : public mlir::OpRewritePattern<BatchNormOp> {
  using mlir::OpRewritePattern<BatchNormOp>::OpRewritePattern;

public:
  // This pattern decomposes the BatchNorm operation into a sequence of
  // arithmetic operations that can be fused with Conv2d operations.
  //
  // Decomposition:
  //    batch_norm(x, scale, offset, mean, variance, epsilon, dimension)
  //    = (x - mean) / sqrt(variance + epsilon) * scale + offset
  //      let alpha = scale / sqrt(variance + epsilon)        -> constant
  //      let beta  = offset - alpha * mean                   -> constant
  //    batch_norm(x,...) = alpha * x + beta
  //
  // Decomposed like this it can later be fused with Conv2d without bias as
  //    batch_norm(conv2d(x, weight),...) = conv2d(x, weight * alpha, beta)
  mlir::LogicalResult
  matchAndRewrite(BatchNormOp batchNormOp,
                  mlir::PatternRewriter &rewriter) const final {

    // Used only paired with convolution
    if (auto conv2dOp = batchNormOp.getOperand().getDefiningOp<Conv2dOp>();
        !conv2dOp || !conv2dOp->hasOneUse()) {
      return mlir::failure();
    }

    // get all attributes
    auto scale = batchNormOp.getScale();
    auto offset = batchNormOp.getOffset();
    auto mean = batchNormOp.getMean();
    auto variance = batchNormOp.getVariance();
    auto input = batchNormOp.getOperand();
    auto epsilon = batchNormOp.getEpsilon();
    auto dimension = batchNormOp.getDimension();

    Location loc = batchNormOp.getLoc();
    RankedTensorType resultType = batchNormOp.getResult().getType();

    RankedTensorType scalarType =
        RankedTensorType::get({1}, variance.getType().getElementType(),
                              variance.getType().getEncoding());

    // Convert epsilon to a tensor
    auto epsilonTensor = rewriter.create<FullOp>(
        loc, scalarType, rewriter.getF32FloatAttr(epsilon.convertToFloat()));

    // variance + epsilon
    auto variancePlusEpsilon = utils::createDPSOp<AddOp>(
        rewriter, loc, variance.getType(), variance, epsilonTensor);

    // std = sqrt(variance + epsilon)
    auto std = utils::createDPSOp<SqrtOp>(rewriter, loc, variance.getType(),
                                          variancePlusEpsilon);
    // alpha = scale / std
    auto alpha =
        utils::createDPSOp<DivOp>(rewriter, loc, scale.getType(), scale, std);

    // alphaMean = alpha * mean
    auto alphaMean = utils::createDPSOp<MultiplyOp>(
        rewriter, loc, mean.getType(), mean, alpha);

    // beta = offset - alphaMean
    auto beta = utils::createDPSOp<SubtractOp>(rewriter, loc, offset.getType(),
                                               offset, alphaMean);

    // Reshape alpha and beta along the specified dimension to match the input
    // shape: for dimension = 3 and input shape (N, H, W, C), reshape from (C)
    // to (1, 1, 1, C).

    SmallVector<int64_t> reshapeShape(4, 1);
    reshapeShape[dimension] = std.getType().getShape()[0];
    SmallVector<int32_t> reshapeShapeI32(reshapeShape.begin(),
                                         reshapeShape.end());
    auto alphaReshaped = utils::createDPSOp<ReshapeOp>(
        rewriter, loc, reshapeShape, alpha.getType().getElementType(),
        alpha.getType().getEncoding(), alpha,
        rewriter.getI32ArrayAttr(reshapeShapeI32));

    auto betaReshaped = utils::createDPSOp<ReshapeOp>(
        rewriter, loc, reshapeShape, beta.getType().getElementType(),
        beta.getType().getEncoding(), beta,
        rewriter.getI32ArrayAttr(reshapeShapeI32));

    // alpha * x
    auto scaled = utils::createDPSOp<MultiplyOp>(rewriter, loc, input.getType(),
                                                 input, alphaReshaped);
    // alpha * x + beta
    auto result = utils::createDPSOp<AddOp>(rewriter, loc, resultType, scaled,
                                            betaReshaped);

    rewriter.replaceOp(batchNormOp, result);

    return mlir::success();
  }
};

class CacheFillUpdatePattern : public mlir::OpRewritePattern<ScatterOp> {
  using mlir::OpRewritePattern<ScatterOp>::OpRewritePattern;

public:
  /// Pattern: scatter(input, indices, updates)
  ///
  /// This pattern detects when a ScatterOp is used as a fill/update for a
  /// cache. We check for its input, indices, and update tensors to ensure they
  /// match the expected cache fill/update pattern.
  ///
  /// Input pattern:
  ///   %result = scatter(%cache, %indices, %updates)
  ///   - Given a cache with shape (B, N, M, H) and a updates tensor with shape
  ///   (B, N, S, H), the indices tensor should have shape (B, N, S, H, 4),
  ///   representing the index where each element in %updates should placed in
  ///   the %cache.
  ///   - %indices can be tracked back to the function's cachePositions input
  ///   that represents the indices of the cache to fill/update.
  /// Output pattern:
  ///   %result = fillCacheOp(%cache, %updates)
  ///   or (if S == 1)
  ///   %result = updateCacheOp(%cache, %updates, %update_index)
  mlir::LogicalResult
  matchAndRewrite(ScatterOp scatterOp,
                  mlir::PatternRewriter &rewriter) const final {
    auto CachePositions = getCacheUpdatePositions(scatterOp);
    if (!CachePositions) {
      return mlir::failure();
    }

    auto cacheUpdateInputType =
        mlir::cast<RankedTensorType>((*CachePositions).getType());
    auto cacheUpdateInputShape = cacheUpdateInputType.getShape();
    if (cacheUpdateInputShape.size() != 1) {
      return mlir::failure();
    }

    Value cache = scatterOp.getInput();
    Value updates = scatterOp.getUpdate();
    auto batchOffsetAttr = rewriter.getI32IntegerAttr(0);

    // If the cachePositions tensor has more than one element we assume it
    // represents a set of aranged indices (0, cachePositions.size), so we
    // replace it with FillCacheOp. If the tensor has only one element, we
    // assume it represents the update index for UpateCacheOp.
    if (cacheUpdateInputShape[0] != 1) {
      rewriter.replaceOpWithNewOp<FillCacheOp>(
          scatterOp, scatterOp.getResult().getType(), // Result type
          cache,                                      // Cache tensor
          updates,                                    // Updates tensor
          batchOffsetAttr                             // Batch offset
      );
    } else {
      rewriter.replaceOpWithNewOp<UpdateCacheOp>(
          scatterOp, scatterOp.getResult().getType(), // Result type
          cache,                                      // Cache tensor
          updates,                                    // Updates tensor
          *CachePositions,                            // Cache Idx
          batchOffsetAttr                             // Batch offset
      );
    }

    return mlir::success();
  }

private:
  // Check if the scatter op is a cache fill/update, and track the
  // cachePositions input tensor if it is.
  //
  // We are looking for:
  // %indices = "ttir.concat"(%0, %1, %2, %3):
  //        (B,N,S,H,1), (B,N,S,H,1), (B,N,S,H,1), (B,N,S,H,1)) -> (B,N,S,H,4)
  // %result = "ttir.scatter"(%cache, %indices, %updates)
  // Where:
  //    1. %0, %1, and %3 come from a const aranged vector representing
  //       the first, second, and fourth dimension indices of the cache.
  //    2. %2 comes from the input representing the cachePositions tensor.
  static std::optional<mlir::Value>
  getCacheUpdatePositions(ttir::ScatterOp scatterOp) {
    // Check that the scatter op inputs represent a cache fill/update:
    //    1. The input is a 4D (B, N, M, H)
    //    2. The update tensor is a 4D tensor (B, N, S, H)
    //    3. The scatter indices is a 5D tensor (B, N, S, H, 4)
    auto scatterIndices = scatterOp.getScatterIndices();
    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(scatterOp.getInput().getType()).getShape();
    ArrayRef<int64_t> scatterIdxShape =
        mlir::cast<RankedTensorType>(scatterIndices.getType()).getShape();
    ArrayRef<int64_t> updateShape =
        mlir::cast<RankedTensorType>(scatterOp.getUpdate().getType())
            .getShape();
    if (inputShape.size() != 4 || scatterIdxShape.size() != 5 ||
        updateShape.size() != 4) {
      return std::nullopt;
    }
    for (size_t i = 0; i < updateShape.size(); ++i) {
      if (scatterIdxShape[i] != updateShape[i]) {
        return std::nullopt;
      }
    }
    if (scatterIdxShape[4] != 4) {
      return std::nullopt;
    }
    if (!(inputShape[0] == updateShape[0] && inputShape[1] == updateShape[1] &&
          inputShape[3] == updateShape[3])) {
      return std::nullopt;
    }

    // Check that the scatter indices input is a concat op that produces the
    // scatter indices for a cache update/fill:
    //    1. Check that the 1st, 2nd and 4th inputs come from a 1D const aranged
    //    tensor
    //    2. Check that the 3rd input comes from the cachePositions func input
    ConcatOp concatOp = scatterIndices.getDefiningOp<ttir::ConcatOp>();
    if (!concatOp) {
      return std::nullopt;
    }

    mlir::OperandRange inputs = concatOp.getInputs();
    int32_t dim = concatOp.getDim();
    if (inputs.size() != 4 || dim != 4) {
      return std::nullopt;
    }

    if (!isBroadcastedDimIndices(inputs[0], 0) ||
        !isBroadcastedDimIndices(inputs[1], 1) ||
        !isBroadcastedDimIndices(inputs[3], 3)) {
      return std::nullopt;
    }

    auto cachePositionInput = getCachePositionsInput(inputs[2]);
    return cachePositionInput;
  }

  // For the concat input that can be tracked to the cachePositions input,
  // check the pattern and track value up to the cachePositions tensor input.
  //  %1 = "ttir.reshape"(%arg, %0) -> (S, 1)
  //  %3 = "ttir.reshape"(%1, %2) -> (1, 1, S, 1)
  //  %5 = "ttir.broadcast"(%3, %4) -> (B, N, S, H)
  //  %7 = "ttir.reshape"(%5, %6) -> (B, N, S, H, 1)
  // Where:
  //  1. %arg is (S, ) and is the cachePositions input tensor.
  //  2. B, S and H are const values.
  static std::optional<mlir::Value>
  getCachePositionsInput(mlir::Value inputValue) {
    // 1. Check that value is a reshape op.
    // 2. Check that the reshape's input and output shapes are compatible:
    //    - Output.rank() = Input.rank() + 1
    //    - All the input dims are the same as the output dims except the
    //      last one which is 1.
    auto reshapeOp = inputValue.getDefiningOp<ttir::ReshapeOp>();
    if (!reshapeOp) {
      return std::nullopt;
    }
    auto reshapeInput = reshapeOp.getInput();
    auto inputShape =
        mlir::cast<RankedTensorType>(reshapeInput.getType()).getShape();
    auto outputShape =
        mlir::cast<RankedTensorType>(reshapeOp.getOutput().getType())
            .getShape();
    if (inputShape.size() != outputShape.size() - 1) {
      return std::nullopt;
    }
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != outputShape[i]) {
        return std::nullopt;
      }
    }

    // 1. Check that input to reshape is a broadcast op.
    // 2. Check that the broadcast's input and output ranks are the same.
    // 3. Check that all dims in broadcast input are 1 or inputShape[i] ==
    //    outputShape[i]
    auto broadcastOp = reshapeInput.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return std::nullopt;
    }
    inputShape = mlir::cast<RankedTensorType>(broadcastOp.getInput().getType())
                     .getShape();
    outputShape =
        mlir::cast<RankedTensorType>(broadcastOp.getOutput().getType())
            .getShape();
    if (inputShape.size() != outputShape.size()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != 1 && inputShape[i] != outputShape[i]) {
        return std::nullopt;
      }
    }

    // Check that the input to broadcast are reshapes and loop through them
    // until we reach the input to the first reshape.
    auto nextInput = broadcastOp.getInput();
    while (auto intermediateReshapeOp =
               nextInput.getDefiningOp<ttir::ReshapeOp>()) {
      nextInput = intermediateReshapeOp.getInput();
    }

    // Check that the input of the reshape chain is a single dim flat
    // tensor. We will assume that this is the cachePositions tensor.
    // TODO(jazpur): Check that this is a 1D aranged tensor coming from the
    // input.
    auto inputTensor = nextInput;
    auto inputTensorShape =
        mlir::cast<RankedTensorType>(inputTensor.getType()).getShape();
    if (inputTensorShape.size() != 1) {
      return std::nullopt;
    }

    return inputTensor;
  }

  // For each (const) input of the concat op we check for:
  //  %2 = "ttir.reshape"(%0, %1) -> (N, 1)
  //  %4 = "ttir.reshape"(%2, %3) -> (N, 1, 1)
  //  %6 = "ttir.reshape"(%4, %5) -> (1, N, 1, 1)
  //  %8 = "ttir.broadcast"(%6, %7) -> (B, N, S, H)
  //  %10 = "ttir.reshape"(%8, %9) -> (B, N, S, H, 1)
  // Where:
  //  1. %0 is a const aranged tensor with shape (N,))
  //  2. depending on dimIdx, same pattern applies for the B, and H dims
  static bool isBroadcastedDimIndices(mlir::Value inputValue, int64_t dimIdx) {
    // 1. Check that valueInput is a reshape op.
    // 2. Check that the reshape's input and output shapes are compatible:
    //    - Output.rank() = input.rank() + 1
    //    - All the input dims are the same as the output dims except the
    //      last one which is 1.
    auto reshapeOp = inputValue.getDefiningOp<ttir::ReshapeOp>();
    if (!reshapeOp) {
      return false;
    }
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

    // 1. Check that input to reshape is a broadcast op.
    // 2. Check that broadcast input and output rank is the same.
    // 3. Check that all dims in broadcast input are 1 or inputShape[i] ==
    //    outputShape[i]
    auto broadcastOp = reshapeInput.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return false;
    }
    inputShape = mlir::cast<RankedTensorType>(broadcastOp.getInput().getType())
                     .getShape();
    outputShape =
        mlir::cast<RankedTensorType>(broadcastOp.getOutput().getType())
            .getShape();
    if (inputShape.size() != outputShape.size()) {
      return false;
    }
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != 1 && inputShape[i] != outputShape[i]) {
        return false;
      }
    }

    // Check that the input to broadcast is a reshape and loop through it until
    // we reach the input to the first reshape.
    auto nextInput = broadcastOp.getInput();
    while (auto intermediateReshapeOp =
               nextInput.getDefiningOp<ttir::ReshapeOp>()) {
      nextInput = intermediateReshapeOp.getInput();
    }

    // 1. Check that the first input of the reshape chain is a single dim flat
    //    tensor.
    // 2. Check that the input is a const aranged tensor.
    auto arangedTensor = nextInput;
    auto arangedTensorShape =
        mlir::cast<RankedTensorType>(arangedTensor.getType()).getShape();
    if (arangedTensorShape.size() != 1) {
      return false;
    }

    auto tensorType = mlir::cast<RankedTensorType>(inputValue.getType());
    auto tensorShape = tensorType.getShape();
    int64_t dim = tensorShape[dimIdx];
    if (!isConstArangedTensor(arangedTensor, dim)) {
      return false;
    }

    return true;
  }

  // We are looking for a pattern like to this (from bottom up):
  //  %0 = "ttir.arange"() {arr_dim = 0, end = N, start = 0, step = 1} -> (N, )
  //  %2 = "ttir.multiply"(%0, 1, %1) -> (N,)
  //  %4 = "ttir.add"(%2, 0, %3) -> (N,)
  static bool isConstArangedTensor(mlir::Value inputValue, int64_t dim) {
    // 1. Check that inputValue is an add op.
    // 2. Check that the add's input is a multiply op.
    auto addOp = inputValue.getDefiningOp<AddOp>();
    if (!addOp) {
      return false;
    }
    auto addInput = addOp.getLhs();
    auto multiplyOp = addInput.getDefiningOp<MultiplyOp>();
    if (!multiplyOp) {
      return false;
    }

    // 1. Check that the multiply's input is an arange op.
    // 2. Check that the arange op has start=0, end=dim, step=1, and
    //    arange_dim=0.
    auto arangeOp = multiplyOp.getLhs().getDefiningOp<ArangeOp>();
    if (!arangeOp) {
      return false;
    }
    const int64_t arangeStart = arangeOp.getStart();
    const int64_t arangeEnd = arangeOp.getEnd();
    const int64_t arangeStep = arangeOp.getStep();
    const int64_t arangeDim = arangeOp.getArangeDimension();
    if (arangeStart != 0 || arangeEnd != dim || arangeStep != 1 ||
        arangeDim != 0) {
      return false;
    }

    return true;
  }
};

class PadPoolingFusionPattern : public mlir::OpRewritePattern<PoolingOp> {
public:
  using mlir::OpRewritePattern<PoolingOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(PoolingOp op, mlir::PatternRewriter &rewriter) const final {
    PadOp padToCompare;
    SmallVector<Value> newInputs;
    for (Value value : op.getInputs()) {
      PadOp padOp = value.getDefiningOp<PadOp>();
      if (!padOp) {
        return failure();
      }
      if (!padToCompare) {
        padToCompare = padOp;
      }

      if (padOp.getPadding() != padToCompare.getPadding()) {
        return failure();
      }
      newInputs.push_back(padOp.getInput());
    }

    ArrayRef<int32_t> padding = padToCompare.getPadding();
    SmallVector<int64_t> newPadding;
    // We add the padding of the input to the ops current padding
    // in the event the current PoolingOp already has non-zero padding
    for (const auto [a, b] : llvm::zip_equal(padding, op.getPadding())) {
      newPadding.push_back(a + b);
    }
    rewriter.modifyOpInPlace(op, [&]() {
      op.getInputsMutable().assign(newInputs);
      op.setPadding(newPadding);
    });

    return success();
  }
};

// The following OpRewritePattern fuses:
//     div(sum_pool(act), broadcast(reshape(sum_pool(const))))
// to:
//     avg_pool(act)
// when the denominator is such that the division is equivalent to an average
// pooling.
class AveragePoolingWithPoolingDenominatorFusionPattern
    : public mlir::OpRewritePattern<DivOp> {
public:
  using mlir::OpRewritePattern<DivOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DivOp op, mlir::PatternRewriter &rewriter) const final {
    // Numerator must be poolingOp.
    PoolingOp numerator = op.getLhs().getDefiningOp<PoolingOp>();
    if (!numerator) {
      return mlir::failure();
    }

    // Numerator must have the Sum pooling method.
    if (numerator.getPoolingMethod() != PoolingMethod::Sum) {
      return mlir::failure();
    }

    // Denominator must trace through first operands to pooling op.
    Value currentDenominator = op.getRhs();
    PoolingOp denominator;
    while (!currentDenominator.getDefiningOp<PoolingOp>()) {
      if (!currentDenominator.getDefiningOp()) {
        return mlir::failure();
      }
      // We expect that the pooling denominator is only reshaped (for rank
      // change) and broadcasted if any ops lies between at all. If any other
      // ops lie between the denominator pooling op and the div op, then we
      // cannot fuse.
      if (!isa<ReshapeOp, BroadcastOp>(currentDenominator.getDefiningOp())) {
        return mlir::failure();
      }
      currentDenominator = currentDenominator.getDefiningOp()->getOperand(0);
    }
    denominator = currentDenominator.getDefiningOp<PoolingOp>();

    // The denominator must have the Sum pooling method.
    if (denominator.getPoolingMethod() != PoolingMethod::Sum) {
      return mlir::failure();
    }

    // Denominator pooling op must have the same number of inputs as numerator.
    if (numerator.getInputs().size() != denominator.getInputs().size()) {
      return mlir::failure();
    }

    // Denominator pooling op must have all inputs be either a FullOp or a
    // ConstantOp.
    if (!llvm::all_of(denominator.getInputs(), [](Value v) {
          return isa<FullOp, ConstantOp>(v.getDefiningOp());
        })) {
      return mlir::failure();
    }

    // Besides the padding attribute, all attributes of the denominator must
    // match the rightmost sublist of the numerator's attributes.
    if (!matchRightMostSubList(numerator.getWindowDimensions(),
                               denominator.getWindowDimensions())) {
      return mlir::failure();
    }

    if (!matchRightMostSubList(numerator.getWindowStrides(),
                               denominator.getWindowStrides())) {
      return mlir::failure();
    }

    if (!matchRightMostSubList(numerator.getBaseDilations(),
                               denominator.getBaseDilations())) {
      return mlir::failure();
    }

    if (!matchRightMostSubList(numerator.getWindowDilations(),
                               denominator.getWindowDilations())) {
      return mlir::failure();
    }

    // The padding attribute of the denominator must
    // be all zeros.
    if (denominator.getPadding().empty()) {
      return mlir::failure();
    }

    for (int64_t paddingValue : denominator.getPadding()) {
      if (paddingValue != 0) {
        return mlir::failure();
      }
    }

    // For each denominator input, if it is a FullOp, its fill value must be 1
    // If it is a constant op, its value must be a tensor filled with ones, and
    // padded with zeroes according to the padding attribute of the numerator.
    for (Value input : denominator.getInputs()) {
      if (FullOp inputOp = input.getDefiningOp<FullOp>()) {
        // If the denominator is a pool of a full op, then
        // the numerator must have a padding attribute of all zeros.
        if (numerator.getPadding().empty()) {
          return mlir::failure();
        }
        if (!llvm::all_of(numerator.getPadding(),
                          [](int64_t p) { return p == 0; })) {
          return mlir::failure();
        }

        if (isa<IntegerAttr>(inputOp.getFillValue())) {
          int64_t value = dyn_cast<IntegerAttr>(inputOp.getFillValue())
                              .getValue()
                              .getSExtValue();
          if (value != 1) {
            return mlir::failure();
          }
        } else if (isa<FloatAttr>(inputOp.getFillValue())) {
          float value = dyn_cast<FloatAttr>(inputOp.getFillValue())
                            .getValue()
                            .convertToFloat();
          if (value != 1.0) {
            return mlir::failure();
          }
        } else {
          return mlir::failure();
        }
      } else if (ConstantOp inputOp = input.getDefiningOp<ConstantOp>()) {
        Type constantElementType = inputOp.getValue().getElementType();
        if (isa<IntegerType>(constantElementType)) {
          auto values = inputOp.getValue().getValues<APInt>();
          if (!constantTensorAllOnesWithPadding(
                  values, inputOp.getType().getShape(), numerator.getPadding(),
                  APInt::getZero(values[0].getBitWidth()),
                  APInt::getOneBitSet(values[0].getBitWidth(), 0))) {
            return mlir::failure();
          }
        } else if (isa<FloatType>(constantElementType)) {
          auto values = inputOp.getValue().getValues<APFloat>();
          if (!constantTensorAllOnesWithPadding(
                  values, inputOp.getType().getShape(), numerator.getPadding(),
                  APFloat::getZero(values[0].getSemantics()),
                  APFloat::getOne(values[0].getSemantics()))) {
            return mlir::failure();
          }
        } else {
          return mlir::failure();
        }
      } else {
        llvm_unreachable("We should have confirmed that all inputs are either "
                         "a FullOp or a ConstantOp by this point.");
      }
    }

    rewriter.replaceOpWithNewOp<PoolingOp>(
        op, numerator.getResultTypes(), numerator.getInputs(),
        numerator.getOutputs(), PoolingMethod::Average,
        numerator.getWindowDimensions(), numerator.getWindowStrides(),
        numerator.getBaseDilations(), numerator.getWindowDilations(),
        numerator.getPadding());

    return mlir::success();
  }

private:
  bool matchRightMostSubList(ArrayRef<int64_t> numeratorAttrs,
                             ArrayRef<int64_t> denominatorAttrs) const {
    if (denominatorAttrs.size() > numeratorAttrs.size()) {
      return false;
    }
    ArrayRef<int64_t> subList =
        numeratorAttrs.take_back(denominatorAttrs.size());
    return llvm::equal(subList, denominatorAttrs);
  }

  inline bool indexInPaddedRegion(ArrayRef<int64_t> index,
                                  ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> padding) const {
    for (size_t i = 0; i < index.size(); ++i) {
      int64_t lowPadding = padding[2 * i];
      int64_t highPadding = padding[2 * i + 1];

      if (index[i] < lowPadding || index[i] >= shape[i] - highPadding) {
        return true;
      }
    }
    return false;
  }

  inline SmallVector<int64_t>
  getTensorIndexFromBufferIndex(int64_t bufferIndex,
                                ArrayRef<int64_t> shape) const {
    SmallVector<int64_t> index;
    for (size_t i = 0; i < shape.size(); ++i) {
      index.push_back(bufferIndex % shape[i]);
      bufferIndex /= shape[i];
    }
    return index;
  }

  // This function will check if a tensor is filled with ones, and padded with
  // zeros according to `padding`. Example 1:
  //    padding: [1, 1, 1, 1]
  //    data with shape: (4, 4):
  //        [[0.0, 0.0, 0.0, 0.0],
  //         [0.0, 1.0, 1.0, 0.0],
  //         [0.0, 1.0, 1.0, 0.0],
  //         [0.0, 0.0, 0.0, 0.0]]
  // returns true
  //
  // Example 2:
  //    padding: [0, 0, 0, 0]
  //    data with shape: (4, 4):
  //        [[1.0, 1.0, 1.0, 1.0],
  //         [1.0, 1.0, 1.0, 1.0],
  //         [1.0, 1.0, 1.0, 1.0],
  //         [1.0, 1.0, 1.0, 1.0]]
  // returns true
  //
  // Example 3:
  //    padding: [1, 1, 1, 1]
  //    data with shape: (4, 4):
  //        [[1.0, 1.0, 1.0, 1.0],
  //         [1.0, 1.0, 1.0, 1.0],
  //         [1.0, 1.0, 1.0, 1.0],
  //         [1.0, 1.0, 1.0, 1.0]]
  // returns false

  template <typename NumericType>
  bool constantTensorAllOnesWithPadding(
      mlir::detail::ElementsAttrRange<
          mlir::detail::ElementsAttrIterator<NumericType>>
          data,
      ArrayRef<int64_t> shape, ArrayRef<int64_t> padding, NumericType zero,
      NumericType one) const {

    padding = padding.take_back(shape.size() * 2);
    for (size_t i = 0; i < data.size(); ++i) {
      SmallVector<int64_t> index = getTensorIndexFromBufferIndex(i, shape);
      if (indexInPaddedRegion(index, shape, padding)) {
        if (data[i] != zero) {
          return false;
        }
      } else if (data[i] != one) {
        return false;
      }
    }
    return true;
  }
};

namespace {
class ConcatenateHeadsUpdatePattern : public mlir::OpRewritePattern<ReshapeOp> {
  using mlir::OpRewritePattern<ReshapeOp>::OpRewritePattern;

  enum InputDimensions {
    INPUT_BATCH = 0,
    INPUT_NUM_HEADS = 1,
    INPUT_SEQ = 2,
    INPUT_HEAD_SIZE = 3
  };

  enum OutputDimensions { OUTPUT_BATCH = 0, OUTPUT_SEQ = 1, OUTPUT_HIDDEN = 2 };

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const final {
    PermuteOp permuteOp = reshapeOp.getInput().getDefiningOp<PermuteOp>();
    if (!permuteOp || !isFusable(permuteOp, reshapeOp)) {
      return mlir::failure();
    }

    ::mlir::TypedValue<::mlir::RankedTensorType> inputTensor =
        permuteOp.getInput();
    RankedTensorType inputTensorType = inputTensor.getType();
    ArrayRef<int64_t> inputShape = inputTensorType.getShape();

    RankedTensorType reshapeResultType =
        mlir::cast<RankedTensorType>(reshapeOp.getResult().getType());
    ArrayRef<int64_t> reshapeOutputShape = reshapeResultType.getShape();

    if (reshapeOutputShape.size() == 3) {
      // default case, the reshape output is 3D.
      // input shape: [batch_size, num_heads, sequence_size, head_size]
      // output shape: [batch_size, sequence_size, num_heads * head_size
      // (hidden)].
      Value concatHeadsOp = utils::createDPSOp<ConcatenateHeadsOp>(
          rewriter, reshapeOp.getLoc(), reshapeResultType, inputTensor);
      rewriter.replaceOp(reshapeOp, concatHeadsOp);
      return mlir::success();
    }

    if (reshapeOutputShape.size() == 2) {
      // input shape: [batch_size, num_heads, sequence_size, head_size]
      // output shape: [batch_size * sequence_size, num_heads * head_size]

      int64_t originalBatchSize = inputShape[INPUT_BATCH];
      int64_t originalSequenceSize = inputShape[INPUT_SEQ];

      // The new 3D shape for ConcatenateHeadsOp
      // [batch_size, sequence_size, num_heads * head_size]
      SmallVector<int64_t> newReshapeShape = {
          originalBatchSize, originalSequenceSize,
          reshapeOutputShape[1] // num_heads * head_size (hidden)
      };

      // Get element type from the original reshape result type
      RankedTensorType newReshapeType = mlir::RankedTensorType::get(
          newReshapeShape, reshapeResultType.getElementType());

      // Create ConcatenateHeadsOp with the new shape.
      Value concatHeadsOp = utils::createDPSOp<ConcatenateHeadsOp>(
          rewriter, reshapeOp.getLoc(), newReshapeType, inputTensor);

      rewriter.modifyOpInPlace(reshapeOp, [&]() {
        reshapeOp.getInputMutable().assign(concatHeadsOp);
      });
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  // The sequence of operations without a 'concatenate_heads' op is:
  // 1. Input Tensor: `tensor<#batch_size, #num_heads, #sequence_size,
  // #head_size>`
  // 2. `tensor.permute` op: Applied  with a permutation attribute of
  // `[0, 2, 1, 3]`, transforming the input to `tensor<#batch_size,
  // #sequence_size, #num_heads, #head_size>`.
  // 3. `reshape` op: `tensor.reshape(<permuted_tensor>, [#batch_size,
  // #sequence_size, #num_heads * #head_size])`
  bool isFusable(PermuteOp permuteOp, ReshapeOp reshapeOp) const {

    // Check that input shape is 4 dimensional.
    ArrayRef<int64_t> inputShape = permuteOp.getInput().getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();

    if (inputShape.size() != 4) {
      return false;
    }

    // Check if the permutation is {0, 2, 1, 3}.
    llvm::ArrayRef<int64_t> permutation = permuteOp.getPermutation();
    llvm::SmallVector<int64_t> expectedPermutation = {
        INPUT_BATCH, INPUT_SEQ, INPUT_NUM_HEADS, INPUT_HEAD_SIZE};

    if (!llvm::equal(permutation, expectedPermutation)) {
      return false;
    }

    if (reshapeOutputShape.size() == 3) {
      // Default case, the reshape output is 3D.
      // input shape: [batch_size, num_heads, sequence_size, head_size]
      // output shape: [batch_size, sequence_size, num_heads * head_size
      // (hidden)].
      if (inputShape[INPUT_BATCH] != reshapeOutputShape[OUTPUT_BATCH] ||
          inputShape[INPUT_SEQ] != reshapeOutputShape[OUTPUT_SEQ]) {
        return false;
      }

      if (inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE] !=
          reshapeOutputShape[OUTPUT_HIDDEN]) {
        return false;
      }
    } else if (reshapeOutputShape.size() == 2) {
      // If reshape output is 2D,
      // input shape: [batch_size, num_heads, sequence_size, head_size]
      // output shape: [batch_size * sequence_size, num_heads * head_size]
      SmallVector<int64_t> expectedReshapeOutputShape = {
          inputShape[INPUT_BATCH] * inputShape[INPUT_SEQ],
          inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE]};

      if (expectedReshapeOutputShape != reshapeOutputShape) {
        return false;
      }
    } else {
      // If the reshape output shape is not 2D or 3D, we cannot fuse.
      return false;
    }
    return true;
  }
};
} // namespace

namespace {
// This pattern fuses: 0.5 * x * gaussianCDF(x) in to ttir.gelu(x), where
// gaussianCDF is a linearly transformed cumulative distribution function of the
// gaussian distribution (or an approximation of one).
class GeluFusionPattern : public mlir::OpRewritePattern<ttir::MultiplyOp> {
public:
  using mlir::OpRewritePattern<ttir::MultiplyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttir::MultiplyOp op,
                  mlir::PatternRewriter &rewriter) const final {

    // arg1, arg2, and arg3 shall be some permutation of 0.5, x, and
    // gaussianCDF(x).

    // In the event that `op` contains `x` as one of its arguments, and `x`
    // itself is the result of a a multiply op, There will be two triplets of
    // arguments to check as we do not know which argument from
    // `multiply(multiply, multiply)` is `x`.
    SmallVector<std::tuple<Value, Value, Value>> args =
        getTripleMultiplyArgs(op);
    for (auto [arg1, arg2, arg3] : args) {
      if (args.empty()) {
        return failure();
      }

      Value halfConstantArg;
      Value gaussianResultArg;

      if (isScalarValue(arg1, HALF)) {
        halfConstantArg = arg1;
      } else if (isScalarValue(arg2, HALF)) {
        halfConstantArg = arg2;
      } else if (isScalarValue(arg3, HALF)) {
        halfConstantArg = arg3;
      } else {
        return failure();
      }

      GaussianCDFType gaussianCDFType;
      Value geluInput;
      if (std::tie(gaussianCDFType, geluInput) =
              getGaussianCDFTypeAndInput(arg1);
          gaussianCDFType != GaussianCDFType::None) {
        gaussianResultArg = arg1;
      } else if (std::tie(gaussianCDFType, geluInput) =
                     getGaussianCDFTypeAndInput(arg2);
                 gaussianCDFType != GaussianCDFType::None) {
        gaussianResultArg = arg2;
      } else if (std::tie(gaussianCDFType, geluInput) =
                     getGaussianCDFTypeAndInput(arg3);
                 gaussianCDFType != GaussianCDFType::None) {
        gaussianResultArg = arg3;
      } else {
        return failure();
      }

      // Now one of the three arguments must be the gelu input
      if (geluInput != arg1 && geluInput != arg2 && geluInput != arg3) {
        return failure();
      }

      // TODO(@LPanosTT): When the 'approximate' flag is modelled in
      // ttir/ttnn/runtime for the gelu op, we want to set it to 'true' if the
      // gaussianCDFType is Tanh.
      //     - For now, we will always use the default erf version. This should
      //       be OK as the tanh approximation is incredibly accurate.
      // https://github.com/tenstorrent/tt-mlir/issues/4486
      (void)gaussianCDFType;
      ttir::utils::replaceOpWithNewDPSOp<ttir::GeluOp>(
          rewriter, op, op.getResult().getType(), geluInput);

      return success();
    }
    return failure();
  }

private:
  enum class GaussianCDFType { Erf, Tanh, None };
  static constexpr double HALF = 0.5;
  static constexpr double RECIPROCAL_SQRT_2 = 0.7071067811865476;
  static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
  static constexpr double CUBED_COEFFICIENT = 0.044715;
  static constexpr double ONE = 1.0;
  static constexpr double THREE = 3.0;

  // Given a MultiplyOp, this will return three values if the input 'multiplyOp'
  // multiply(multiply(a, b), c) or multiply(a, multiply(b, c)).
  SmallVector<std::tuple<Value, Value, Value>>
  getTripleMultiplyArgs(ttir::MultiplyOp multiplyOp) const {
    Value arg1 = multiplyOp.getLhs();
    Value arg2 = multiplyOp.getRhs();
    SmallVector<std::tuple<Value, Value, Value>> args;
    if (ttir::MultiplyOp multiplyOp2 = arg1.getDefiningOp<ttir::MultiplyOp>()) {
      args.push_back(
          std::make_tuple(arg2, multiplyOp2.getLhs(), multiplyOp2.getRhs()));
    }
    if (ttir::MultiplyOp multiplyOp2 = arg2.getDefiningOp<ttir::MultiplyOp>()) {
      args.push_back(
          std::make_tuple(multiplyOp2.getLhs(), multiplyOp2.getRhs(), arg1));
    }

    return args;
  }

  // The gaussianCDF will be either 1 + erf(x/sqrt(2)) or 1 +
  // tanh(sqrt(2/pi) * (x + 0.044715 * x^3)) if gelu approximation is true.
  std::tuple<GaussianCDFType, Value>
  getGaussianCDFTypeAndInput(Value gaussianCDFResult) const {
    if (Value gaussianCDFInput = getErfGaussianCDFInput(gaussianCDFResult)) {
      return std::make_tuple(GaussianCDFType::Erf, gaussianCDFInput);
    }
    if (Value gaussianCDFInput = getTanhGaussianCDFInput(gaussianCDFResult)) {
      return std::make_tuple(GaussianCDFType::Tanh, gaussianCDFInput);
    }
    return std::make_tuple(GaussianCDFType::None, nullptr);
  }

  Value getTanhGaussianCDFInput(Value gaussianCDFResult) const {
    // The final op in this pattern must be:
    // 1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    //   ^ this add

    ttir::AddOp gaussianCDFAdd = gaussianCDFResult.getDefiningOp<ttir::AddOp>();
    if (!gaussianCDFAdd) {
      return nullptr;
    }

    // The add must have a constant operand of 1

    // isArgLhs will track if the argument to gelu is on the lhs or rhs of the
    // add op
    bool isArgLhs = false;
    ttir::FullOp one = getFullOpThroughTMChain(gaussianCDFAdd.getLhs());
    if (!one) {
      one = getFullOpThroughTMChain(gaussianCDFAdd.getRhs());
      isArgLhs = true;
    }
    if (!one) {
      return nullptr;
    }

    if (!isa<FloatAttr>(one.getFillValue())) {
      return nullptr;
    }
    APFloat value = dyn_cast<FloatAttr>(one.getFillValue()).getValue();
    if (!checkFloatIsNear(value.convertToFloat(), ONE)) {
      return nullptr;
    }

    // The other operand must be tanh.
    ttir::TanhOp tanh = isArgLhs
                            ? gaussianCDFAdd.getLhs().getDefiningOp<TanhOp>()
                            : gaussianCDFAdd.getRhs().getDefiningOp<TanhOp>();
    if (!tanh) {
      return nullptr;
    }

    // The argument to tanh must be:
    // tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    //                 ^ this multiply

    ttir::MultiplyOp multiplyArg = tanh.getInput().getDefiningOp<MultiplyOp>();
    if (!multiplyArg) {
      return nullptr;
    }

    bool argIsLhs;
    Value scalingArgument;
    if (isScalarValue(multiplyArg.getLhs(), SQRT_2_OVER_PI)) {
      argIsLhs = false;
      scalingArgument = multiplyArg.getRhs();
    } else if (isScalarValue(multiplyArg.getRhs(), SQRT_2_OVER_PI)) {
      argIsLhs = true;
      scalingArgument = multiplyArg.getLhs();
    } else {
      return nullptr;
    }

    // Scaling argument must be the result of:
    // x + 0.044715 * x^3
    //   ^ this add
    ttir::AddOp xPlusScaledXCubed =
        argIsLhs ? multiplyArg.getLhs().getDefiningOp<ttir::AddOp>()
                 : multiplyArg.getRhs().getDefiningOp<ttir::AddOp>();
    if (!xPlusScaledXCubed) {
      return nullptr;
    }

    Value xCubedResult;
    Value x;

    // Find the value of x in the pattern: x + 0.044715 * x^3.
    bool foundX = false;
    if (ttir::MultiplyOp lhsMultiply =
            xPlusScaledXCubed.getLhs().getDefiningOp<ttir::MultiplyOp>()) {
      x = xPlusScaledXCubed.getRhs();

      // find x^3 in the pattern: 0.044715 * x^3.

      // If the lhs of lhsMultiply is the scaling constant, then the rhs of
      // lhsMultiply must be the result of x^3.
      xCubedResult = isScalarValue(lhsMultiply.getLhs(), CUBED_COEFFICIENT)
                         ? lhsMultiply.getRhs()
                         : nullptr;

      // Otherwise, the rhs of lhsMultiply must be the result of x^3.
      if (!xCubedResult) {
        xCubedResult = isScalarValue(lhsMultiply.getRhs(), CUBED_COEFFICIENT)
                           ? lhsMultiply.getLhs()
                           : nullptr;
      }
      // Ensure that x is the same value which is cubed in the pattern.
      // If xCubedResult is not found (i.e neither argument to the multiply is
      // the result of a cubing), then foundX will remain false.
      if (xCubedResult) {
        foundX = x == getXCubedInput(xCubedResult);
      }
    }

    if (ttir::MultiplyOp rhsMultiply =
            xPlusScaledXCubed.getRhs().getDefiningOp<ttir::MultiplyOp>();
        !foundX && rhsMultiply) {
      x = xPlusScaledXCubed.getLhs();

      // find x^3 in the pattern: 0.044715 * x^3.

      // If the lhs of rhsMultiply is the scaling constant, then the rhs of
      // rhsMultiply must be the result of x^3.
      xCubedResult = isScalarValue(rhsMultiply.getLhs(), CUBED_COEFFICIENT)
                         ? rhsMultiply.getRhs()
                         : nullptr;

      // Otherwise, the rhs of rhsMultiply must be the result of x^3.
      if (!xCubedResult) {
        xCubedResult = isScalarValue(rhsMultiply.getRhs(), CUBED_COEFFICIENT)
                           ? rhsMultiply.getLhs()
                           : nullptr;
      }
      // Ensure that x is the same value which is cubed in the pattern.
      // If xCubedResult is not found (i.e neither argument to the multiply is
      // the result of a cubing), then foundX will remain false.
      if (xCubedResult) {
        foundX = x == getXCubedInput(xCubedResult);
      }
    }

    if (!foundX) {
      return nullptr;
    }

    return x;
  }

  // This will return the input Value of a sequence of ops which computes x^3 if
  // it exists, given the result of the sequence.
  Value getXCubedInput(Value xCubedResult) const {
    if (PowOp xCubed = xCubedResult.getDefiningOp<ttir::PowOp>()) {
      ttir::FullOp power = xCubed.getRhs().getDefiningOp<ttir::FullOp>();
      if (!power) {
        return nullptr;
      }
      if (!isa<FloatAttr>(power.getFillValue())) {
        return nullptr;
      }

      APFloat powerValue = dyn_cast<FloatAttr>(power.getFillValue()).getValue();
      if (!checkFloatIsNear(powerValue.convertToFloat(), THREE)) {
        return nullptr;
      }

      return xCubed.getLhs();
    }
    if (ttir::MultiplyOp xCubed =
            xCubedResult.getDefiningOp<ttir::MultiplyOp>()) {

      Value lhs = xCubed.getLhs();
      Value rhs = xCubed.getRhs();

      if (ttir::MultiplyOp lhsMultiply =
              lhs.getDefiningOp<ttir::MultiplyOp>()) {
        if (rhs == lhsMultiply.getRhs() && rhs == lhsMultiply.getLhs()) {
          return lhsMultiply.getLhs();
        }
      }

      if (ttir::MultiplyOp rhsMultiply =
              rhs.getDefiningOp<ttir::MultiplyOp>()) {
        if (lhs == rhsMultiply.getRhs() && lhs == rhsMultiply.getLhs()) {
          return rhsMultiply.getLhs();
        }
      }
    }
    return nullptr;
  }

  Value getErfGaussianCDFInput(Value gaussianCDFResult) const {
    // The final op in this pattern must be:
    // 1 + erf(x/sqrt(2))
    //   ^ this add

    ttir::AddOp gaussianCDFAdd = gaussianCDFResult.getDefiningOp<ttir::AddOp>();
    if (!gaussianCDFAdd) {
      return nullptr;
    }

    // The add must have a constant operand of 1.

    // isArgLhs will track if the argument to gelu is on the lhs or rhs of the
    // add op.
    bool isArgLhs = false;
    ttir::FullOp one = gaussianCDFAdd.getLhs().getDefiningOp<ttir::FullOp>();
    if (!one) {
      one = gaussianCDFAdd.getRhs().getDefiningOp<ttir::FullOp>();
      isArgLhs = true;
    }
    if (!one) {
      return nullptr;
    }

    if (!isa<FloatAttr>(one.getFillValue())) {
      return nullptr;
    }
    APFloat value = dyn_cast<FloatAttr>(one.getFillValue()).getValue();
    if (!checkFloatIsNear(value.convertToFloat(), ONE)) {
      return nullptr;
    }

    // erf must be the other operand.
    ttir::ErfOp erf =
        isArgLhs ? gaussianCDFAdd.getLhs().getDefiningOp<ttir::ErfOp>()
                 : gaussianCDFAdd.getRhs().getDefiningOp<ttir::ErfOp>();
    if (!erf) {
      return nullptr;
    }

    // If this pattern matches the erf version of the gaussianCDF, then
    // the erf argument must evaluate to x/sqrt(2). However, that means that
    // this pattern could be x / sqrt(2) or x / 1.4142.. or
    // x * reciprocal(sqrt(2)) or x * 0.7070...

    // So far the decomposition we have received from our frontends are in the
    // form: x * 0.70703125 So we will only check for this pattern for now.
    ttir::MultiplyOp multiplyArg =
        erf.getOperand(0).getDefiningOp<ttir::MultiplyOp>();
    if (!multiplyArg) {
      return nullptr;
    }

    Value scalingArgument;
    if (isScalarValue(multiplyArg.getLhs(), RECIPROCAL_SQRT_2)) {
      scalingArgument = multiplyArg.getRhs();
    } else if (isScalarValue(multiplyArg.getRhs(), RECIPROCAL_SQRT_2)) {
      scalingArgument = multiplyArg.getLhs();
    } else {
      return nullptr;
    }

    return scalingArgument;
  }

  bool checkFloatIsNear(double value, double trueValue) const {
    return value / trueValue - 1.0 <= 1.5e-3 &&
           value / trueValue - 1.0 >= -1.5e-3;
  }

  // This function will return true if the Value 'val' is a FullOp (or the
  // result of tensor-manipulation ops (Reshape, Permute, Broadcast) beginning
  // with a FullOp), with the fill_value near 'scalar'. It allows for an error
  // of 1.5%
  bool isScalarValue(Value val, double scalar) const {
    if (ttir::FullOp fullOp = getFullOpThroughTMChain(val)) {
      if (isa<FloatAttr>(fullOp.getFillValue())) {
        APFloat value = dyn_cast<FloatAttr>(fullOp.getFillValue()).getValue();
        if (checkFloatIsNear(value.convertToFloat(), scalar)) {
          return true;
        }
      }
    }

    return false;
  }

  FullOp getFullOpThroughTMChain(Value value) const {
    Operation *currentOp = value.getDefiningOp();

    while (isa_and_nonnull<ttir::ReshapeOp, ttir::BroadcastOp, ttir::PermuteOp>(
        currentOp)) {
      currentOp = currentOp->getOperand(0).getDefiningOp();
    }
    return mlir::dyn_cast_if_present<ttir::FullOp>(currentOp);
  }
};
} // namespace

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

      patterns.add<ReluFusionPattern>(&getContext());

      if (conv2dWithMultiplyEnabled) {
        patterns.add<Conv2dWithMultiply>(&getContext());
      }
      patterns.add<CacheFillUpdatePattern>(&getContext());
      patterns.add<ConcatenateHeadsUpdatePattern>(&getContext());

      patterns.add<PadPoolingFusionPattern>(&getContext());
      patterns.add<AveragePoolingWithPoolingDenominatorFusionPattern>(
          &getContext());

      patterns.add<BatchNormDecomposition>(&getContext());

      patterns.add<GeluFusionPattern>(&getContext());

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir
