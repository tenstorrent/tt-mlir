// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFUSING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
// Check if we can fuse conv followed by add into conv with bias.
// This pattern supports both:
// 1. Adding bias to conv without bias: conv(x, w) + b -> conv(x, w, b)
// 2. Adding bias to conv with existing bias: conv(x, w, b1) + b2 -> conv(x, w,
// b1+b2)
template <typename ConvOpType>
class ConvAddBias : public mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const final {
    auto components = getConvAndBias(srcOp);
    if (!components) {
      return mlir::failure();
    }

    auto convOp = components->first;
    auto bias = components->second;

    //  If conv already has bias we need to add it to the new bias.
    //  We can do it like this because we already checked that bias has valid
    //  shape.
    if (convOp.getBias()) {
      bias = utils::createDPSOp<AddOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(bias.getLoc(), "_bias_add"),
          mlir::cast<RankedTensorType>(bias.getType()), convOp.getBias(), bias);
    }

    if (bias.getDefiningOp() &&
        !bias.getDefiningOp()->isBeforeInBlock(convOp)) {

      // To move bias before conv, we need to ensure that all operations
      // in the UD chain are also moved before convOp.

      SetVector<Value> udChain = ttmlir::utils::getUseDefChain(bias);
      SetVector<Operation *> udChainOps =
          ttmlir::utils::filterOperations(udChain.getArrayRef());
      SetVector<Operation *> udChainSorted = topologicalSort(udChainOps);

      for (auto *op : udChainSorted) {
        if (op->isBeforeInBlock(convOp)) {
          continue;
        }
        op->moveBefore(convOp);
      }
    }

    rewriter.modifyOpInPlace(convOp,
                             [&]() { convOp.getBiasMutable().assign(bias); });
    rewriter.replaceAllOpUsesWith(srcOp, convOp);
    // The original conv op will be removed by DCE since it's no longer
    // used.
    return mlir::success();
  }

private:
  std::optional<mlir::TypedValue<mlir::RankedTensorType>>
  isFusable(ConvOpType convOp,
            mlir::TypedValue<mlir::RankedTensorType> bias) const {
    if (!convOp || !convOp->hasOneUse()) {
      return std::nullopt;
    }

    size_t outputFeatureDim = 0;
    if constexpr (std::is_same_v<ConvOpType, Conv2dOp>) {
      outputFeatureDim = 3;
    } else if constexpr (std::is_same_v<ConvOpType, ConvolutionOp>) {
      outputFeatureDim =
          convOp.getConvolutionLayoutAttr().getOutputFeatureDimension();
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    if (auto bcastOp = bias.getDefiningOp<BroadcastOp>()) {
      bias = bcastOp.getInput();
    }
    auto biasShape = bias.getType().getShape();
    auto outputShape =
        mlir::cast<mlir::RankedTensorType>(convOp.getOutput().getType())
            .getShape();

    if (biasShape.size() != outputShape.size()) {
      return std::nullopt;
    }

    auto featureDimSize =
        convOp.getResult().getType().getShape()[outputFeatureDim];
    for (auto [dim, dimSize] : llvm::enumerate(biasShape)) {
      if (dim == outputFeatureDim && dimSize != featureDimSize) {
        return std::nullopt;
      }
      if (dim != outputFeatureDim && dimSize != 1) {
        return std::nullopt;
      }
    }

    return bias;
  }

  std::optional<std::pair<ConvOpType, mlir::Value>>
  getConvAndBias(AddOp srcOp) const {
    auto lhs = srcOp.getLhs();
    auto rhs = srcOp.getRhs();
    auto lhsConvOp = lhs.getDefiningOp<ConvOpType>();
    auto rhsConvOp = rhs.getDefiningOp<ConvOpType>();
    if (auto fusableBias = isFusable(lhsConvOp, rhs)) {
      return std::make_pair(lhsConvOp, *fusableBias);
    }
    if (auto fusableBias = isFusable(rhsConvOp, lhs)) {
      return std::make_pair(rhsConvOp, *fusableBias);
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
    utils::replaceOpWithNewDPSOp<SoftmaxOp>(
        rewriter, divOp, divOp.getResult().getType(), expOp.getInput(),
        reduceDim, /*numericStable=*/false);

    return mlir::success();
  }
};

// This pattern detects and fuses numerically stable softmax operations:
// softmax(x - max(x)) -> softmax(x, numericStable=true).
//
// The pattern matches the following sequence:
// 1. max(x) along a dimension with keep_dim=true
// 2. broadcast the max value
// 3. subtract: x - broadcasted_max
// 4. softmax(x - broadcasted_max)
class NumericStableSoftmaxFusionPattern
    : public mlir::OpRewritePattern<SoftmaxOp> {
  using mlir::OpRewritePattern<SoftmaxOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(SoftmaxOp softmaxOp,
                  mlir::PatternRewriter &rewriter) const final {

    // Get the input to softmax.
    mlir::Value softmaxInput = softmaxOp.getInput();

    // Check if input is a subtract operation.
    auto subOp = softmaxInput.getDefiningOp<SubtractOp>();
    if (!subOp) {
      return mlir::failure();
    }

    // Get the operands of the subtract operation.
    mlir::Value originalInput = subOp.getLhs();
    mlir::Value subtractedValue = subOp.getRhs();

    auto broadcastOp = subtractedValue.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return mlir::failure();
    }

    mlir::Value maxValue = broadcastOp.getInput();

    // Check if the broadcasted value is a max operation.
    auto maxOp = maxValue.getDefiningOp<MaxOp>();
    if (!maxOp) {
      return mlir::failure();
    }

    // Verify that max operates on the same input as the original.
    if (maxOp.getInput() != originalInput) {
      return mlir::failure();
    }

    // Verify that max reduces along the same dimension as softmax
    // and has keep_dim=true for proper broadcasting.
    if (!maxOp.getDimArg() || !maxOp.getKeepDim()) {
      return mlir::failure();
    }

    mlir::ArrayAttr maxReduceDims = *maxOp.getDimArg();
    if (maxReduceDims.size() != 1) {
      return mlir::failure();
    }

    int64_t maxReduceDim =
        mlir::cast<mlir::IntegerAttr>(maxReduceDims[0]).getInt();
    if (maxReduceDim != softmaxOp.getDimension()) {
      return mlir::failure();
    }

    // Check usage patterns to ensure we can safely fuse.
    if (!subOp.getResult().hasOneUse() ||
        !broadcastOp.getResult().hasOneUse() ||
        !maxOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Replace with numerically stable softmax.
    utils::replaceOpWithNewDPSOp<SoftmaxOp>(
        rewriter, softmaxOp, softmaxOp.getResult().getType(), originalInput,
        softmaxOp.getDimension(),
        /*numericStable=*/true);

    return mlir::success();
  }
};

static bool isFullOpWithValue(Value value, float expectedValue,
                              float tolerance = 1e-6f) {
  Operation *op = value.getDefiningOp();
  if (!op) {
    return false;
  }

  // Check if it's a ZerosOp and expected value is 0
  if (expectedValue == 0.0f && isa<ZerosOp>(op)) {
    return true;
  }

  // Check if it's a FullOp with the expected fill value
  if (auto fullOp = dyn_cast<FullOp>(op)) {
    mlir::Attribute fillValue = fullOp.getFillValue();
    if (auto integerAttr = dyn_cast<IntegerAttr>(fillValue)) {
      return integerAttr.getValue().getSExtValue() ==
             static_cast<int64_t>(expectedValue);
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(fillValue)) {
      float actualValue = floatAttr.getValue().convertToFloat();
      return std::abs(actualValue - expectedValue) <= tolerance;
    }
    llvm_unreachable("Fill value should be integer or float");
  }

  return false;
}

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

    if (isFullOpWithValue(maximumOp.getLhs(), 0)) {
      input = maximumOp.getRhs();
    } else if (isFullOpWithValue(maximumOp.getRhs(), 0)) {
      input = maximumOp.getLhs();
    } else {
      return mlir::failure();
    }

    rewriter.setInsertionPoint(maximumOp);
    utils::replaceOpWithNewDPSOp<ReluOp>(
        rewriter, maximumOp, maximumOp.getResult().getType(), input);

    return mlir::success();
  }
};

// ReLU6 fusion pattern matcher that transforms:
//   minimum(relu(x), 6)
// into:
//   relu6(x)
//
// The pattern matches both operand orders:
//   - minimum(relu(x), 6)
//   - minimum(6, relu(x))
//
// The constant 6 can be represented as either an integer or float fill value.
class Relu6FusionPattern : public mlir::OpRewritePattern<MinimumOp> {
  using mlir::OpRewritePattern<MinimumOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MinimumOp minimumOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Try first to match pattern minimum(relu(x), 6)
    Value relu = minimumOp.getLhs();
    Value fullSix = minimumOp.getRhs();

    auto reluOp = relu.getDefiningOp<ReluOp>();

    // If lhs is not relu, try the reversed pattern minimum(6, relu(x))
    if (!reluOp) {
      reluOp = fullSix.getDefiningOp<ReluOp>();
      std::swap(relu, fullSix);
    }

    // Verify we found the expected pattern
    if (!reluOp || !reluOp.getInput().hasOneUse() ||
        !isFullOpWithValue(fullSix, 6)) {
      return mlir::failure();
    }

    // Replace with ReLU6
    utils::replaceOpWithNewDPSOp<Relu6Op>(rewriter, minimumOp,
                                          minimumOp.getResult().getType(),
                                          reluOp.getInput());
    return mlir::success();
  }
};

// Hardsigmoid fusion pattern matcher that transforms:
//   relu6(x+3) / 6
// into:
//   hardsigmoid(x)
class HardsigmoidFusionPattern : public mlir::OpRewritePattern<DivOp> {
  using mlir::OpRewritePattern<DivOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(DivOp divOp, mlir::PatternRewriter &rewriter) const final {
    // Match pattern: Div(Relu6(Add(x, 3)), 6)
    Value numerator = divOp.getLhs();
    Value denominator = divOp.getRhs();
    auto relu6Op = numerator.getDefiningOp<Relu6Op>();
    if (!relu6Op || !relu6Op.getResult().hasOneUse()) {
      return mlir::failure();
    }
    if (!isFullOpWithValue(denominator, 6)) {
      return mlir::failure();
    }

    auto addOp = relu6Op.getInput().getDefiningOp<AddOp>();
    if (!addOp || !addOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Check if one operand is constant 3 and the other is the input
    TypedValue<RankedTensorType> input;
    if (isFullOpWithValue(addOp.getRhs(), 3)) {
      input = addOp.getLhs();
    } else if (isFullOpWithValue(addOp.getLhs(), 3)) {
      input = addOp.getRhs();
    } else {
      return mlir::failure();
    }

    auto hardsigmoidOp = utils::createDPSOp<HardsigmoidOp>(
        rewriter, divOp.getLoc(), input.getType(), input);

    rewriter.replaceAllOpUsesWith(divOp, hardsigmoidOp);

    return mlir::success();
  }
};

// SiLU fusion pattern matcher that transforms:
// multiply(sigmoid(x), x)
// into:
// silu(x)
//
// The pattern matches both operand orders:
// - multiply(sigmoid(x), x)
// - multiply(x, sigmoid(x))
//
// When multiply inputs and output are typecasted, the typecasts are ignored and
// silu op is done in the type of the sigmoid input.
class SiluFusionPattern : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::Value lhs = multiplyOp.getLhs();
    mlir::Value rhs = multiplyOp.getRhs();

    // If either operand is a typecast, we want to look through it.
    if (auto lhsTypecast = lhs.getDefiningOp<TypecastOp>()) {
      lhs = lhsTypecast.getInput();
    }
    if (auto rhsTypecast = rhs.getDefiningOp<TypecastOp>()) {
      rhs = rhsTypecast.getInput();
    }

    if (lhs.getType() != rhs.getType()) {
      return mlir::failure();
    }

    SigmoidOp sigmoidOp = nullptr;
    mlir::Value otherOperand;

    if (auto lhsSigmoid = lhs.getDefiningOp<SigmoidOp>()) {
      sigmoidOp = lhsSigmoid;
      otherOperand = rhs;
    } else if (auto rhsSigmoid = rhs.getDefiningOp<SigmoidOp>()) {
      sigmoidOp = rhsSigmoid;
      otherOperand = lhs;
    }

    if (!sigmoidOp || !sigmoidOp->hasOneUse()) {
      return mlir::failure();
    }

    if (sigmoidOp.getInput() != otherOperand) {
      return mlir::failure();
    }

    auto inputType = sigmoidOp.getInput().getType();
    auto outputType = multiplyOp.getResult().getType();
    auto siluOp = utils::createDPSOp<SiluOp>(rewriter, multiplyOp->getLoc(),
                                             inputType, otherOperand);

    // If multiply inputs and output are typecasted, we need to add a typecast
    // after silu to convert back to the multiply output type.
    if (inputType != outputType) {
      auto typecastOp = utils::createDPSOp<TypecastOp>(
          rewriter, multiplyOp->getLoc(), inputType, siluOp.getResult());
      rewriter.replaceAllOpUsesWith(multiplyOp, typecastOp);
    } else {
      rewriter.replaceAllOpUsesWith(multiplyOp, siluOp);
    }
    return mlir::success();
  }
};

template <typename ConvOpType>
class ConvWithMultiply : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  /// Pattern: conv(input, weight) * scale or conv(input, weight, bias) * scale
  ///
  /// This pattern detects when a Convolution operation (with constant weights)
  /// is followed by a multiplication with a constant scale factor. It optimizes
  /// this pattern by pre-multiplying the convolution weights with the scale
  /// factor, which eliminates the runtime multiplication operation.
  /// Supports both convolutions with and without bias.
  ///
  /// Input pattern without bias:
  ///   %conv = conv(%input, %weight)       // weight is constant
  ///   %result = multiply(%conv, %scale)   // scale is constant
  ///   (1,1,1,out_channels)
  ///
  /// Output pattern:
  ///   %reshaped_scale = reshape(%scale) to (out_channels,1,1,1)
  ///   %scaled_weight = multiply(%weight, %reshaped_scale)
  ///   %result = conv(%input, %scaled_weight)
  ///
  /// Input pattern with bias:
  ///   %conv = conv(%input, %weight, %bias)  // weight and bias are constant
  ///   %result = multiply(%conv, %scale)     // scale is constant
  ///   (1,1,1,out_channels)
  ///
  /// Output pattern:
  ///   %reshaped_scale = reshape(%scale) to (out_channels,1,1,1)
  ///   %scaled_weight = multiply(%weight, %reshaped_scale)
  ///   %scaled_bias = multiply(%bias, %scale)
  ///   %result = conv(%input, %scaled_weight, %scaled_bias)

  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if this pattern is applicable.
    auto components = getConvAndScale(multiplyOp);
    if (!components) {
      return mlir::failure();
    }

    ConvOpType convOp = components->first;
    Value scaleValue = components->second;

    // Reshape scale to match weight dimensions and pre-multiply weights.
    Value reshapedScale = createReshapedScale(rewriter, scaleValue, convOp);

    // Get UD chain starting from the reshaped scale. This chain will be
    // moved before the convOp to ensure that weight scale can be
    // const-evaled.
    SetVector<Value> udChain = ttmlir::utils::getUseDefChain(reshapedScale);
    SetVector<Operation *> udChainOps =
        ttmlir::utils::filterOperations(udChain.getArrayRef());
    SetVector<Operation *> udChainSorted = topologicalSort(udChainOps);

    // We are not moving ops in UD chain that are already before the conv, as
    // they could have descendants that are also before conv but are not in
    // the UD chain.
    for (auto *op : udChainSorted) {
      if (op->isBeforeInBlock(convOp)) {
        continue;
      }
      op->moveBefore(convOp);
    }

    rewriter.setInsertionPoint(convOp);

    // Create scaled weights by multiplying the original weights with the
    // resshaped scale.
    Value scaledWeights = createScaledWeights(
        rewriter, convOp.getLoc(), convOp.getWeight(), reshapedScale);

    // Update conv to use scaled weights and replace multiply operation.
    rewriter.modifyOpInPlace(
        convOp, [&]() { convOp.getWeightMutable().assign(scaledWeights); });

    if (auto bias = convOp.getBias()) {
      Value scaledBias = createScaledBias(rewriter, bias, convOp, scaleValue);
      rewriter.modifyOpInPlace(
          convOp, [&]() { convOp.getBiasMutable().assign(scaledBias); });
    }

    rewriter.replaceAllOpUsesWith(multiplyOp, convOp);

    return mlir::success();
  }

private:
  static std::optional<std::pair<ConvOpType, mlir::Value>>
  getConvAndScale(MultiplyOp multiplyOp) {
    auto lhs = multiplyOp.getLhs();
    auto rhs = multiplyOp.getRhs();
    ConvOpType lhsConv = lhs.getDefiningOp<ConvOpType>();
    ConvOpType rhsConv = rhs.getDefiningOp<ConvOpType>();
    if (isCommutable(lhsConv, rhs)) {
      return std::make_pair(lhsConv, rhs);
    }
    if (isCommutable(rhsConv, lhs)) {
      return std::make_pair(rhsConv, lhs);
    }
    return std::nullopt;
  }

  // We can commute if scale is constant, has valid shape, and is not dependent
  // on convOp.
  static bool isCommutable(ConvOpType convOp,
                           mlir::TypedValue<RankedTensorType> scale) {
    // Conv should only have one use and that use should be a multiply op.
    if (!convOp || !convOp.getResult().hasOneUse()) {
      return false;
    }
    mlir::func::FuncOp funcOp =
        convOp->template getParentOfType<mlir::func::FuncOp>();
    llvm::SmallPtrSet<BlockArgument, 4> constParams =
        mlir::tt::ttcore::getConstsAndParams(funcOp);
    auto isConstant = [&constParams](mlir::Value value) {
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
        return constParams.contains(blockArg);
      }

      Operation *defOp = value.getDefiningOp();
      return defOp->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>();
    };

    RankedTensorType scaleType = scale.getType();

    // If scale is comming from broadcast then we want to use the input type
    // to the broadcast to check the shape.
    if (auto bcastOp =
            mlir::dyn_cast_if_present<BroadcastOp>(scale.getDefiningOp())) {
      scaleType = bcastOp.getInput().getType();
    }

    // Check if scale shape is with conv weight.
    if (!hasValidScaleShape(convOp, scaleType)) {
      return false;
    }

    // Now we want to check if operations which produce scale are
    // const-evalable. We do this by getting UD chain of the scale and then
    // checking if all inputs into this chain are constants.
    SetVector<Value> useDefChain = ttmlir::utils::getUseDefChain(scale);
    SetVector<BlockArgument> useDefChainBlockArgs =
        ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());

    // If there are block arguments in use def chain that are not
    // constants we cannot commute.
    if (!all_of(useDefChainBlockArgs, isConstant)) {
      return false;
    }

    // Since we want to move the scale chain before convOp we want to make
    // sure that the scale chain does not contain convOp.
    if (useDefChain.contains(convOp)) {
      return false;
    }

    return true;
  }

  // Scale must have rank 4 and size 1 in all dimensions except the output
  // feature dim.
  // For Conv2dOp: shape (1, 1, 1, out_channels)
  // For ConvolutionOp: depends on the convolution layout attribute
  static bool hasValidScaleShape(ConvOpType convOp,
                                 RankedTensorType scaleType) {
    if (!scaleType || scaleType.getRank() != 4) {
      return false;
    }

    size_t outputFeatureDim = 0;
    if constexpr (std::is_same_v<ConvOpType, Conv2dOp>) {
      outputFeatureDim = 3;
    } else if constexpr (std::is_same_v<ConvOpType, ConvolutionOp>) {
      outputFeatureDim =
          convOp.getConvolutionLayoutAttr().getOutputFeatureDimension();
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    auto outputShape =
        mlir::cast<mlir::RankedTensorType>(convOp.getOutput().getType())
            .getShape();

    for (auto [dim, dimSize] : llvm::enumerate(scaleType.getShape())) {
      if (dim == outputFeatureDim && dimSize != outputShape[outputFeatureDim]) {
        return false;
      }
      if (dim != outputFeatureDim && dimSize != 1) {
        return false;
      }
    }

    return true;
  }

  // There are two cases we want to handle here:
  // 1. Input scale is a constant tensor that only neeeds reshaping
  // 2. Input scale is a broadcast operation that needs reshaping
  //
  // In case of 1 we just add reshape operation to the scale tensor such that
  // it has the output feature dimension at the kernel output feature dimension
  // to match the weight layout.
  // For Conv2dOp: shape (out_channels, 1, 1, 1) from (1, 1, 1, out_channels)
  // For ConvolutionOp: depends on the convolution layout attribute
  //
  // In case of 2 we need to add reshape operation to the input of the of bcast
  // and then we create new broadcast operation with the new reshaped scale
  // which broadcasts the reshaped scale to the shape of the weight tensor.
  static Value createReshapedScale(mlir::PatternRewriter &rewriter,
                                   Value scaleValue, ConvOpType convOp) {
    // If scaleValue is broadcast operation we want to reshape its input.
    // Otherwise we reshape the scaleValue itself.
    Location loc = scaleValue.getLoc();
    RankedTensorType weightType =
        mlir::cast<RankedTensorType>(convOp.getWeight().getType());
    Value reshapeInput = scaleValue;
    if (auto bcastOp = mlir::dyn_cast_if_present<BroadcastOp>(
            scaleValue.getDefiningOp())) {
      rewriter.setInsertionPoint(bcastOp);
      reshapeInput = bcastOp.getInput();
    }

    // Get the scale's type.
    RankedTensorType scaleType =
        mlir::cast<RankedTensorType>(reshapeInput.getType());

    // Create a new shape to match weight layout.
    llvm::SmallVector<int64_t> newShape(scaleType.getShape());
    assert(newShape.size() == 4 &&
           "Scale tensor must have 4 dimensions for reshaping.");

    size_t outputFeatureDim = 0;
    size_t kernelOutputFeatureDim = 0;
    if constexpr (std::is_same_v<ConvOpType, Conv2dOp>) {
      outputFeatureDim = 3;
      kernelOutputFeatureDim = 0;
    } else if constexpr (std::is_same_v<ConvOpType, ConvolutionOp>) {
      outputFeatureDim =
          convOp.getConvolutionLayoutAttr().getOutputFeatureDimension();
      kernelOutputFeatureDim =
          convOp.getConvolutionLayoutAttr().getKernelOutputFeatureDimension();
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    // Swap between output and kernel output feature dimensions.
    std::swap(newShape[outputFeatureDim], newShape[kernelOutputFeatureDim]);

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
  /// Create pre-multiplied bias.
  static Value createScaledBias(mlir::PatternRewriter &rewriter,
                                Value biasValue, ConvOpType convOp,
                                Value scaleValue) {
    // Create a multiplication of the bias by the scale.
    return utils::createDPSOp<MultiplyOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(biasValue.getLoc(),
                                            "_bias_multiply"),
        mlir::cast<RankedTensorType>(biasValue.getType()), biasValue,
        scaleValue);
  }
};

// Tag all block arguments which are direct inputs to Conv2dOp and ConvolutionOp
// with discardable attribute. This is used during Layouting to check if
// function argument need to be put to Host/RM. This is temporary
// solution until we complete refactor of TTNNLayout see:
// https://github.com/tenstorrent/tt-mlir/issues/3432.
template <typename ConvOpType>
class ConvTagWeights : public mlir::OpRewritePattern<ConvOpType> {
public:
  ConvTagWeights(MLIRContext *context)
      : OpRewritePattern<ConvOpType>(context, PatternBenefit(2)) {}

  mlir::LogicalResult
  matchAndRewrite(ConvOpType conv,
                  mlir::PatternRewriter &rewriter) const final {
    Value weightValue = conv.getWeight();

    // Special case for bfp8 weights which don't come directly from
    // function argument but are created by TypecastOp.
    if (auto typecast =
            dyn_cast_if_present<TypecastOp>(weightValue.getDefiningOp())) {
      weightValue = typecast.getInput();
    }

    if (BlockArgument blockArg = dyn_cast<BlockArgument>(weightValue)) {
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

class BatchNormDecomposition
    : public mlir::OpRewritePattern<BatchNormInferenceOp> {
  using mlir::OpRewritePattern<BatchNormInferenceOp>::OpRewritePattern;

public:
  // This pattern decomposes the BatchNormInference operation into a sequence of
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
  matchAndRewrite(BatchNormInferenceOp batchNormOp,
                  mlir::PatternRewriter &rewriter) const final {

    // Used only paired with convolution
    auto *definingOp = batchNormOp.getOperand().getDefiningOp();
    if (!definingOp || (!isa<Conv2dOp, ConvolutionOp>(definingOp)) ||
        !definingOp->hasOneUse()) {
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

    // Validate that scale, offset, mean, and variance are either 1D or 4D
    // tensors
    auto isValidRank = [](RankedTensorType type) {
      return type.getRank() == 1 || type.getRank() == 4;
    };
    if (!isValidRank(scale.getType()) || !isValidRank(offset.getType()) ||
        !isValidRank(mean.getType()) || !isValidRank(variance.getType())) {
      return rewriter.notifyMatchFailure(
          batchNormOp, "BatchNorm arguments must be 1D or 4D tensors");
    }

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

    Value alphaReshaped = alpha;
    Value betaReshaped = beta;

    // Only reshape if alpha/beta don't already have rank 4
    if (alpha.getType().getRank() != 4) {
      SmallVector<int64_t> reshapeShape(4, 1);
      reshapeShape[dimension] = alpha.getType().getShape()[0];
      SmallVector<int32_t> reshapeShapeI32(reshapeShape.begin(),
                                           reshapeShape.end());
      alphaReshaped = utils::createDPSOp<ReshapeOp>(
          rewriter, loc, reshapeShape, alpha.getType().getElementType(),
          alpha.getType().getEncoding(), alpha,
          rewriter.getI32ArrayAttr(reshapeShapeI32));
    }

    if (beta.getType().getRank() != 4) {
      SmallVector<int64_t> reshapeShape(4, 1);
      reshapeShape[dimension] = beta.getType().getShape()[0];
      SmallVector<int32_t> reshapeShapeI32(reshapeShape.begin(),
                                           reshapeShape.end());
      betaReshaped = utils::createDPSOp<ReshapeOp>(
          rewriter, loc, reshapeShape, beta.getType().getElementType(),
          beta.getType().getEncoding(), beta,
          rewriter.getI32ArrayAttr(reshapeShapeI32));
    }

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
  ///   (B, N, S, H), the indices tensor represents the index where each element
  ///   in %updates should placed in the %cache.
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
  // %result = "ttir.scatter"(%cache, %indices, %updates)
  // Where:
  //    1. %cache and %updates are 4D tensors who's shape match except on the
  //    3rd dimension,
  //       (B, N, M, H) and (B, N, S, H) respectively, M being the max cache
  //       length and S being the sequence length of the update.
  //    2. %indices comes from a block argument representing the cachePositions
  //    tensor.
  static std::optional<mlir::Value>
  getCacheUpdatePositions(ttir::ScatterOp scatterOp) {
    // Check that the scatter op inputs represent a cache fill/update:
    //    1. The input is a 4D (B, N, M, H)
    //    2. The update tensor is a 4D tensor (B, N, S, H)
    //    3. The scatter indices is either a 1D equivalent tensor or 5D index
    //       grid tensor (B, N, S, H, 4). Both can be tracked to a block
    //       argument representing the cachePositions input.
    auto scatterIndices = scatterOp.getScatterIndices();
    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(scatterOp.getInput().getType()).getShape();
    ArrayRef<int64_t> scatterIdxShape =
        mlir::cast<RankedTensorType>(scatterIndices.getType()).getShape();
    ArrayRef<int64_t> updateShape =
        mlir::cast<RankedTensorType>(scatterOp.getUpdate().getType())
            .getShape();
    if (inputShape.size() != 4 || updateShape.size() != 4) {
      return std::nullopt;
    }

    if (!(inputShape[0] == updateShape[0] && inputShape[1] == updateShape[1] &&
          inputShape[3] == updateShape[3])) {
      return std::nullopt;
    }

    int cacheUpdateSize = updateShape[2];

    bool effectively1D = isEffectively1D(scatterIdxShape);
    if (effectively1D &&
        ttmlir::utils::volume(scatterIdxShape) != cacheUpdateSize) {
      return std::nullopt;
    }

    bool isIndexGrid =
        (scatterIdxShape.size() == 5 && scatterIdxShape[0] == inputShape[0] &&
         scatterIdxShape[1] == inputShape[1] &&
         scatterIdxShape[2] == cacheUpdateSize &&
         scatterIdxShape[3] == inputShape[3] && scatterIdxShape[4] == 4);

    // Check that scatter indices is either a 1D cache positions tensor or a 5D
    // index grid.
    if (!effectively1D && !isIndexGrid) {
      return std::nullopt;
    }

    // The cachePositions tensor is expected to be a 1D blockargument tensor
    // with the same size as the cache update size.
    auto useDefChain = ttmlir::utils::getUseDefChain(scatterIndices);
    auto blockArgs =
        ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
    for (auto blockArg : blockArgs) {
      // Check if the block argument is a cachePositions input.
      auto argTensorShape =
          mlir::cast<RankedTensorType>(blockArg.getType()).getShape();
      effectively1D = isEffectively1D(argTensorShape);
      if (!effectively1D) {
        continue;
      }
      if (ttmlir::utils::volume(argTensorShape) == cacheUpdateSize) {
        // We found the cachePositions input tensor.
        return blockArg;
      }
    }

    return std::nullopt;
  }

  static bool isEffectively1D(ArrayRef<int64_t> shape) {
    return llvm::count_if(shape, [](int64_t dim) { return dim != 1; }) <= 1;
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

// Fuse MatmulOp followed by AddOp into a single LinearOp.
// This pattern looks for an AddOp where one of its operands is the result of
// a MatmulOp and the other operand is a bias term. It then replaces the
// AddOp with a LinearOp that combines the functionality of both operations.
// The pattern also handles the case where the MatmulOp is followed by a
// ReshapeOp before the AddOp. In this case, it reshapes the bias term
// accordingly and creates a LinearOp followed by a ReshapeOp to maintain the
// original output shape.
class MatmulWithBiasFusionPattern : public mlir::OpRewritePattern<AddOp> {
public:
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final {
    if (MatmulOp matmulOp = getFusableMatmulOp(addOp); matmulOp) {
      TypedValue<RankedTensorType> bias = addOp.getLhs() == matmulOp.getResult()
                                              ? addOp.getRhs()
                                              : addOp.getLhs();
      Value validBias = findAndValidateBias(bias, matmulOp);
      Value matmulOpA = matmulOp.getA();
      Value matmulOpB = matmulOp.getB();
      LinearOp linearOp = ttir::utils::createDPSOp<ttir::LinearOp>(
          rewriter, addOp.getLoc(), addOp.getResult().getType(), matmulOpA,
          matmulOpB, validBias, matmulOp.getTransposeA(),
          matmulOp.getTransposeB());

      rewriter.replaceOp(addOp, linearOp);

      return mlir::success();
    }
    if (MatmulOp matmulOp = getFusableReshapedMatmulOp(addOp); matmulOp) {
      ReshapeOp reshapeOp =
          mlir::dyn_cast<ReshapeOp>(*matmulOp.getResult().getUsers().begin());
      TT_assertv(reshapeOp,
                 "MatmulWithBiasFusionPattern getFusable logic is broken.");

      TypedValue<RankedTensorType> bias =
          (addOp.getLhs() == reshapeOp.getResult()) ? addOp.getRhs()
                                                    : addOp.getLhs();
      Value validBias = findAndValidateBias(bias, matmulOp);

      Value matmulOpA = matmulOp.getA();
      Value matmulOpB = matmulOp.getB();
      llvm::ArrayRef<int64_t> addOpShape =
          addOp.getResult().getType().getShape();
      SmallVector<int32_t> addShapeI32(addOpShape.begin(), addOpShape.end());

      llvm::SmallVector<int64_t> linearOutputShape;
      auto validBiasType = mlir::cast<RankedTensorType>(validBias.getType());
      if (!OpTrait::util::getBroadcastedShape(matmulOp.getType().getShape(),
                                              validBiasType.getShape(),
                                              linearOutputShape)) {
        return mlir::failure();
      }
      auto matmulOutputType = matmulOp.getResult().getType();
      auto newOutputType = RankedTensorType::get(
          linearOutputShape, matmulOutputType.getElementType());

      LinearOp linearOp = ttir::utils::createDPSOp<ttir::LinearOp>(
          rewriter, addOp.getLoc(), newOutputType, matmulOpA, matmulOpB,
          validBias, matmulOp.getTransposeA(), matmulOp.getTransposeB());

      RankedTensorType addOpType = addOp.getType();

      Value finalReshape = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, addOp.getLoc(), addOpShape, addOpType.getElementType(),
          addOpType.getEncoding(), linearOp.getResult(),
          rewriter.getI32ArrayAttr(addShapeI32));
      rewriter.replaceOp(addOp, finalReshape);

      return mlir::success();
    }
    return mlir::failure();
  }

private:
  // Shared helper function to validate the matmul ops from an add op.
  MatmulOp getValidMatmulOp(MatmulOp matmulOpLHS, MatmulOp matmulOpRHS) const {
    if (matmulOpLHS && matmulOpRHS) {
      // Both operands are MatmulOps, cannot fuse.
      return nullptr;
    }

    MatmulOp matmulOp = matmulOpLHS ? matmulOpLHS : matmulOpRHS;
    if (!matmulOp) {
      return nullptr;
    }
    // Check that the MatmulOp has only one user.
    if (!matmulOp.getResult().hasOneUse()) {
      return nullptr;
    }

    return matmulOp;
  }

  MatmulOp getFusableReshapedMatmulOp(AddOp addOp) const {
    // Check MatmulOp -> ReshapeOp -> AddOp pattern.
    // This pattern should be either the LHS or RHS of the AddOp.

    ReshapeOp reshapeLHS = addOp.getLhs().getDefiningOp<ReshapeOp>();
    ReshapeOp reshapeRHS = addOp.getRhs().getDefiningOp<ReshapeOp>();

    MatmulOp matmulOpLHS =
        reshapeLHS ? reshapeLHS.getInput().getDefiningOp<MatmulOp>() : nullptr;

    MatmulOp matmulOpRHS =
        reshapeRHS ? reshapeRHS.getInput().getDefiningOp<MatmulOp>() : nullptr;

    MatmulOp validMatmulOp = getValidMatmulOp(matmulOpLHS, matmulOpRHS);
    if (!validMatmulOp) {
      return nullptr;
    }

    ReshapeOp reshapeOp =
        (validMatmulOp == matmulOpLHS) ? reshapeLHS : reshapeRHS;
    if (!reshapeOp.getResult().hasOneUse()) {
      return nullptr;
    }

    TypedValue<RankedTensorType> bias =
        (validMatmulOp == matmulOpLHS) ? addOp.getRhs() : addOp.getLhs();

    Value validBias = findAndValidateBias(bias, validMatmulOp);
    if (!validBias) {
      return nullptr;
    }
    return validMatmulOp;
  }

  MatmulOp getFusableMatmulOp(AddOp addOp) const {
    // Check if one operand is a MatmulOp with only this AddOp as its user.
    MatmulOp matmulOpLHS = addOp.getLhs().getDefiningOp<MatmulOp>();
    MatmulOp matmulOpRHS = addOp.getRhs().getDefiningOp<MatmulOp>();
    MatmulOp validMatmulOp = getValidMatmulOp(matmulOpLHS, matmulOpRHS);
    if (!validMatmulOp) {
      return nullptr;
    }
    TypedValue<RankedTensorType> bias =
        (validMatmulOp == matmulOpLHS) ? addOp.getRhs() : addOp.getLhs();
    Value validBias = findAndValidateBias(bias, validMatmulOp);
    if (!validBias) {
      return nullptr;
    }
    return validMatmulOp;
  }

  Value findAndValidateBias(TypedValue<RankedTensorType> bias,
                            MatmulOp matmulOp) const {
    // Check unsqueezed valid bias if it exists. Otherwise, check if bias is
    // valid and return.
    if (!bias) {
      return nullptr;
    }

    // Return unsqueezed valid 1D bias if it exists.
    if (ReshapeOp reshapeOp = bias.getDefiningOp<ReshapeOp>()) {
      TypedValue<RankedTensorType> reshapeInput = reshapeOp.getInput();
      RankedTensorType reshapeInputType = reshapeInput.getType();
      RankedTensorType reshapeOutputType = reshapeOp.getResult().getType();
      SmallVector<int64_t> squeezedOutputShape;
      for (int64_t dim : reshapeOutputType.getShape()) {
        if (dim != 1) {
          squeezedOutputShape.push_back(dim);
        }
      }
      ArrayRef<int64_t> inputShape = reshapeInputType.getShape();
      if (inputShape.size() == 1 &&
          squeezedOutputShape ==
              ArrayRef<int64_t>(reshapeInputType.getShape())) {
        // This is an unsqueeze operation. Check that reshape input is broadcast
        // compatible with MatmulOp output.
        llvm::SmallVector<int64_t> broadcastedShape;
        if (!OpTrait::util::getBroadcastedShape(matmulOp.getType().getShape(),
                                                reshapeInputType.getShape(),
                                                broadcastedShape)) {
          return nullptr;
        }
        return reshapeInput;
      }
    }

    // Check if bias is broadcast compatible with MatmulOp output.
    // Not checking if broadcast compatible with AddOp output and invalidating
    // such cases. This is because such cases would require additional ReshapeOp
    // after LinearOp to match the original AddOp output shape, which is not
    // handled in this pattern.
    RankedTensorType biasType = bias.getType();
    SmallVector<int64_t> broadcastedShape;
    if (OpTrait::util::getBroadcastedShape(matmulOp.getType().getShape(),
                                           biasType.getShape(),
                                           broadcastedShape)) {
      return bias;
    }

    return nullptr;
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
          return llvm::isa_and_present<FullOp, ConstantOp>(v.getDefiningOp());
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

// Global average pooling pattern matcher that transforms:
//   multiply(sum<dim=[2,3]>(act), 1/(h*w))
// into:
//   avg_pool(act, window=[h,w], stride=1, padding=0)
//
// Supports 4D input tensors with reduction over spatial dimensions (height and
// width).
//
// Input tensor layout is NCHW, while AvgPool2dOp expects NHWC layout, so
// PermuteOps are added before and after the AvgPool2dOp to convert between the
// two layouts, and a ReshapeOp is added after the permutes if the
// keepDim attribute of the original SumOp is false.

class GlobalAveragePoolingPattern : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

private:
  static constexpr float FLOAT_TOLERANCE = 1e-4f;
  static constexpr int64_t EXPECTED_INPUT_RANK = 4;
  static constexpr int64_t SPATIAL_HEIGHT_DIM = 2;
  static constexpr int64_t SPATIAL_WIDTH_DIM = 3;
  static constexpr int64_t CHANNEL_DIM = 1;

public:
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {

    SumOp sumOp = multiplyOp.getLhs().getDefiningOp<SumOp>();
    if (!validateSumOp(sumOp)) {
      return mlir::failure();
    }

    FullOp fullOp = multiplyOp.getRhs().getDefiningOp<FullOp>();
    auto inputShape = sumOp.getInput().getType().getShape();
    if (!validateFullOp(fullOp, inputShape)) {
      return mlir::failure();
    }
    // Create AvgPoolOp wrapped with two PermuteOps needed for NCHW<->NHWC
    // layout changes.
    auto insertedOp = createWrappedGlobalAvgPool(rewriter, sumOp);

    // If keepDim is false, we need to reshape the output of the pooling op
    // to keep output dimension same as input.
    if (!sumOp.getKeepDim()) {
      auto outputType = multiplyOp.getOutput().getType();
      auto reshapePoolOp =
          createReshapePoolOutOp(rewriter, insertedOp, outputType);
      rewriter.replaceOp(multiplyOp, reshapePoolOp.getResult());
    } else {
      rewriter.replaceOp(multiplyOp, insertedOp.getResult());
    }

    return mlir::success();
  };

private:
  bool validateFullOp(FullOp fullOp, ArrayRef<int64_t> inputShape) const {
    if (!fullOp) {
      return false;
    }
    int64_t inputHeight = inputShape[SPATIAL_HEIGHT_DIM];
    int64_t inputWidth = inputShape[SPATIAL_WIDTH_DIM];
    float expectedValue = 1.0f / static_cast<float>(inputHeight * inputWidth);
    float tolerance =
        std::max(FLOAT_TOLERANCE, std::abs(expectedValue) * FLOAT_TOLERANCE);
    return isFullOpWithValue(fullOp, expectedValue, tolerance);
  }

  bool validateSumOp(SumOp sumOp) const {
    if (!sumOp || !sumOp.getDimArg() || !sumOp->hasOneUse()) {
      return false;
    }

    auto inputShape = sumOp.getInput().getType().getShape();
    if (inputShape.size() != EXPECTED_INPUT_RANK) {
      return false;
    }

    // Check that dimensions 2 and 3 are being reduced (height and width)
    auto reduceDims = *sumOp.getDimArg();
    if (reduceDims.size() != 2) {
      return false;
    }

    llvm::SmallSet<int64_t, 2> dimSet;
    for (mlir::Attribute attr : reduceDims) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (!intAttr) {
        return false;
      }
      dimSet.insert(intAttr.getInt());
    }
    return dimSet.contains(SPATIAL_HEIGHT_DIM) &&
           dimSet.contains(SPATIAL_WIDTH_DIM);
  }

  PermuteOp createPermuteOp(mlir::PatternRewriter &rewriter,
                            std::vector<int64_t> currentLayout,
                            std::vector<int64_t> desiredLayout,
                            Value input) const {

    auto permutation = ttmlir::utils::generatePermutation(
        llvm::ArrayRef(currentLayout), llvm::ArrayRef(desiredLayout));
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputShape =
        ::ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
    auto outputType = RankedTensorType::get(
        outputShape, inputType.getElementType(), inputType.getEncoding());

    auto permuteOp = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(input.getLoc(), "_permute"),
        outputType, input, permutation);

    return permuteOp;
  }

  PermuteOp createWrappedGlobalAvgPool(mlir::PatternRewriter &rewriter,
                                       SumOp sumOp) const {

    // Input must be permuted from NCHW to NHWC for GlobalAvgPool2dOp, and then
    // back to NCHW after pooling.
    std::vector<int64_t> currentLayout{0, CHANNEL_DIM, SPATIAL_HEIGHT_DIM,
                                       SPATIAL_WIDTH_DIM};
    std::vector<int64_t> desiredLayout{0, SPATIAL_HEIGHT_DIM, SPATIAL_WIDTH_DIM,
                                       CHANNEL_DIM};

    // Permute input from NCHW to NHWC
    auto permuteOp = createPermuteOp(rewriter, currentLayout, desiredLayout,
                                     sumOp.getInput());

    auto outputType = createPoolingOutputType(permuteOp.getOutput().getType());

    auto loc = sumOp.getLoc();
    auto poolingOp = ttir::utils::createDPSOp<ttir::GlobalAvgPool2dOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_global_avg_pool"),
        outputType, permuteOp.getResult());

    // Permute output from NHWC back to NCHW
    auto inversePermuteOp = createPermuteOp(
        rewriter, desiredLayout, currentLayout, poolingOp.getResult());

    return inversePermuteOp;
  }

  RankedTensorType createPoolingOutputType(RankedTensorType inputType) const {
    SmallVector<int64_t> poolOutputShape(inputType.getShape());
    poolOutputShape[SPATIAL_HEIGHT_DIM - 1] = 1;
    poolOutputShape[SPATIAL_WIDTH_DIM - 1] = 1;

    return RankedTensorType::get(poolOutputShape, inputType.getElementType(),
                                 inputType.getEncoding());
  }

  ReshapeOp createReshapePoolOutOp(mlir::PatternRewriter &rewriter,
                                   Operation *op,
                                   RankedTensorType outputType) const {
    auto loc = op->getLoc();
    llvm::SmallVector<int64_t> outputShape(outputType.getShape());
    SmallVector<int32_t> outputShapeI32(outputShape.begin(), outputShape.end());

    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshape"),
        outputType, op->getResult(0), rewriter.getI32ArrayAttr(outputShapeI32));
  }
};

namespace {
using namespace ttmlir::utils::transformer;

class ConcatenateHeadsUpdatePattern : public mlir::OpRewritePattern<ReshapeOp> {
  using mlir::OpRewritePattern<ReshapeOp>::OpRewritePattern;

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
      patterns.add<ConvTagWeights<Conv2dOp>>(&getContext());
      patterns.add<ConvTagWeights<ConvolutionOp>>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<ConvAddBias<Conv2dOp>>(&getContext());
      patterns.add<ConvAddBias<ConvolutionOp>>(&getContext());

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
      patterns.add<NumericStableSoftmaxFusionPattern>(&getContext());

      patterns.add<ReluFusionPattern>(&getContext());

      if (conv2dWithMultiplyEnabled) {
        patterns.add<ConvWithMultiply<Conv2dOp>>(&getContext());
        patterns.add<ConvWithMultiply<ConvolutionOp>>(&getContext());
        patterns.add<BatchNormDecomposition>(&getContext());
      }
      patterns.add<CacheFillUpdatePattern>(&getContext());
      patterns.add<ConcatenateHeadsUpdatePattern>(&getContext());

      patterns.add<PadPoolingFusionPattern>(&getContext());
      patterns.add<AveragePoolingWithPoolingDenominatorFusionPattern>(
          &getContext());
      if (globalPoolFusingEnabled) {
        patterns.add<GlobalAveragePoolingPattern>(&getContext());
      }

      patterns.add<MatmulWithBiasFusionPattern>(&getContext());

      patterns.add<GeluFusionPattern>(&getContext());
      patterns.add<Relu6FusionPattern>(&getContext());
      patterns.add<SiluFusionPattern>(&getContext());
      patterns.add<HardsigmoidFusionPattern>(&getContext());

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir
