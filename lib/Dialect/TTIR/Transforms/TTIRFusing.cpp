// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>

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
      bias = rewriter.create<AddOp>(
          ttmlir::utils::appendLocationSuffix(bias.getLoc(), "_bias_add"),
          mlir::cast<RankedTensorType>(bias.getType()), convOp.getBias(), bias);
    }

    // Move bias UD chain before conv to keep the ordering of ops.
    utils::moveUDChainBefore(bias, convOp);

    rewriter.modifyOpInPlace(convOp,
                             [&]() { convOp.getBiasMutable().assign(bias); });
    rewriter.replaceOp(srcOp, convOp);
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
    if constexpr (std::is_same_v<ConvOpType, Conv2dOp> ||
                  std::is_same_v<ConvOpType, ConvTranspose2dOp> ||
                  std::is_same_v<ConvOpType, Conv3dOp>) {
      outputFeatureDim = convOp.getChannelDim();
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    bias = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
        utils::lookThrough<BroadcastOp>(bias));
    auto biasShape = bias.getType().getShape();
    auto outputShape =
        mlir::cast<mlir::RankedTensorType>(convOp.getType()).getShape();

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
    rewriter.replaceOpWithNewOp<ReductionOpTy>(
        reductionOp, reshapeOp.getResult().getType(), reductionOp.getInput(),
        /*keep_dim=*/rewriter.getBoolAttr(true), reductionOp.getDimArgAttr());

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
// 2. sum reduce(exp(x)) along a dimension with keep_dim=true
// 3. (optional) broadcast the sum reduce
// 4. divide exp(x) by the (broadcasted) sum reduce
class SoftmaxFusionPattern : public mlir::OpRewritePattern<DivOp> {
  using mlir::OpRewritePattern<DivOp>::OpRewritePattern;

public:
  // Pattern: div(exp(x), sum reduce(exp(x), keep_dim=true)) or
  //          div(exp(x), broadcast(sum reduce(exp(x), keep_dim=true))).
  mlir::LogicalResult
  matchAndRewrite(DivOp divOp, mlir::PatternRewriter &rewriter) const final {

    // Get the numerator (exp) and denominator.
    mlir::Value numerator = divOp.getLhs();
    mlir::Value denominator = divOp.getRhs();

    // Check that the numerator is an exp operation.
    auto expOp = numerator.getDefiningOp<ExpOp>();
    if (!expOp) {
      return mlir::failure();
    }

    // Check if the denominator is a broadcast operation or directly a sum
    // reduce. If broadcast folding has occurred, the broadcast may be
    // eliminated.
    BroadcastOp broadcastOp = denominator.getDefiningOp<BroadcastOp>();
    // Check that we have a sum reduce operation with keep_dim=true.
    auto sumOp = utils::findOpThrough<SumOp, BroadcastOp>(denominator);
    if (!sumOp || !sumOp.getKeepDim()) {
      return mlir::failure();
    }

    // Check that the sum reduce input is the same exp operation as the
    // numerator.
    if (sumOp.getInput() != expOp.getResult()) {
      return mlir::failure();
    }

    // Check correct user counts for each operation in the pattern:
    // - exp should have exactly 2 users (sum reduce and div)
    // - sum reduce should have exactly 1 user (div or broadcast)
    // - broadcast (if present) should have exactly 1 user (div)
    // TODO(milant): Relax requirements for fusing #3183.
    if (ttmlir::utils::countUsers(expOp.getResult()) != 2 ||
        !sumOp.getResult().hasOneUse()) {
      return mlir::failure();
    }
    if (broadcastOp && !broadcastOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Get the reduction dimension from the sum reduce operation.
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
    rewriter.replaceOpWithNewOp<SoftmaxOp>(divOp, divOp.getType(),
                                           expOp.getInput(), reduceDim,
                                           /*numericStable=*/false);

    return mlir::success();
  }
};

// This pattern detects and fuses numerically stable softmax operations:
// softmax(x - max(x)) -> softmax(x, numericStable=true).
//
// The pattern matches the following sequence:
// 1. max(x) along a dimension with keep_dim=true
// 2. (optional) broadcast the max value
// 3. subtract: x - (broadcasted) max
// 4. softmax(x - (broadcasted) max)
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

    // Check if the subtracted value is a broadcast operation or directly a max.
    // If broadcast folding has occurred, the broadcast may be eliminated.
    BroadcastOp broadcastOp = subtractedValue.getDefiningOp<BroadcastOp>();
    // Check if we have a max operation.
    auto maxOp = utils::findOpThrough<MaxOp, BroadcastOp>(subtractedValue);
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
    if (!subOp.getResult().hasOneUse() || !maxOp.getResult().hasOneUse()) {
      return mlir::failure();
    }
    if (broadcastOp && !broadcastOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Replace with numerically stable softmax.
    rewriter.replaceOpWithNewOp<SoftmaxOp>(
        softmaxOp, softmaxOp.getResult().getType(), originalInput,
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
    rewriter.replaceOpWithNewOp<ReluOp>(maximumOp, maximumOp.getType(), input);

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
    rewriter.replaceOpWithNewOp<Relu6Op>(
        minimumOp, minimumOp.getResult().getType(), reluOp.getInput());
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

    auto hardsigmoidOp =
        rewriter.create<HardsigmoidOp>(divOp.getLoc(), input.getType(), input);

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
    mlir::Value lhs = utils::lookThrough<TypecastOp>(multiplyOp.getLhs());
    mlir::Value rhs = utils::lookThrough<TypecastOp>(multiplyOp.getRhs());

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

    mlir::Value sigmoidInput =
        utils::lookThrough<TypecastOp>(sigmoidOp.getInput());
    if (sigmoidInput != otherOperand) {
      return mlir::failure();
    }

    auto inputType = sigmoidInput.getType();
    auto outputType = multiplyOp.getResult().getType();
    auto siluOp =
        rewriter.create<SiluOp>(multiplyOp->getLoc(), inputType, otherOperand);

    // If multiply inputs and output are typecasted, we need to add a typecast
    // after silu to convert back to the multiply output type.
    if (inputType != outputType) {
      auto typecastOp = rewriter.create<TypecastOp>(
          multiplyOp->getLoc(), outputType, siluOp.getResult());
      rewriter.replaceAllOpUsesWith(multiplyOp, typecastOp);
    } else {
      rewriter.replaceAllOpUsesWith(multiplyOp, siluOp);
    }
    return mlir::success();
  }
};

std::optional<mlir::Value> matchNumericallyStableSoftplus(mlir::Value op) {
  // common utility to match the numerically stable softplus
  // NumericallyStableSoftplus = where[cond, x, ln(1 + e^x)]
  WhereOp whereOp = op.getDefiningOp<WhereOp>();
  if (!whereOp || !whereOp->hasOneUse()) {
    return std::nullopt;
  }
  auto softplusInput = whereOp.getSecond();

  // [cond, x, ln(1 + e^x)]
  Log1pOp log1pOp = whereOp.getThird().getDefiningOp<Log1pOp>();
  if (!log1pOp || !log1pOp->hasOneUse()) {
    return std::nullopt;
  }

  GreaterThanOp gtOp = whereOp.getFirst().getDefiningOp<GreaterThanOp>();
  if (!gtOp) {
    return std::nullopt;
  }

  if (gtOp.getLhs() != softplusInput) {
    return std::nullopt;
  }

  ExpOp expOp = log1pOp.getInput().getDefiningOp<ExpOp>();
  if (!expOp || !expOp->hasOneUse()) {
    return std::nullopt;
  }

  if (softplusInput != expOp.getInput()) {
    return std::nullopt;
  }

  return softplusInput;
}

// Mish Fusion Pattern
class MishFusingPattern : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  // Match Pattern -  mish(x) = x * tanh(Softplus(x))
  // where, Softplus(x) = x > C? x : ln(1 + e^x) (numerically stable variant of
  // softplus(x)) Because for very large x, (1 + e^x) ~ e^x ln(e^x) = x
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::Value lhs = utils::lookThrough<TypecastOp>(multiplyOp.getLhs());
    mlir::Value rhs = utils::lookThrough<TypecastOp>(multiplyOp.getRhs());

    // Match multiply(x, tanh(softplus(x))) and
    // multiply(tanh(softplus(x)), x)
    mlir::Value originalInput;
    TanhOp tanhOp = nullptr;
    if (auto lhsTanhOp = lhs.getDefiningOp<TanhOp>()) {
      tanhOp = lhsTanhOp;
      originalInput = rhs;
    } else if (auto rhsTanhOp = rhs.getDefiningOp<TanhOp>()) {
      tanhOp = rhsTanhOp;
      originalInput = lhs;
    }

    if (!tanhOp || !tanhOp->hasOneUse()) {
      return mlir::failure();
    }

    // Check if tanh's input is softplus
    auto softplusInput = matchNumericallyStableSoftplus(tanhOp.getInput());
    if (!softplusInput || *softplusInput != originalInput) {
      return mlir::failure();
    }

    auto inputType = originalInput.getType();
    auto outputType = multiplyOp.getResult().getType();
    auto mishOp =
        rewriter.create<MishOp>(multiplyOp.getLoc(), inputType, originalInput);

    // If multiply inputs and output are typecasted, we need to add a typecast
    // after mish to convert it back to the original multiply Op's output type.
    if (inputType != outputType) {
      auto typecastOp = rewriter.create<TypecastOp>(
          multiplyOp->getLoc(), outputType, mishOp.getResult());
      rewriter.replaceOp(multiplyOp, typecastOp.getResult());
    } else {
      rewriter.replaceOp(multiplyOp, mishOp.getResult());
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

    // Move scale UD chain before conv to keep the ordering of ops.
    utils::moveUDChainBefore(reshapedScale, convOp);

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

  // Scale must have rank 4/5 and size 1 in all dimensions except the output
  // feature dim.
  // For Conv2dOp: shape (1, 1, 1, out_channels)
  // For Conv3dOp: shape (1, 1, 1, 1, out_channels)
  static bool hasValidScaleShape(ConvOpType convOp,
                                 RankedTensorType scaleType) {
    const int64_t expectedRank = std::is_same_v<ConvOpType, Conv3dOp> ? 5 : 4;
    if (!scaleType || scaleType.getRank() != expectedRank) {
      return false;
    }

    size_t outputFeatureDim = 0;
    if constexpr (std::is_same_v<ConvOpType, Conv2dOp> ||
                  std::is_same_v<ConvOpType, ConvTranspose2dOp> ||
                  std::is_same_v<ConvOpType, Conv3dOp>) {
      outputFeatureDim = convOp.getChannelDim();
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    auto outputShape =
        mlir::cast<mlir::RankedTensorType>(convOp.getType()).getShape();

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
    const size_t expectedRank = std::is_same_v<ConvOpType, Conv3dOp> ? 5 : 4;
    assert(newShape.size() == expectedRank &&
           "Scale tensor must have expected rank for reshaping.");

    size_t outputFeatureDim = 0;
    size_t kernelOutputFeatureDim = 0;
    if constexpr (std::is_same_v<ConvOpType, Conv2dOp> ||
                  std::is_same_v<ConvOpType, Conv3dOp>) {
      outputFeatureDim = convOp.getChannelDim();
      kernelOutputFeatureDim = 0;
    } else if constexpr (std::is_same_v<ConvOpType, ConvTranspose2dOp>) {
      outputFeatureDim = convOp.getChannelDim();
      kernelOutputFeatureDim = 1;
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    // Swap between output and kernel output feature dimensions.
    std::swap(newShape[outputFeatureDim], newShape[kernelOutputFeatureDim]);

    // Convert to int32 for the reshape operation.
    llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());

    // Create the reshape operation.
    auto reshapedScale = rewriter.create<ttir::ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_reshape"),
        RankedTensorType::get(newShape, scaleType.getElementType(),
                              scaleType.getEncoding()),
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
    return rewriter.create<ttir::BroadcastOp>(scaleValue.getLoc(), weightType,
                                              reshapedScale, broadcastDims);
  }

  /// Create pre-multiplied weights.
  static Value createScaledWeights(mlir::PatternRewriter &rewriter,
                                   Location loc, Value weightValue,
                                   Value reshapedScale) {
    // Create a multiplication of the weights by the reshaped scale.
    return rewriter.create<MultiplyOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_multiply"),
        mlir::cast<RankedTensorType>(weightValue.getType()), weightValue,
        reshapedScale);
  }
  /// Create pre-multiplied bias.
  static Value createScaledBias(mlir::PatternRewriter &rewriter,
                                Value biasValue, ConvOpType convOp,
                                Value scaleValue) {
    // Create a multiplication of the bias by the scale.
    return rewriter.create<MultiplyOp>(
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
    if (!definingOp ||
        (!isa<Conv2dOp>(definingOp) && !isa<ConvTranspose2dOp>(definingOp)) ||
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
    auto variancePlusEpsilon = rewriter.create<AddOp>(loc, variance.getType(),
                                                      variance, epsilonTensor);

    // std = sqrt(variance + epsilon)
    auto std =
        rewriter.create<SqrtOp>(loc, variance.getType(), variancePlusEpsilon);
    // alpha = scale / std
    auto alpha = rewriter.create<DivOp>(loc, scale.getType(), scale, std);

    // alphaMean = alpha * mean
    auto alphaMean =
        rewriter.create<MultiplyOp>(loc, mean.getType(), mean, alpha);

    // beta = offset - alphaMean
    auto beta =
        rewriter.create<SubtractOp>(loc, offset.getType(), offset, alphaMean);

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
      alphaReshaped = rewriter.create<ReshapeOp>(
          loc,
          RankedTensorType::get(reshapeShape, alpha.getType().getElementType(),
                                alpha.getType().getEncoding()),
          alpha, rewriter.getI32ArrayAttr(reshapeShapeI32));
    }

    if (beta.getType().getRank() != 4) {
      SmallVector<int64_t> reshapeShape(4, 1);
      reshapeShape[dimension] = beta.getType().getShape()[0];
      SmallVector<int32_t> reshapeShapeI32(reshapeShape.begin(),
                                           reshapeShape.end());
      betaReshaped = rewriter.create<ReshapeOp>(
          loc,
          RankedTensorType::get(reshapeShape, beta.getType().getElementType(),
                                beta.getType().getEncoding()),
          beta, rewriter.getI32ArrayAttr(reshapeShapeI32));
    }

    // alpha * x
    auto scaled =
        rewriter.create<MultiplyOp>(loc, input.getType(), input, alphaReshaped);
    // alpha * x + beta
    auto result = rewriter.create<AddOp>(loc, resultType, scaled, betaReshaped);

    rewriter.replaceOp(batchNormOp, result);

    return mlir::success();
  }
};

// Fuses the sum of two convolutions into a single convolution:
//   conv2d(x, w1, b1) + conv2d(x, w2, b2)
//   ->  conv2d(x, w1 + padded_w2, b1 + b2)
//
// Implements the RepVGG pattern where:
//   - w1 is a 3x3 kernel with padding 1
//   - w2 is a 1x1 kernel with padding 0
//   - padded_w2 is w2 padded to 3x3 with 0s
//   - Both convolutions share the same input, stride, dilation, and groups
//
// This pattern was is used in YOLOv9.
class RepVGGConvSumFusionPattern : public mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final {
    auto components = getConvPair(addOp);
    if (!components) {
      return mlir::failure();
    }

    Conv2dOp conv3x3 = components->first;
    Conv2dOp conv1x1 = components->second;

    auto paddedWeight = createPaddedWeight(rewriter, conv1x1);

    rewriter.setInsertionPoint(conv3x3);
    addWeight(rewriter, conv3x3, paddedWeight);
    addBias(rewriter, conv3x3, conv1x1);

    rewriter.replaceOp(addOp, conv3x3);

    return mlir::success();
  }

private:
  static std::optional<std::pair<Conv2dOp, Conv2dOp>> getConvPair(AddOp addOp) {
    auto lhs = addOp.getLhs();
    auto rhs = addOp.getRhs();

    auto lhsConv = lhs.getDefiningOp<Conv2dOp>();
    auto rhsConv = rhs.getDefiningOp<Conv2dOp>();

    if (!lhsConv || !rhsConv) {
      return std::nullopt;
    }

    if (!lhsConv->hasOneUse() || !rhsConv->hasOneUse()) {
      return std::nullopt;
    }

    if (lhsConv.getInput() != rhsConv.getInput()) {
      return std::nullopt;
    }

    if (isRepVGGConvPair(lhsConv, rhsConv)) {
      return std::make_pair(lhsConv, rhsConv);
    }
    if (isRepVGGConvPair(rhsConv, lhsConv)) {
      return std::make_pair(rhsConv, lhsConv);
    }
    return std::nullopt;
  }

  // Checks if the two convolutions match the RepVGG conv sum pattern:
  // - First has 3x3 kernel with padding 1
  // - Second has 1x1 kernel with padding 0
  // - They must have the same stride, dilation, and groups
  static bool isRepVGGConvPair(Conv2dOp conv1, Conv2dOp conv2) {
    if (!hasKernelShape(conv1, 3, 3) || !hasKernelShape(conv2, 1, 1)) {
      return false;
    }
    if (!hasPaddingValues(conv1, 1) || !hasPaddingValues(conv2, 0)) {
      return false;
    }
    if (conv1.getStride() != conv2.getStride() ||
        conv1.getDilation() != conv2.getDilation() ||
        conv1.getGroups() != conv2.getGroups()) {
      return false;
    }
    return true;
  }

  static bool hasKernelShape(Conv2dOp conv, int64_t kH, int64_t kW) {
    auto shape = conv.getWeight().getType().getShape();
    return shape.size() == 4 && shape[2] == kH && shape[3] == kW;
  }

  static bool hasPaddingValues(Conv2dOp conv, int32_t padValue) {
    auto padding = conv.getPadding();
    if (auto denseAttr = llvm::dyn_cast<DenseI32ArrayAttr>(padding)) {
      auto paddingValues = denseAttr.asArrayRef();
      return llvm::all_of(paddingValues,
                          [padValue](int32_t v) { return v == padValue; });
    }
    if (auto i32Attr = llvm::dyn_cast<IntegerAttr>(padding)) {
      return i32Attr.getInt() == padValue;
    }
    return false;
  }

  // Creates a padded weight by padding the 1x1 weight to 3x3.
  static Value createPaddedWeight(mlir::PatternRewriter &rewriter,
                                  Conv2dOp conv) {
    auto weight1x1 = conv.getWeight();
    auto weight1x1Type = mlir::cast<RankedTensorType>(weight1x1.getType());
    auto weight1x1Shape = weight1x1Type.getShape();

    // Create new shape for 3x3 weight: (N, C, 3, 3)
    SmallVector<int64_t> weight3x3Shape = {weight1x1Shape[0], weight1x1Shape[1],
                                           3, 3};
    auto weight3x3Type =
        RankedTensorType::get(weight3x3Shape, weight1x1Type.getElementType(),
                              weight1x1Type.getEncoding());

    // Pad the 1x1 weight to 3x3 by adding zeros around it
    SmallVector<int32_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1};
    return rewriter.create<PadOp>(
        ttmlir::utils::appendLocationSuffix(conv.getLoc(), "_pad"),
        weight3x3Type, weight1x1, rewriter.getDenseI32ArrayAttr(paddingValues),
        rewriter.getF32FloatAttr(0.0));
  }

  // Modifies conv to use a combined weight created by adding the additional
  // weight to the existing weight from conv.
  static void addWeight(mlir::PatternRewriter &rewriter, Conv2dOp conv,
                        Value additionalWeight) {
    auto existingWeight = conv.getWeight();
    assert(existingWeight.getType() == additionalWeight.getType() &&
           "Expected same weight type");

    // Move additional weight UD chain before conv to ensure it is before addOp.
    utils::moveUDChainBefore(additionalWeight, conv);

    auto combinedWeight = rewriter.create<AddOp>(
        ttmlir::utils::appendLocationSuffix(conv.getLoc(), "_weight_add"),
        existingWeight.getType(), additionalWeight, existingWeight);
    rewriter.modifyOpInPlace(
        conv, [&]() { conv.getWeightMutable().assign(combinedWeight); });
  }

  // Modifies conv1 to use a combined bias created by adding bias2 to bias1. If
  // only conv2 has bias, assigns it to conv1.
  static void addBias(mlir::PatternRewriter &rewriter, Conv2dOp conv1,
                      Conv2dOp conv2) {
    auto bias1 = conv1.getBias();
    auto bias2 = conv2.getBias();

    if (bias1 && bias2) {
      assert(bias1.getType() == bias2.getType() && "Expected same bias type");

      // Move bias2 UD chain before conv1 to ensure it is before addOp.
      utils::moveUDChainBefore(bias2, conv1);

      auto combinedBias = rewriter.create<AddOp>(
          ttmlir::utils::appendLocationSuffix(conv1.getLoc(), "_bias_add"),
          bias1.getType(), bias1, bias2);
      rewriter.modifyOpInPlace(
          conv1, [&]() { conv1.getBiasMutable().assign(combinedBias); });
    } else if (bias2) {
      rewriter.modifyOpInPlace(conv1,
                               [&]() { conv1.getBiasMutable().assign(bias2); });
    }
  }
};

// Fuse MatmulOp followed by AddOp into a single LinearOp.
// This pattern looks for an AddOp where one of its operands is the result of
// a MatmulOp and the other operand is a bias term. It then replaces the
// AddOp with a LinearOp that combines the functionality of both operations.
// The pattern also handles the case where the MatmulOp is followed by a
// ReshapeOp before the AddOp. In this case, it creates a LinearOp followed by
// a ReshapeOp to maintain the original output shape.
class MatmulWithBiasFusionPattern : public mlir::OpRewritePattern<AddOp> {
public:
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final {
    // Matmul -> Add pattern.
    MatmulOp matmulOp = nullptr;
    TypedValue<RankedTensorType> bias = nullptr;
    if (matmulOp = getFusableMatmulOp(addOp); matmulOp) {
      bias = addOp.getLhs() == matmulOp.getResult() ? addOp.getRhs()
                                                    : addOp.getLhs();
    } else if (matmulOp = getFusableReshapedMatmulOp(addOp); matmulOp) {
      ReshapeOp reshapeOp =
          mlir::dyn_cast<ReshapeOp>(*matmulOp.getResult().getUsers().begin());
      bias = (addOp.getLhs() == reshapeOp.getResult()) ? addOp.getRhs()
                                                       : addOp.getLhs();
    } else {
      return mlir::failure();
    }

    ReshapeOp biasReshapeOp = bias.getDefiningOp<ReshapeOp>();
    llvm::SmallVector<int64_t> broadcastShape;
    // Remove bias reshape op if the input can be broadcasted to matmul or add
    // output shape.
    if (biasReshapeOp &&
        mlir::OpTrait::util::getBroadcastedShape(
            matmulOp.getType().getShape(),
            biasReshapeOp.getInput().getType().getShape(), broadcastShape) &&
        (llvm::equal(broadcastShape, addOp.getType().getShape()) ||
         llvm::equal(broadcastShape, matmulOp.getType().getShape()))) {
      bias = biasReshapeOp.getInput();
    }

    Value matmulOpA = matmulOp.getA();
    Value matmulOpB = matmulOp.getB();
    RankedTensorType outputType = matmulOp.getResult().getType();
    RankedTensorType biasType = bias.getType();
    // tt-metal uses a composite LinearOp where the bias is added after the
    // matmul, and ttnn.add supports broadcasting of both operands. Otherwise,
    // tt-metal lowers to a fused LinearOp, which uses the matmul result shape
    // as the output shape. The composite LinearOp requires that the bias
    // second-to-last dim (of padded shape) does not match the tile height.
    // Update the output type to match the broadcasted shape in this case.
    llvm::SmallVector<int64_t> paddedBiasShape =
        ttnn::utils::getTilePaddedShape(biasType.getShape());
    if (paddedBiasShape.size() > 1 &&
        paddedBiasShape[paddedBiasShape.size() - 2] != ttnn::TILE_HEIGHT) {
      llvm::SmallVector<int64_t> broadcastOutputShape;
      mlir::OpTrait::util::getBroadcastedShape(matmulOp.getType().getShape(),
                                               bias.getType().getShape(),
                                               broadcastOutputShape);
      outputType = RankedTensorType::get(broadcastOutputShape,
                                         outputType.getElementType(),
                                         outputType.getEncoding());
    }
    LinearOp linearOp = rewriter.create<ttir::LinearOp>(
        addOp.getLoc(), outputType, matmulOpA, matmulOpB, bias,
        matmulOp.getTransposeA(), matmulOp.getTransposeB());

    llvm::SmallVector<int32_t> addShapeI32(addOp.getType().getShape().begin(),
                                           addOp.getType().getShape().end());
    Value finalReshape = rewriter.create<ttir::ReshapeOp>(
        addOp.getLoc(), addOp.getType(), linearOp.getResult(),
        rewriter.getI32ArrayAttr(addShapeI32));
    rewriter.replaceOp(addOp, finalReshape);

    return mlir::success();
  }

private:
  // Shared helper function to validate the matmul ops from an add op or reshape
  // op.
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
    // Find the valid matmul op and the reshape op it is coming from.
    ReshapeOp reshapeOnAddLHS = addOp.getLhs().getDefiningOp<ReshapeOp>();
    ReshapeOp reshapeOnAddRHS = addOp.getRhs().getDefiningOp<ReshapeOp>();

    MatmulOp matmulOnReshapeLHS =
        reshapeOnAddLHS ? reshapeOnAddLHS.getInput().getDefiningOp<MatmulOp>()
                        : nullptr;
    MatmulOp matmulOnReshapeRHS =
        reshapeOnAddRHS ? reshapeOnAddRHS.getInput().getDefiningOp<MatmulOp>()
                        : nullptr;

    MatmulOp validMatmulOp =
        getValidMatmulOp(matmulOnReshapeLHS, matmulOnReshapeRHS);
    if (!validMatmulOp) {
      return nullptr;
    }
    ReshapeOp validReshapeOp = (validMatmulOp == matmulOnReshapeLHS)
                                   ? reshapeOnAddLHS
                                   : reshapeOnAddRHS;
    if (!validReshapeOp.getResult().hasOneUse()) {
      return nullptr;
    }

    // Bias will come from the other operand of the AddOp. Check that its shape
    // is broadcastable with the matmul output shape. Check that expected new
    // linear shape volume matches the add output shape volume.
    TypedValue<RankedTensorType> bias =
        (validMatmulOp == matmulOnReshapeLHS) ? addOp.getRhs() : addOp.getLhs();
    if (!bias.hasOneUse()) {
      return nullptr;
    }

    RankedTensorType biasType = bias.getType();
    SmallVector<int64_t> linearWithBiasExpectedShape;
    if (!OpTrait::util::getBroadcastedShape(validMatmulOp.getType().getShape(),
                                            biasType.getShape(),
                                            linearWithBiasExpectedShape)) {
      return nullptr;
    }

    RankedTensorType addOpType = addOp.getType();
    ArrayRef<int64_t> addOpShape = addOpType.getShape();
    if (ttmlir::utils::volume(
            llvm::ArrayRef<int64_t>(linearWithBiasExpectedShape)) !=
        ttmlir::utils::volume(addOpShape)) {
      return nullptr;
    }

    return validMatmulOp;
  }

  MatmulOp getFusableMatmulOp(AddOp addOp) const {
    // Check if one operand is a MatmulOp with only this AddOp as its user.
    // Check bias operand has only one use.
    MatmulOp matmulOpLHS = addOp.getLhs().getDefiningOp<MatmulOp>();
    MatmulOp matmulOpRHS = addOp.getRhs().getDefiningOp<MatmulOp>();
    MatmulOp validMatmulOp = getValidMatmulOp(matmulOpLHS, matmulOpRHS);
    if (!validMatmulOp) {
      return nullptr;
    }
    TypedValue<RankedTensorType> bias =
        (validMatmulOp == matmulOpLHS) ? addOp.getRhs() : addOp.getLhs();
    if (!bias.hasOneUse()) {
      return nullptr;
    }
    return validMatmulOp;
  }
};

// Scaled sum to mean pattern matcher that transforms:
//   multiply(sum<dim=[...]>(act), 1/(dim1*dim2*...))
// into:
//   mean<dim=[...]>(act)
//
// Matches decomposed global average pooling from torch-xla.
class ScaledSumToMeanPattern : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

private:
  static constexpr float FLOAT_TOLERANCE = 1e-4f;

public:
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {
    SumOp sumOp = multiplyOp.getLhs().getDefiningOp<SumOp>();
    if (!sumOp || !sumOp.getDimArg() || !sumOp->hasOneUse()) {
      return mlir::failure();
    }

    auto reduceDims = *sumOp.getDimArg();
    auto inputShape = sumOp.getInput().getType().getShape();

    FullOp fullOp = multiplyOp.getRhs().getDefiningOp<FullOp>();
    if (!isValidScale(fullOp, inputShape, reduceDims)) {
      return mlir::failure();
    }

    auto meanOp = createMeanOp(rewriter, sumOp, reduceDims);

    rewriter.replaceOp(multiplyOp, meanOp.getResult());

    return mlir::success();
  }

private:
  static bool isValidScale(FullOp fullOp, ArrayRef<int64_t> inputShape,
                           ArrayAttr reduceDims) {
    if (!fullOp) {
      return false;
    }

    // Calculate expected value as 1 / (dim1 * dim2 * ...)
    int64_t product = 1;
    for (mlir::Attribute attr : reduceDims) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (!intAttr) {
        return false;
      }
      int64_t dim = (intAttr.getInt() + inputShape.size()) % inputShape.size();
      if (inputShape[dim] <= 0) {
        return false;
      }
      product *= inputShape[dim];
    }

    float expectedValue = 1.0f / product;
    float tolerance =
        std::max(FLOAT_TOLERANCE, std::abs(expectedValue) * FLOAT_TOLERANCE);
    return isFullOpWithValue(fullOp, expectedValue, tolerance);
  }

  MeanOp createMeanOp(mlir::PatternRewriter &rewriter, SumOp sumOp,
                      ArrayAttr reduceDims) const {
    auto inputType = sumOp.getInput().getType();
    auto outputType =
        createMeanOutputType(inputType, sumOp.getKeepDim(), reduceDims);

    auto loc = sumOp.getLoc();

    auto meanOp = rewriter.create<MeanOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_mean"), outputType,
        sumOp.getInput(),
        /*keep_dim=*/rewriter.getBoolAttr(sumOp.getKeepDim()),
        /*dim_arg=*/reduceDims);

    return meanOp;
  }

  RankedTensorType createMeanOutputType(RankedTensorType inputType,
                                        bool keepDim,
                                        ArrayAttr reduceDims) const {
    SmallVector<int64_t> outputShape;
    ArrayRef<int64_t> inputShape = inputType.getShape();

    SmallVector<int64_t> reduceDimIndices;
    reduceDimIndices.reserve(reduceDims.size());
    for (mlir::Attribute attr : reduceDims) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (intAttr) {
        reduceDimIndices.push_back(intAttr.getInt());
      }
    }

    if (keepDim) {
      outputShape.assign(inputShape.begin(), inputShape.end());
      for (int64_t dim : reduceDimIndices) {
        dim = (dim + inputShape.size()) % inputShape.size();
        outputShape[dim] = 1;
      }
    } else {
      outputShape.reserve(inputShape.size() - reduceDimIndices.size());
      for (size_t i = 0; i < inputShape.size(); ++i) {
        if (!llvm::is_contained(reduceDimIndices, i)) {
          outputShape.push_back(inputShape[i]);
        }
      }
    }
    return RankedTensorType::get(outputShape, inputType.getElementType(),
                                 inputType.getEncoding());
  }
};

// Spatial mean optimization pattern that transforms:
//   mean<dim=[1,2]>(act)
// into:
//   mean<dim=2>(reshape(act, [N,1,H*W,C]))
//
// The pattern reshapes input from [N, C, H, W] to [N, 1, H*W, C], then applies
// mean on dimension 2 with keepdim=true, as it is more efficient than reducing
// by two dimensions.
// If the original mean had keepdim=false, then the result is
// reshaped to remove the spatial dimensions too.
class SpatialMeanOptimizationPattern : public mlir::OpRewritePattern<MeanOp> {
  using mlir::OpRewritePattern<MeanOp>::OpRewritePattern;

private:
  static constexpr int64_t EXPECTED_INPUT_RANK = 4;
  static constexpr int64_t SPATIAL_HEIGHT_DIM = 1;
  static constexpr int64_t SPATIAL_WIDTH_DIM = 2;

public:
  mlir::LogicalResult
  matchAndRewrite(MeanOp meanOp, mlir::PatternRewriter &rewriter) const final {
    if (!isValidMean(meanOp)) {
      return mlir::failure();
    }

    auto input = meanOp.getInput();
    auto loc = meanOp.getLoc();
    bool keepDim = meanOp.getKeepDim();

    auto reshapedInput = createInputReshape(rewriter, loc, input);
    auto newMeanOp = createMean(rewriter, loc, reshapedInput);

    auto result = newMeanOp.getResult();
    if (!keepDim) {
      result = createOutputReshape(rewriter, loc, result);
    }

    rewriter.replaceOp(meanOp, result);
    return mlir::success();
  }

private:
  static bool isValidMean(MeanOp meanOp) {
    if (!meanOp || !meanOp.getDimArg() || !meanOp->hasOneUse()) {
      return false;
    }

    auto inputShape = meanOp.getInput().getType().getShape();
    if (inputShape.size() != EXPECTED_INPUT_RANK) {
      return false;
    }

    auto reduceDims = *meanOp.getDimArg();
    if (reduceDims.size() != 2) {
      return false;
    }

    llvm::SmallSet<int64_t, 4> dimSet;
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

  static ReshapeOp createInputReshape(mlir::PatternRewriter &rewriter,
                                      Location loc, Value input) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto shape = inputType.getShape();

    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[SPATIAL_HEIGHT_DIM] = 1;
    newShape[SPATIAL_WIDTH_DIM] =
        shape[SPATIAL_HEIGHT_DIM] * shape[SPATIAL_WIDTH_DIM];

    SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
    auto outputType = RankedTensorType::get(
        newShape, inputType.getElementType(), inputType.getEncoding());
    return rewriter.create<ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_input_reshape"), outputType,
        input, rewriter.getI32ArrayAttr(newShapeI32));
  }

  static MeanOp createMean(mlir::PatternRewriter &rewriter, Location loc,
                           Value reshaped) {
    auto reshapedType = mlir::cast<RankedTensorType>(reshaped.getType());
    auto shape = reshapedType.getShape();

    SmallVector<int64_t> outputShape(shape.begin(), shape.end());
    outputShape[SPATIAL_WIDTH_DIM] = 1;

    auto outputType = RankedTensorType::get(
        outputShape, reshapedType.getElementType(), reshapedType.getEncoding());

    return rewriter.create<MeanOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_mean"), outputType, reshaped,
        /*keep_dim=*/rewriter.getBoolAttr(true),
        /*dim_arg=*/
        rewriter.getArrayAttr({rewriter.getI32IntegerAttr(SPATIAL_WIDTH_DIM)}));
  }

  static ReshapeOp createOutputReshape(mlir::PatternRewriter &rewriter,
                                       Location loc, Value meanResult) {
    auto meanType = mlir::cast<RankedTensorType>(meanResult.getType());
    auto shape = meanType.getShape();

    SmallVector<int64_t> newShape;
    newShape.reserve(shape.size() - 2);
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != SPATIAL_HEIGHT_DIM && i != SPATIAL_WIDTH_DIM) {
        newShape.push_back(shape[i]);
      }
    }

    SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
    auto outputType = RankedTensorType::get(newShape, meanType.getElementType(),
                                            meanType.getEncoding());
    return rewriter.create<ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_output_reshape"), outputType,
        meanResult, rewriter.getI32ArrayAttr(newShapeI32));
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
      Value concatHeadsOp = rewriter.create<ConcatenateHeadsOp>(
          reshapeOp.getLoc(), reshapeResultType, inputTensor);
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
      Value concatHeadsOp = rewriter.create<ConcatenateHeadsOp>(
          reshapeOp.getLoc(), newReshapeType, inputTensor);

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

template <typename MatMulOpType>
class SplitQueryKeyValueAndSplitHeadsUpdatePattern
    : public mlir::OpRewritePattern<ReshapeOp> {
  using mlir::OpRewritePattern<ReshapeOp>::OpRewritePattern;

  // A multi-head attention pattern looks like the following
  // (for each of Query, Key, and Value):
  //
  // Attention --> Reshape ----\
  //                           \
  // Q/K/V Weight ------------+--> Matmul/Linear --> Reshape --> Permute (...)
  //                           /
  // (opt) Q/K/V Bias --------/
  //
  // The fused pattern concatenates weights, (if any) biases, and replaces the
  // permute ops with outputs from splitquerykeyvalueandsplitheads op.
  // The fused pattern looks like the following:
  //
  // Q/K/V Weights -----> Concat Weights \
  //                                      \
  // (opt)Q/K/V Biases -> Concat Biases --+--> Matmul/Linear --> Split QKV
  //                                      /
  // Attention ---------> Reshape -------/

  // Reshape input is [batch_size, sequence_length, hidden_dimensions].
  // For Multi-Head Attention:
  // Query is [batch_size, number of kv heads, sequence_length, head_size].
  // Key is [batch_size, number of kv heads, sequence_length, head_size].
  // Value is [batch_size, number of kv heads, sequence_length, head_size].
  // If key is transposed:
  // Key is [batch_size, number of kv heads, head_size, sequence_length].

  enum InputDimensions {
    I_BATCH_SIZE = 0,
    I_SEQUENCE_LENGTH = 1,
    I_HIDDEN_DIMENSION = 2,
  };
  enum MHAExpectedOutputDimensions {
    O_BATCH_SIZE = 0,
    O_NUM_HEADS = 1,
    O_SEQUENCE_LENGTH = 2,
    O_HEAD_SIZE = 3,
  };
  enum MHATransposedKeyDimensions {
    K_BATCH_SIZE = 0,
    K_NUM_KV_HEADS = 1,
    K_HEAD_SIZE = 2,
    K_SEQUENCE_LENGTH = 3,
  };
  enum AttentionType {
    MHA = 0,  // Multi-Head Attention
    GQA = 1,  // Grouped Query Attention
    NONE = 2, // Not an attention pattern
  };

  llvm::SmallVector<int64_t> expectedPermute = {0, 2, 1, 3};
  llvm::SmallVector<int64_t> expectedTransposedPermute = {0, 2, 3, 1};

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check for Multi-Head Attention linear or matmul ops
    // that use reshape output.
    llvm::SmallVector<MatMulOpType> matmulOps = {};
    llvm::SmallVector<PermuteOp> permuteOps = {};

    AttentionType attentionType =
        findAttentionType(reshapeOp, matmulOps, permuteOps);
    if (attentionType == AttentionType::NONE) {
      return failure();
    }

    // Hoist preprocessing ops for key and value operands to appear before the
    // query head. Insert the fused op immediately after the query head. This
    // ensures all preprocessing for query, key, and value happens before the
    // fused op, and any uses of their results remain correctly ordered after
    // the fused op.

    PermuteOp firstPermuteOp = permuteOps[0];
    for (size_t i = 1; i < permuteOps.size(); ++i) {
      if (permuteOps[i]->isBeforeInBlock(firstPermuteOp)) {
        firstPermuteOp = permuteOps[i];
      }
    }
    hoistPreprocessingOps(permuteOps);
    rewriter.setInsertionPointAfter(firstPermuteOp);

    if (attentionType == AttentionType::MHA) {
      return fuseMHA(rewriter, reshapeOp, matmulOps, permuteOps);
    }
    if (attentionType == AttentionType::GQA) {
      return fuseGQA(rewriter, reshapeOp, matmulOps, permuteOps);
    }
    return mlir::failure();
  }

private:
  mlir::LogicalResult fuseGQA(mlir::PatternRewriter &rewriter,
                              ReshapeOp reshapeOp,
                              llvm::SmallVector<MatMulOpType> &matmulOps,
                              llvm::SmallVector<PermuteOp> &permuteOps) const {

    // Extract dimensions from reshape op and matmul ops.
    Value reshapeOutput = reshapeOp.getResult();
    MatMulOpType queryMatmulOp = matmulOps[0];
    MatMulOpType keyMatmulOp = matmulOps[1];

    // Concatenate key and value weights along dimension determined by
    // transposeB attribute.
    TypedValue<RankedTensorType> keyWeightMatrix = matmulOps[1].getB();
    TypedValue<RankedTensorType> valueWeightMatrix = matmulOps[2].getB();

    std::size_t dimToConcatWeights = keyMatmulOp.getTransposeB() ? 0 : 1;

    // Assert that keyMatmulOp A and B have rank 2.
    TT_assertv((keyMatmulOp.getA().getType().getRank() == 2 &&
                keyMatmulOp.getB().getType().getRank() == 2),
               "Expected rank 2 for MatMulOp operands A and B");

    ttir::ConcatOp concatenatedWeightMatrix = concatenateAlongDim(
        rewriter, valueWeightMatrix.getLoc(),
        ValueRange{keyWeightMatrix, valueWeightMatrix}, dimToConcatWeights);

    Value concatenatedBias = nullptr;
    if constexpr (std::is_same_v<LinearOp, MatMulOpType>) {
      Value keyBias = matmulOps[1].getBias();
      Value valueBias = matmulOps[2].getBias();
      std::size_t concatDim = matmulOps[1].getBias().getType().getRank() - 1;
      concatenatedBias =
          concatenateAlongDim(rewriter, valueBias.getLoc(),
                              ValueRange{keyBias, valueBias}, concatDim);
    }

    // Get the original output shape from one of the original linear ops.
    ArrayRef<int64_t> originalLinearOutputShape =
        keyMatmulOp.getType().getShape();
    llvm::SmallVector<int64_t> linearOutputShape(
        originalLinearOutputShape.begin(), originalLinearOutputShape.end());

    // Multiply the last dimension by 2 (since we're concatenating K, V).
    linearOutputShape.back() = originalLinearOutputShape.back() * 2;

    // Create linear operation with concatenated weights and bias.
    RankedTensorType linearOutputType = RankedTensorType::get(
        linearOutputShape, keyMatmulOp.getType().getElementType());

    SmallVector<Value> inputs;
    if (concatenatedBias) {
      inputs = {reshapeOutput, concatenatedWeightMatrix, concatenatedBias};
    } else {
      inputs = {reshapeOutput, concatenatedWeightMatrix};
    }

    MatMulOpType matrixMultOp = rewriter.create<MatMulOpType>(
        keyMatmulOp.getLoc(), TypeRange{linearOutputType}, inputs,
        keyMatmulOp->getAttrs());

    TT_assertv(matrixMultOp, "Expected valid matrix multiplication operation");

    // Reshape query matmul / linear op from [batch_size * sequence_size,
    // query_hidden_size] to [batch_size, sequence_size, query_hidden_size].
    // Reshape kv matmul / linear op from [batch_size * sequence_size,
    // 2 * kv_hidden_size] to [batch_size, sequence_size, 2 * kv_hidden_size].

    ArrayRef<int64_t> inputShape = reshapeOp.getInput().getType().getShape();
    int32_t batchSize = inputShape[I_BATCH_SIZE];
    int32_t sequenceLength = inputShape[I_SEQUENCE_LENGTH];

    auto queryType = permuteOps[0].getType();
    auto keyType = permuteOps[1].getType();
    auto valueType = permuteOps[2].getType();

    auto queryShape = queryType.getShape();
    auto keyShape = keyType.getShape();
    auto valueShape = valueType.getShape();

    int32_t headSize = keyShape[O_HEAD_SIZE];           // 128
    int32_t numQueryHeads = queryShape[O_NUM_HEADS];    // 24
    int numKVHeads = keyShape[O_NUM_HEADS];             // 8
    int32_t KVHiddenSize = headSize * numKVHeads;       // 1024
    int32_t queryHiddenSize = headSize * numQueryHeads; // 3072

    SmallVector<int64_t> queryReshapeShape = {batchSize, sequenceLength,
                                              queryHiddenSize};
    auto queryReshapeElementType = queryMatmulOp.getType().getElementType();
    auto queryReshapeEncoding = queryMatmulOp.getType().getEncoding();
    SmallVector<int32_t> queryReshapeShapeI32(queryReshapeShape.begin(),
                                              queryReshapeShape.end());
    RankedTensorType queryReshapeTy = RankedTensorType::get(
        queryReshapeShape, queryReshapeElementType, queryReshapeEncoding);
    ReshapeOp queryReshapeOp = rewriter.create<ReshapeOp>(
        matrixMultOp.getLoc(), queryReshapeTy, queryMatmulOp.getResult(),
        rewriter.getI32ArrayAttr(queryReshapeShapeI32));

    SmallVector<int64_t> kvReshapeShape = {batchSize, sequenceLength,
                                           KVHiddenSize * 2};
    auto kvReshapeElementType = keyMatmulOp.getType().getElementType();
    auto kvReshapeEncoding = keyMatmulOp.getType().getEncoding();
    SmallVector<int32_t> kvReshapeShapeI32(kvReshapeShape.begin(),
                                           kvReshapeShape.end());
    RankedTensorType kvReshapeTy = RankedTensorType::get(
        kvReshapeShape, kvReshapeElementType, kvReshapeEncoding);
    ReshapeOp kvReshapeOp = rewriter.create<ReshapeOp>(
        matrixMultOp.getLoc(), kvReshapeTy, matrixMultOp.getResult(),
        rewriter.getI32ArrayAttr(kvReshapeShapeI32));

    // Create split qkv op.
    // Determine if need to transpose key based on key and value.
    bool transposeKey = isKeyTransposed(keyShape, valueShape);
    auto splitOp = rewriter.create<SplitQueryKeyValueAndSplitHeadsOp>(
        matrixMultOp->getLoc(), ArrayRef<Type>{queryType, keyType, valueType},
        queryReshapeOp.getResult(), kvReshapeOp.getResult(),
        rewriter.getUI32IntegerAttr(numQueryHeads),
        rewriter.getUI32IntegerAttr(numKVHeads),
        rewriter.getBoolAttr(transposeKey) /*transpose_key*/);

    rewriter.replaceOp(permuteOps[0], splitOp.getQuery());
    rewriter.replaceOp(permuteOps[1], splitOp.getKey());
    rewriter.replaceOp(permuteOps[2], splitOp.getValue());

    return mlir::success();
  }

  mlir::LogicalResult fuseMHA(mlir::PatternRewriter &rewriter,
                              ReshapeOp reshapeOp,
                              llvm::SmallVector<MatMulOpType> &matmulOps,
                              llvm::SmallVector<PermuteOp> &permuteOps) const {
    // Extract dimensions from reshape op and permute ops.
    Value reshapeOutput = reshapeOp.getResult();
    ArrayRef<int64_t> inputShape = reshapeOp.getInput().getType().getShape();
    int32_t batchSize = inputShape[I_BATCH_SIZE];
    int32_t sequenceLength = inputShape[I_SEQUENCE_LENGTH];

    // Concatenate weights along dimension determined by transposeB attribute.

    TypedValue<RankedTensorType> queryWeightMatrix = matmulOps[0].getB();
    TypedValue<RankedTensorType> keyWeightMatrix = matmulOps[1].getB();
    TypedValue<RankedTensorType> valueWeightMatrix = matmulOps[2].getB();

    MatMulOpType queryMatmulOp = matmulOps[0];
    std::size_t dimToConcatWeights = queryMatmulOp.getTransposeB() ? 0 : 1;

    // Assert that queryMatmulOp A and B have rank 2.
    TT_assertv((queryMatmulOp.getA().getType().getRank() == 2 &&
                queryMatmulOp.getB().getType().getRank() == 2),
               "Expected rank 2 for MatMulOp operands A and B");

    ttir::ConcatOp concatenatedWeightMatrix = concatenateAlongDim(
        rewriter, valueWeightMatrix.getLoc(),
        ValueRange{queryWeightMatrix, keyWeightMatrix, valueWeightMatrix},
        dimToConcatWeights);

    Value concatenatedBias = nullptr;
    if constexpr (std::is_same_v<LinearOp, MatMulOpType>) {
      Value queryBias = matmulOps[0].getBias();
      Value keyBias = matmulOps[1].getBias();
      Value valueBias = matmulOps[2].getBias();
      std::size_t concatDim = matmulOps[0].getBias().getType().getRank() - 1;
      concatenatedBias = concatenateAlongDim(
          rewriter, valueBias.getLoc(),
          ValueRange{queryBias, keyBias, valueBias}, concatDim);
    }

    // Get the output shape from one of the original linear ops.
    ArrayRef<int64_t> originalLinearOutputShape =
        queryMatmulOp.getType().getShape();

    llvm::SmallVector<int64_t> linearOutputShape(
        originalLinearOutputShape.begin(), originalLinearOutputShape.end());

    // Multiply the last dimension by 3 (since we're concatenating Q, K, V).
    linearOutputShape.back() = originalLinearOutputShape.back() * 3;

    // Create linear operation with concatenated weights and bias.
    RankedTensorType linearOutputType = RankedTensorType::get(
        linearOutputShape, queryMatmulOp.getType().getElementType());

    SmallVector<Value> inputs;
    if (concatenatedBias) {
      inputs = {reshapeOutput, concatenatedWeightMatrix, concatenatedBias};
    } else {
      inputs = {reshapeOutput, concatenatedWeightMatrix};
    }

    MatMulOpType matrixMultOp = rewriter.create<MatMulOpType>(
        queryMatmulOp.getLoc(), TypeRange{linearOutputType}, inputs,
        queryMatmulOp->getAttrs());

    TT_assertv(matrixMultOp, "Expected valid matrix multiplication operation");

    // Create reshape operation to convert from [batch*seq, hidden*3] to [batch,
    // seq, hidden*3]. If it is the same output shape, reshape folder will fold
    // this operation.
    auto queryType = permuteOps[0].getType();
    auto keyType = permuteOps[1].getType();
    auto valueType = permuteOps[2].getType();

    auto keyShape = keyType.getShape();
    auto valueShape = valueType.getShape();
    int32_t numHeads = queryType.getShape()[O_NUM_HEADS];
    int32_t hiddenSize = queryType.getShape()[O_HEAD_SIZE] * numHeads;

    SmallVector<int64_t> reshapeToSplitShape = {batchSize, sequenceLength,
                                                hiddenSize * 3};
    auto reshapeElementType = queryMatmulOp.getType().getElementType();
    auto reshapeEncoding = queryMatmulOp.getType().getEncoding();

    SmallVector<int32_t> reshapeToSplitShapeI32(reshapeToSplitShape.begin(),
                                                reshapeToSplitShape.end());

    RankedTensorType reshapeTy = RankedTensorType::get(
        reshapeToSplitShape, reshapeElementType, reshapeEncoding);
    ReshapeOp reshapeToSplit = rewriter.create<ttir::ReshapeOp>(
        matrixMultOp.getLoc(), reshapeTy, matrixMultOp,
        rewriter.getI32ArrayAttr(reshapeToSplitShapeI32));

    // Determine if need to transpose key based on key and value.
    bool transposeKey = isKeyTransposed(keyShape, valueShape);

    auto splitOp = rewriter.create<SplitQueryKeyValueAndSplitHeadsOp>(
        matrixMultOp.getLoc(), ArrayRef<Type>{queryType, keyType, valueType},
        reshapeToSplit, Value(), rewriter.getUI32IntegerAttr(numHeads),
        IntegerAttr(), rewriter.getBoolAttr(transposeKey) /*transpose_key*/);

    rewriter.replaceOp(permuteOps[0], splitOp.getQuery());
    rewriter.replaceOp(permuteOps[1], splitOp.getKey());
    rewriter.replaceOp(permuteOps[2], splitOp.getValue());
    return mlir::success();
  }

  AttentionType
  findAttentionType(ReshapeOp reshapeOp,
                    llvm::SmallVector<MatMulOpType> &matmulOps,
                    llvm::SmallVector<PermuteOp> &permuteOps) const {
    populateMatmulOps(reshapeOp, matmulOps);
    if (!matmulOps.empty() && validateMatmulOrLinearOps(reshapeOp, matmulOps)) {
      populatePermuteOps(matmulOps, permuteOps);
      if (validatePermuteOps(reshapeOp, matmulOps, permuteOps)) {
        if (isMHA(matmulOps, permuteOps)) {
          return AttentionType::MHA;
        }
        if (isGQA(matmulOps, permuteOps)) {
          return AttentionType::GQA;
        }
      }
    }

    return AttentionType::NONE;
  }

  bool isMHA(llvm::SmallVector<MatMulOpType> matmulOps,
             llvm::SmallVector<PermuteOp> permuteOps) const {
    // This function checks if the pattern is Multi-Head Attention (MHA):
    // - Matmul/ Linear B shape must be equal.
    // - Linear op bias must be equal.
    // - Permute op number of kv_heads must be equal.

    auto queryBShape = matmulOps[0].getB().getType().getShape();
    auto keyBShape = matmulOps[1].getB().getType().getShape();
    auto valueBShape = matmulOps[2].getB().getType().getShape();
    if (queryBShape != keyBShape || queryBShape != valueBShape) {
      return false;
    }
    if constexpr (std::is_same_v<LinearOp, MatMulOpType>) {
      auto queryBias = matmulOps[0].getBias();
      auto keyBias = matmulOps[1].getBias();
      auto valueBias = matmulOps[2].getBias();
      if (queryBias.getType() != keyBias.getType() ||
          queryBias.getType() != valueBias.getType()) {
        return false;
      }
    }
    int32_t queryNumHeads = permuteOps[0].getType().getShape()[O_NUM_HEADS];
    int32_t keyNumHeads = permuteOps[1].getType().getShape()[O_NUM_HEADS];
    int32_t valueNumHeads = permuteOps[2].getType().getShape()[O_NUM_HEADS];
    if (queryNumHeads != keyNumHeads || queryNumHeads != valueNumHeads) {
      return false;
    }
    return true;
  }

  bool isGQA(llvm::SmallVector<MatMulOpType> matmulOps,
             llvm::SmallVector<PermuteOp> permuteOps) const {
    // This function checks if the pattern is Grouped Query Attention (GQA):
    // - Matmul / linear b shape outer dim for query should be divisible by
    //   key / value b shape outer dim and have the same multiplier.
    // - Permute op number of heads should reflect the multiplier.
    //  i.e. query_num_heads = key_num_heads * multiplier
    //      query_num_heads = value_num_heads * multiplier
    // TODO(@ddilbazTT): Extend to support bias checks for linear op.

    // If there is transpose B, check index 0, else check index 1.
    int32_t indexToCheck = matmulOps[0].getTransposeB() ? 0 : 1;

    auto queryBShape = matmulOps[0].getB().getType().getShape();
    auto keyBShape = matmulOps[1].getB().getType().getShape();
    auto valueBShape = matmulOps[2].getB().getType().getShape();
    if (queryBShape[indexToCheck] % keyBShape[indexToCheck] != 0 ||
        queryBShape[indexToCheck] % valueBShape[indexToCheck] != 0) {
      return false;
    }

    auto multiplier = queryBShape[indexToCheck] / keyBShape[indexToCheck];
    if (multiplier != queryBShape[indexToCheck] / valueBShape[indexToCheck]) {
      return false;
    }

    int32_t queryNumHeads = permuteOps[0].getType().getShape()[O_NUM_HEADS];
    int32_t keyNumHeads = permuteOps[1].getType().getShape()[O_NUM_HEADS];
    int32_t valueNumHeads = permuteOps[2].getType().getShape()[O_NUM_HEADS];

    if (queryNumHeads != keyNumHeads * multiplier ||
        queryNumHeads != valueNumHeads * multiplier) {
      return false;
    }
    return true;
  }

  // Helper function to concatenate tensors along given dimension.
  ttir::ConcatOp concatenateAlongDim(OpBuilder &rewriter, Location loc,
                                     ValueRange tensors,
                                     std::size_t dim) const {
    TT_assertv(!tensors.empty(), "Cannot concatenate empty tensor list.");

    auto firstType = mlir::cast<RankedTensorType>(tensors[0].getType());
    ArrayRef<int64_t> firstShape = firstType.getShape();

    // Calculate the concatenated size along the given dimension.
    int64_t concatDimSize = 0;
    for (Value tensor : tensors) {
      auto tensorType = mlir::cast<RankedTensorType>(tensor.getType());
      ArrayRef<int64_t> shape = tensorType.getShape();
      concatDimSize += shape[dim];
    }

    // Build the concatenated shape.
    SmallVector<int64_t> concatenatedShape(firstShape.begin(),
                                           firstShape.end());
    concatenatedShape[dim] = concatDimSize;

    RankedTensorType concatenatedType =
        RankedTensorType::get(concatenatedShape, firstType.getElementType());

    // Create concat op along given dimension.
    return rewriter.create<ttir::ConcatOp>(
        loc, concatenatedType, tensors,
        rewriter.getSI32IntegerAttr(dim) /* axis */);
  }

  void hoistPreprocessingOps(llvm::SmallVector<PermuteOp> permuteOps) const {
    // Find the topologically first PermuteOp
    PermuteOp firstPermuteOp = permuteOps[0];
    for (size_t i = 1; i < permuteOps.size(); ++i) {
      if (permuteOps[i]->isBeforeInBlock(firstPermuteOp)) {
        firstPermuteOp = permuteOps[i];
      }
    }

    // Collect all values that need to be hoisted (from all non-first operands).
    llvm::SetVector<mlir::Value> allValues;

    // Collect use-def chains for all PermuteOps except the first one
    for (size_t i = 0; i < permuteOps.size(); ++i) {
      if (permuteOps[i] == firstPermuteOp) {
        continue; // Skip the topologically first one
      }

      for (Value operand : permuteOps[i]->getOperands()) {
        llvm::SetVector<mlir::Value> udChain =
            ttmlir::utils::getUseDefChain(operand);
        allValues.insert(udChain.begin(), udChain.end());
      }
    }

    // Filter to get operations.
    llvm::SetVector<mlir::Operation *> opsToMove =
        ttmlir::utils::filterOperations(allValues.getArrayRef());

    // Topologically sort to maintain dependencies.
    llvm::SetVector<mlir::Operation *> sortedOps = topologicalSort(opsToMove);

    // Move only operations that appear after firstPermuteOp.
    for (mlir::Operation *op : sortedOps) {
      TT_assertv(op->getBlock() == firstPermuteOp->getBlock(),
                 "Expected all ops to be in the same block.");
      if (!op->isBeforeInBlock(firstPermuteOp)) {
        op->moveBefore(firstPermuteOp);
      }
    }
  }

  PermuteOp getPermuteOp(MatMulOpType op) const {
    // Permute op comes from: linear/ matmul op -> reshape op -> permute op.
    auto users = op->getResult(0).getUsers();
    if (llvm::range_size(users) != 1) {
      return nullptr;
    }
    ReshapeOp reshapeOp = mlir::dyn_cast<ReshapeOp>(*users.begin());
    if (!reshapeOp) {
      return nullptr;
    }
    auto reshapeUsers = reshapeOp.getResult().getUsers();
    if (llvm::range_size(reshapeUsers) != 1) {
      return nullptr;
    }
    PermuteOp permuteOp = mlir::dyn_cast<PermuteOp>(*reshapeUsers.begin());
    if (!permuteOp) {
      return nullptr;
    }
    return permuteOp;
  }

  void populatePermuteOps(llvm::SmallVector<MatMulOpType> matrixOps,
                          llvm::SmallVector<PermuteOp> &permuteOps) const {
    for (MatMulOpType matrixOp : matrixOps) {
      if (PermuteOp permuteOp = getPermuteOp(matrixOp); permuteOp) {
        permuteOps.push_back(permuteOp);
      }
    }
    // Sort to ensure Q, K, V order.
    std::sort(permuteOps.begin(), permuteOps.end(),
              [](mlir::Operation *a, mlir::Operation *b) {
                return a->isBeforeInBlock(b);
              });
  }

  void populateMatmulOps(ReshapeOp reshapeOp,
                         llvm::SmallVector<MatMulOpType> &matmulOps) const {
    for (auto *user : reshapeOp.getResult().getUsers()) {
      if (MatMulOpType matmulOp = mlir::dyn_cast<MatMulOpType>(user)) {
        matmulOps.push_back(matmulOp);
      }
    }
    // Sort to ensure Q, K, V order.
    std::sort(matmulOps.begin(), matmulOps.end(),
              [](mlir::Operation *a, mlir::Operation *b) {
                return a->isBeforeInBlock(b);
              });
  }

  bool isKeyTransposed(ArrayRef<int64_t> keyShape,
                       ArrayRef<int64_t> queryShape) const {
    return (keyShape[K_HEAD_SIZE] == queryShape[O_HEAD_SIZE] &&
            keyShape[K_SEQUENCE_LENGTH] == queryShape[O_SEQUENCE_LENGTH]);
  }

  std::optional<std::array<size_t, 3>>
  returnQKVIndices(llvm::SmallVector<MatMulOpType> &matrixOps,
                   llvm::SmallVector<PermuteOp> &permuteOps) const {
    // Return indices of Q, K, V in permuteOps.
    // If not found, return std::nullopt.
    if (permuteOps.size() != 3) {
      return std::nullopt;
    }

    size_t queryIndex = 0;
    size_t keyIndex = 1;
    size_t valueIndex = 2;

    bool foundQuery = false;
    bool foundKey = false;

    // Query has the greatest number of heads in GQA, same for MHA.
    int32_t maxHeads = -1;
    for (size_t i = 0; i < 3; ++i) {
      // Each of Q, K, V heads must be 4D.
      if (permuteOps[i].getType().getShape().size() != 4) {
        return std::nullopt;
      }
      int32_t numHeads = permuteOps[i].getType().getShape()[O_NUM_HEADS];
      if (numHeads > maxHeads) {
        maxHeads = numHeads;
        queryIndex = i;
        foundQuery = true;
      }
    }

    for (size_t i = 0; i < 3; ++i) {
      // If permutation attribute is equal to expectedTransposedPermute, that is
      // Key. Key can still have expectedPermute.
      llvm::ArrayRef<int64_t> permutation = permuteOps[i].getPermutation();
      if (llvm::equal(permutation, expectedTransposedPermute)) {
        keyIndex = i;
        foundKey = true;
      }
    }

    // Query and Key indices could or could not be found. We should still be
    // able to order.
    if (foundQuery && foundKey && queryIndex == keyIndex) {
      return std::nullopt;
    }

    if (foundQuery && foundKey) {
      // Both Query and Key indices found.
      // Assign remaining index to Value.
      for (size_t i = 0; i < 3; ++i) {
        if (i != keyIndex && i != queryIndex) {
          valueIndex = i;
          break;
        }
      }
    } else if (foundQuery && !foundKey) {
      // Query Index found, Key index not found.
      // Assign remaining indices to Key and Value, respectively.
      bool assignedKey = false;
      for (size_t i = 0; i < 3; ++i) {
        if (i != queryIndex) {
          if (!assignedKey) {
            keyIndex = i;
            assignedKey = true;
          } else {
            valueIndex = i;
          }
        }
      }
    } else if (foundKey && !foundQuery) {
      // Key Index found, Query index not found.
      // Assign remaining indices to Query and Value, respectively.
      bool assignedQuery = false;
      for (size_t i = 0; i < 3; ++i) {
        if (i != keyIndex) {
          if (!assignedQuery) {
            queryIndex = i;
            assignedQuery = true;
          } else {
            valueIndex = i;
          }
        }
      }
    }
    // else: neither found, use defaults 0, 1, 2

    // Check that key and value B shape are equal.
    auto keyBShape = matrixOps[keyIndex].getB().getType().getShape();
    auto valueBShape = matrixOps[valueIndex].getB().getType().getShape();
    if (keyBShape != valueBShape) {
      return std::nullopt;
    }

    // If matrix op is linear, check that key and value bias are equal shape.
    if constexpr (std::is_same_v<LinearOp, MatMulOpType>) {
      auto keyBias = matrixOps[keyIndex].getBias();
      auto valueBias = matrixOps[valueIndex].getBias();
      auto keyBiasShape = keyBias.getType().getShape();
      auto valueBiasShape = valueBias.getType().getShape();
      if (keyBiasShape != valueBiasShape) {
        return std::nullopt;
      }
    }

    return std::array<size_t, 3>{queryIndex, keyIndex, valueIndex};
  }

  void reorderToQKV(llvm::SmallVector<MatMulOpType> &matmulOps,
                    llvm::SmallVector<PermuteOp> &permuteOps,
                    const std::array<size_t, 3> &indices) const {

    auto [qIdx, kIdx, vIdx] = indices;

    // Create reordered copies
    llvm::SmallVector<MatMulOpType> orderedMatmuls = {
        matmulOps[qIdx], matmulOps[kIdx], matmulOps[vIdx]};
    llvm::SmallVector<PermuteOp> orderedPermutes = {
        permuteOps[qIdx], permuteOps[kIdx], permuteOps[vIdx]};

    // Replace original vectors
    matmulOps = std::move(orderedMatmuls);
    permuteOps = std::move(orderedPermutes);
  }

  bool validatePermuteOps(ReshapeOp reshapeOp,
                          llvm::SmallVector<MatMulOpType> &matrixOps,
                          llvm::SmallVector<PermuteOp> &permuteOps) const {

    // Output of permute ops are the query, key, value heads respectively.
    // There should be exactly 3 permute ops.
    // Permute op output shapes must match expected MHA head shapes:
    // [batch_size, number of kv heads, sequence_length, head_size].
    // If key is transposed, its shape must match expected transposed shape:
    // [batch_size, number of kv heads, head_size, sequence_length].
    // Expected permutation for query and value is [0, 2, 1, 3].
    // Expected permutation for key is either [0, 2, 1, 3] or [0, 2, 3, 1]
    // (transposed).

    auto indices = returnQKVIndices(matrixOps, permuteOps);
    if (!indices) {
      return false;
    }
    auto [qIdx, kIdx, vIdx] = *indices;

    TypedValue<RankedTensorType> queryTensor = permuteOps[qIdx].getResult();
    TypedValue<RankedTensorType> keyTensor = permuteOps[kIdx].getResult();
    TypedValue<RankedTensorType> valueTensor = permuteOps[vIdx].getResult();
    ArrayRef<int64_t> queryShape = queryTensor.getType().getShape();
    ArrayRef<int64_t> keyShape = keyTensor.getType().getShape();
    ArrayRef<int64_t> valueShape = valueTensor.getType().getShape();

    if (queryShape[O_BATCH_SIZE] != keyShape[O_BATCH_SIZE] ||
        queryShape[O_BATCH_SIZE] != valueShape[O_BATCH_SIZE]) {
      // Batch size must match.
      return false;
    }

    // Key and Value must have the same number of kv heads.
    // Query can have a different number of heads in GQA.
    if (keyShape[O_NUM_HEADS] != valueShape[O_NUM_HEADS]) {
      return false;
    }

    if (queryShape[O_SEQUENCE_LENGTH] != valueShape[O_SEQUENCE_LENGTH]) {
      // Query and value sequence length must match.
      return false;
    }
    if (queryShape[O_HEAD_SIZE] != valueShape[O_HEAD_SIZE]) {
      // Query and value head size must match.
      return false;
    }

    // Query and value must have the same expected permutation.
    llvm::ArrayRef<int64_t> queryPermutation = permuteOps[0].getPermutation();
    llvm::ArrayRef<int64_t> valuePermutation = permuteOps[2].getPermutation();
    if (!llvm::equal(queryPermutation, expectedPermute) ||
        !llvm::equal(valuePermutation, expectedPermute)) {
      return false;
    }

    // Key can be transposed or not.
    bool transposeKey = isKeyTransposed(keyShape, queryShape);
    llvm::ArrayRef<int64_t> keyPermutation = permuteOps[1].getPermutation();

    if (!transposeKey) {
      // If key is not transposed, it should have the expected output shape.
      if (keyShape[O_SEQUENCE_LENGTH] != queryShape[O_SEQUENCE_LENGTH] ||
          keyShape[O_HEAD_SIZE] != queryShape[O_HEAD_SIZE]) {
        return false;
      }
      if (!llvm::equal(keyPermutation, expectedPermute)) {
        return false;
      }
    }

    if (transposeKey) {
      // If key is transposed, it should have the expected transposed shape.
      if (keyShape[K_HEAD_SIZE] != queryShape[O_HEAD_SIZE] ||
          keyShape[K_SEQUENCE_LENGTH] != queryShape[O_SEQUENCE_LENGTH]) {
        return false;
      }
      if (!llvm::equal(keyPermutation, expectedTransposedPermute)) {
        return false;
      }
    }
    // Order matrixOps and permuteOps to Q, K, V.
    reorderToQKV(matrixOps, permuteOps, *indices);
    return true;
  }

  bool
  validateMatmulOrLinearOps(ReshapeOp reshapeOp,
                            llvm::SmallVector<MatMulOpType> matrixOps) const {
    // There must be exactly 3 matmul/linear ops (for Q, K, V).
    if (matrixOps.size() != 3) {
      return false;
    }

    ArrayRef<int64_t> reshapeInputShape =
        reshapeOp.getInput().getType().getShape();
    // If the input shape is not 3D, we cannot fuse. [BATCH_SIZE,
    // SEQUENCE_LENGTH, INPUT_HIDDEN_DIMENSION].
    if (reshapeInputShape.size() != 3) {
      return false;
    }

    // Reshape transforms [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_HIDDEN_DIMENSION]
    // to [BATCH_SIZE * SEQUENCE_LENGTH, INPUT_HIDDEN_DIMENSION].
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();
    if (reshapeOutputShape.size() != 2) {
      return false;
    }
    if (reshapeOutputShape[0] != reshapeInputShape[I_BATCH_SIZE] *
                                     reshapeInputShape[I_SEQUENCE_LENGTH] ||
        reshapeOutputShape[1] != reshapeInputShape[I_HIDDEN_DIMENSION]) {
      return false;
    }

    for (MatMulOpType matrixOp : matrixOps) {
      // Each op must have the same input as the reshape op output.
      if (matrixOp.getA() != reshapeOp.getResult()) {
        return false;
      }
      auto bShape = matrixOp.getB().getType().getShape();
      // Each op must have a 2D weight matrix.
      if (bShape.size() != 2) {
        return false;
      }
    }

    // Check that B shape is the same for at least two of the ops.
    auto firstBShape = matrixOps[0].getB().getType().getShape();
    auto secondBShape = matrixOps[1].getB().getType().getShape();
    auto thirdBShape = matrixOps[2].getB().getType().getShape();
    bool atLeastTwoBMatch = (firstBShape == secondBShape) ||
                            (firstBShape == thirdBShape) ||
                            (secondBShape == thirdBShape);
    if (!atLeastTwoBMatch) {
      return false;
    }

    // Check transpose A is false for all ops.
    if (matrixOps[0].getTransposeA() || matrixOps[1].getTransposeA() ||
        matrixOps[2].getTransposeA()) {
      return false;
    }

    // Check transpose B is the same for all ops.
    if ((matrixOps[0].getTransposeB() != matrixOps[1].getTransposeB()) ||
        (matrixOps[0].getTransposeB() != matrixOps[2].getTransposeB())) {
      return false;
    }

    // If matrix op is Linear Op, check that bias is present.
    // At least two matrix ops must have the same bias shape.
    if constexpr (std::is_same_v<LinearOp, MatMulOpType>) {
      TypedValue<RankedTensorType> firstBias = matrixOps[0].getBias();
      TypedValue<RankedTensorType> secondBias = matrixOps[1].getBias();
      TypedValue<RankedTensorType> thirdBias = matrixOps[2].getBias();
      if (!firstBias || !secondBias || !thirdBias) {
        return false;
      }
      auto firstBiasShape = firstBias.getType().getShape();
      auto secondBiasShape = secondBias.getType().getShape();
      auto thirdBiasShape = thirdBias.getType().getShape();
      // Check if at least two bias shapes match.
      bool atLeastTwoBiasMatch = (firstBiasShape == secondBiasShape) ||
                                 (firstBiasShape == thirdBiasShape) ||
                                 (secondBiasShape == thirdBiasShape);

      if (!atLeastTwoBiasMatch) {
        return false;
      }
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
      rewriter.replaceOpWithNewOp<ttir::GeluOp>(op, op.getType(), geluInput);

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
        erf.getInput().getDefiningOp<ttir::MultiplyOp>();
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

namespace {

// Fuses: (x * rsqrt(mean(x^2) + epsilon)) * gamma -> RMSNormOp
class RMSNormFusionPattern : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp outerMul,
                  mlir::PatternRewriter &rewriter) const final {
    MultiplyOp innerMul =
        utils::findOpThrough<MultiplyOp, TypecastOp>(outerMul.getLhs());
    mlir::Value gammaRaw = outerMul.getRhs();
    if (!innerMul) {
      innerMul =
          utils::findOpThrough<MultiplyOp, TypecastOp>(outerMul.getRhs());
      gammaRaw = outerMul.getLhs();
    }
    if (!innerMul) {
      return mlir::failure();
    }

    RsqrtOp rsqrtOp = utils::findOpThroughLayoutOps<RsqrtOp>(innerMul.getLhs());
    mlir::Value xRaw = innerMul.getRhs();
    if (!rsqrtOp) {
      rsqrtOp = utils::findOpThroughLayoutOps<RsqrtOp>(innerMul.getRhs());
      xRaw = innerMul.getLhs();
    }
    if (!rsqrtOp) {
      return mlir::failure();
    }

    auto addOp = utils::findOpThroughLayoutOps<AddOp>(rsqrtOp.getInput());
    if (!addOp) {
      return mlir::failure();
    }
    MeanOp meanOp = addOp.getLhs().getDefiningOp<MeanOp>();
    mlir::Value epsilon = addOp.getRhs();
    if (!meanOp) {
      meanOp = addOp.getRhs().getDefiningOp<MeanOp>();
      epsilon = addOp.getLhs();
    }
    if (!meanOp) {
      return mlir::failure();
    }

    // TTNN RMS norm only supports normalization over the last dimension.
    auto dimArg = meanOp.getDimArg();
    if (!dimArg || dimArg->size() != 1) {
      return mlir::failure();
    }
    auto meanInputType =
        mlir::cast<RankedTensorType>(meanOp.getInput().getType());
    int64_t dim = mlir::cast<mlir::IntegerAttr>((*dimArg)[0]).getInt();
    int64_t actualDim = dim < 0 ? meanInputType.getRank() + dim : dim;
    if (actualDim != meanInputType.getRank() - 1) {
      return mlir::failure();
    }

    // Match x^2 as mul(x,x) or pow(x,2).
    // Look through layout ops to find the original input.
    mlir::Value meanInput = meanOp.getInput();
    mlir::Value squareInput = nullptr;
    mlir::Value x = utils::lookThroughLayoutOps(xRaw);

    if (auto sq = meanInput.getDefiningOp<MultiplyOp>()) {
      if (sq.getLhs() == sq.getRhs()) {
        squareInput = utils::lookThroughLayoutOps(sq.getLhs());
      }
    } else if (auto pw = meanInput.getDefiningOp<PowOp>()) {
      if (isFullOpWithValue(pw.getRhs(), 2.0f)) {
        squareInput = utils::lookThroughLayoutOps(pw.getLhs());
      }
    }

    if (!squareInput || squareInput != x) {
      return mlir::failure();
    }

    // Look through layout ops to find the epsilon FullOp
    auto epsFull = utils::findOpThroughLayoutOps<FullOp>(epsilon);
    if (!epsFull) {
      return mlir::failure();
    }
    auto epsAttr = mlir::dyn_cast<FloatAttr>(epsFull.getFillValue());
    if (!epsAttr) {
      return mlir::failure();
    }

    mlir::Value gamma = utils::lookThroughLayoutOps(gammaRaw);
    auto inputType = mlir::cast<RankedTensorType>(x.getType());
    auto outputType = mlir::cast<RankedTensorType>(outerMul.getType());

    llvm::SmallVector<int64_t> normalizedShape{inputType.getShape().back()};

    // Reshape gamma to match normalized_shape if needed.
    // Gamma may have extra dimensions (e.g., [1, 1, 2048] instead of [2048]).
    auto gammaType = mlir::cast<RankedTensorType>(gamma.getType());
    if (gammaType.getShape() != llvm::ArrayRef(normalizedShape)) {
      // Check if gamma can be squeezed to normalized_shape.
      // All dimensions except the last must be 1.
      for (int64_t i = 0; i < gammaType.getRank() - 1; ++i) {
        if (gammaType.getShape()[i] != 1) {
          return mlir::failure();
        }
      }
      // Check if the last dimension matches.
      if (gammaType.getShape().back() != normalizedShape[0]) {
        return mlir::failure();
      }
      // Reshape gamma to normalized_shape.
      auto reshapedGammaType = RankedTensorType::get(
          normalizedShape, gammaType.getElementType(), gammaType.getEncoding());
      llvm::SmallVector<int32_t> targetShape(normalizedShape.begin(),
                                             normalizedShape.end());
      gamma = rewriter.create<ReshapeOp>(outerMul.getLoc(), reshapedGammaType,
                                         gamma,
                                         rewriter.getI32ArrayAttr(targetShape));
    }

    // Create RMSNormOp with output shape and dtype matching input
    auto rmsNormOutputType =
        RankedTensorType::get(inputType.getShape(), inputType.getElementType(),
                              inputType.getEncoding());
    auto rmsNorm = rewriter.create<RMSNormOp>(
        outerMul.getLoc(), rmsNormOutputType, x, gamma,
        /*bias=*/nullptr, rewriter.getDenseI64ArrayAttr(normalizedShape),
        rewriter.getF32FloatAttr(epsAttr.getValue().convertToFloat()));

    mlir::Value result = rmsNorm;

    // If output shape differs from input shape, add a reshape
    if (inputType.getShape() != outputType.getShape()) {
      llvm::SmallVector<int32_t> targetShape(outputType.getShape());
      auto reshapeOutputType = RankedTensorType::get(outputType.getShape(),
                                                     inputType.getElementType(),
                                                     inputType.getEncoding());
      result = rewriter.create<ReshapeOp>(
          outerMul.getLoc(), reshapeOutputType, result,
          rewriter.getI32ArrayAttr(targetShape));
    }

    // If output dtype differs from input dtype, add a typecast
    if (inputType.getElementType() != outputType.getElementType()) {
      result =
          rewriter.create<TypecastOp>(outerMul.getLoc(), outputType, result);
    }

    rewriter.replaceOp(outerMul, result);
    return mlir::success();
  }
};

// If value is defined by PermuteOp with permute dimensions
// (..., rank - 2, rank - 1), return the input of the PermuteOp, otherwise
// return std::nullopt. This is used for fusing permute into MatmulOp and
// LinearOp.
static std::optional<mlir::TypedValue<mlir::RankedTensorType>>
getPermuteOpOperand(mlir::TypedValue<mlir::RankedTensorType> value) {
  auto producerPermuteOp = value.getDefiningOp<PermuteOp>();
  if (!producerPermuteOp) {
    return std::nullopt;
  }

  int64_t rank = value.getType().getRank();
  // If the rank is less than two than it is impossible for this permute to be
  // a transpose
  bool rankIsLessThan2 = rank < 2;
  // Ensure that the rightmost two dims are swapped by the permute
  bool xyDimsTransposed =
      producerPermuteOp.getPermutation()[rank - 2] == rank - 1 &&
      producerPermuteOp.getPermutation()[rank - 1] == rank - 2;
  // Ensure that the other dims are unchanged by the permute.
  // Therefore this permute is equivalent to transpose(-1, -2)
  bool otherDimsUnchanged =
      std::is_sorted(producerPermuteOp.getPermutation().begin(),
                     producerPermuteOp.getPermutation().end() - 2);
  if (rankIsLessThan2 || !xyDimsTransposed || !otherDimsUnchanged) {
    return std::nullopt;
  }

  return producerPermuteOp.getInput();
}

// Fuse permute ops into matmul/linear transpose attributes.
// Handles both input A and input B for MatmulOp and LinearOp.
//
// matmul(transpose(a), b, transpose_a, transpose_b) ->
//   matmul(a, b, !transpose_a, transpose_b)
// matmul(a, transpose(b), transpose_a, transpose_b) ->
//   matmul(a, b, transpose_a, !transpose_b)
// linear(transpose(a), b, bias, transpose_a, transpose_b) ->
//   linear(a, b, bias, !transpose_a, transpose_b)
// linear(a, transpose(b), bias, transpose_a, transpose_b) ->
//   linear(a, b, bias, transpose_a, !transpose_b)
template <typename OpType>
class PermuteMatmulFusionPattern : public mlir::OpRewritePattern<OpType> {
  using mlir::OpRewritePattern<OpType>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(OpType op, mlir::PatternRewriter &rewriter) const override {
    auto inputACanonical = getPermuteOpOperand(op.getA());
    auto inputBCanonical = getPermuteOpOperand(op.getB());

    if (!inputACanonical && !inputBCanonical) {
      return mlir::failure();
    }

    auto newA = inputACanonical.value_or(op.getA());
    auto newB = inputBCanonical.value_or(op.getB());
    bool newTransposeA =
        inputACanonical ? !op.getTransposeA() : op.getTransposeA();
    bool newTransposeB =
        inputBCanonical ? !op.getTransposeB() : op.getTransposeB();

    if constexpr (std::is_same_v<OpType, MatmulOp>) {
      rewriter.replaceOpWithNewOp<MatmulOp>(op, op.getType(), newA, newB,
                                            newTransposeA, newTransposeB);
    } else if constexpr (std::is_same_v<OpType, LinearOp>) {
      rewriter.replaceOpWithNewOp<LinearOp>(op, op.getType(), newA, newB,
                                            op.getBias(), newTransposeA,
                                            newTransposeB);
    } else {
      static_assert(ttmlir::utils::always_false<OpType>(),
                    "Unsupported OpType for PermuteMatmulFusionPattern");
    }

    return mlir::success();
  }
};

// Pattern to fuse reshape -> broadcast -> reshape into either:
// - repeat_interleave (when final reshape multiplies the dim left of inserted
// 1)
// - repeat (when final reshape multiplies the dim right of inserted 1)
//
// This pattern is commonly used for GQA (Grouped Query Attention) to expand KV
// heads to match Q heads.
class ReshapeBroadcastReshapeToRepeatPattern
    : public mlir::OpRewritePattern<ReshapeOp> {
  using mlir::OpRewritePattern<ReshapeOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp finalReshape,
                  mlir::PatternRewriter &rewriter) const final {
    // Match: reshape -> broadcast -> reshape.
    auto broadcastOp = finalReshape.getInput().getDefiningOp<BroadcastOp>();
    if (!broadcastOp || !broadcastOp->hasOneUse()) {
      return mlir::failure();
    }

    auto firstReshape = broadcastOp.getInput().getDefiningOp<ReshapeOp>();
    if (!firstReshape || !firstReshape->hasOneUse()) {
      return mlir::failure();
    }

    auto inputType =
        mlir::cast<RankedTensorType>(firstReshape.getInput().getType());
    auto intermediateType =
        mlir::cast<RankedTensorType>(firstReshape.getResult().getType());
    auto outputType =
        mlir::cast<RankedTensorType>(finalReshape.getResult().getType());

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> intermediateShape = intermediateType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> broadcastDims = broadcastOp.getBroadcastDimensions();

    if (intermediateShape.size() != inputShape.size() + 1) {
      return mlir::failure();
    }
    if (broadcastDims.size() != intermediateShape.size()) {
      return mlir::failure();
    }
    if (outputShape.size() != inputShape.size()) {
      return mlir::failure();
    }

    // Use broadcast evidence to pick the inserted dim. This avoids ambiguity
    // when the input shape already contains size-1 dims. This rewrite only
    // models a single expanded dimension.
    int64_t insertedDim = -1;
    int64_t repeatCount = 1;
    for (size_t i = 0; i < broadcastDims.size(); ++i) {
      if (broadcastDims[i] == 1) {
        continue;
      }
      if (insertedDim != -1) {
        return mlir::failure();
      }
      insertedDim = static_cast<int64_t>(i);
      repeatCount = broadcastDims[i];
    }
    if (repeatCount <= 1) {
      return mlir::failure();
    }

    // Validate that `intermediateShape` is exactly `inputShape` with a single
    // size-1 dimension inserted at `insertedDim`.
    if (insertedDim < 0 ||
        static_cast<size_t>(insertedDim) >= intermediateShape.size()) {
      return mlir::failure();
    }
    if (intermediateShape[insertedDim] != 1) {
      return mlir::failure();
    }
    for (size_t i = 0; i < inputShape.size(); ++i) {
      size_t intermediateIdx = i < static_cast<size_t>(insertedDim) ? i : i + 1;
      if (intermediateShape[intermediateIdx] != inputShape[i]) {
        return mlir::failure();
      }
    }

    // Find which input dimension was multiplied by repeatCount in the output.
    int64_t changedDim = -1;
    for (size_t i = 0; i < outputShape.size(); ++i) {
      if (outputShape[i] == inputShape[i]) {
        continue;
      }
      if (changedDim != -1) {
        return mlir::failure();
      }
      changedDim = static_cast<int64_t>(i);
    }
    if (changedDim == -1) {
      return mlir::failure();
    }
    if (outputShape[changedDim] != inputShape[changedDim] * repeatCount) {
      return mlir::failure();
    }

    // Decide between repeat_interleave (left-merge) and repeat (right-merge).
    if (changedDim == insertedDim - 1) {
      rewriter.replaceOpWithNewOp<RepeatInterleaveOp>(
          finalReshape, outputType, firstReshape.getInput(),
          static_cast<uint32_t>(repeatCount), static_cast<int32_t>(changedDim));
      return mlir::success();
    }

    if (changedDim == insertedDim) {
      llvm::SmallVector<int64_t> repeatDims(inputShape.size(), 1);
      repeatDims[changedDim] = repeatCount;
      rewriter.replaceOpWithNewOp<RepeatOp>(
          finalReshape, outputType, firstReshape.getInput(),
          rewriter.getDenseI64ArrayAttr(repeatDims));
      return mlir::success();
    }

    return mlir::failure();
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
      patterns.add<ConvTagWeights<Conv3dOp>>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<ConvAddBias<Conv2dOp>>(&getContext());
      patterns.add<ConvAddBias<ConvTranspose2dOp>>(&getContext());
      patterns.add<ConvAddBias<Conv3dOp>>(&getContext());

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
        patterns.add<ConvWithMultiply<ConvTranspose2dOp>>(&getContext());
        patterns.add<BatchNormDecomposition>(&getContext());
      }
      if (permuteMatmulEnabled) {
        patterns.add<PermuteMatmulFusionPattern<MatmulOp>>(&getContext());
        patterns.add<PermuteMatmulFusionPattern<LinearOp>>(&getContext());
      }
      patterns.add<RepVGGConvSumFusionPattern>(&getContext());
      patterns.add<ConcatenateHeadsUpdatePattern>(&getContext());
      patterns.add<SplitQueryKeyValueAndSplitHeadsUpdatePattern<MatmulOp>>(
          &getContext());
      patterns.add<SplitQueryKeyValueAndSplitHeadsUpdatePattern<LinearOp>>(
          &getContext());
      patterns.add<ScaledSumToMeanPattern>(&getContext());
      patterns.add<SpatialMeanOptimizationPattern>(&getContext());
      patterns.add<MatmulWithBiasFusionPattern>(&getContext());
      patterns.add<RMSNormFusionPattern>(&getContext());

      patterns.add<GeluFusionPattern>(&getContext());
      patterns.add<Relu6FusionPattern>(&getContext());
      patterns.add<SiluFusionPattern>(&getContext());
      patterns.add<HardsigmoidFusionPattern>(&getContext());
      patterns.add<MishFusingPattern>(&getContext());
      patterns.add<ReshapeBroadcastReshapeToRepeatPattern>(&getContext());

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir
