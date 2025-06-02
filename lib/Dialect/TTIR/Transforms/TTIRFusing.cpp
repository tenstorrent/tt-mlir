// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

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

mlir::LogicalResult matchFloatAttr(FloatAttr attr, float constant) {
  float constValue = attr.getValue().convertToFloat();
  if (constValue != constant) {
    return failure();
  }

  return success();
}

mlir::LogicalResult matchConstantValue(FullOp full, float constant) {
  FloatAttr valueAttr = mlir::dyn_cast_or_null<FloatAttr>(full.getFillValue());
  if (!valueAttr) {
    return failure();
  }

  return matchFloatAttr(valueAttr, constant);
}

// Matches: (((ax + coefficients[-1])x + coefficients[-2])x + ...)
mlir::LogicalResult matchNestedPolynomial(AddOp polynomial, float a,
                                          SmallVector<float> coefficients,
                                          Value &polynomialArg) {
  AddOp current = polynomial;
  MultiplyOp finalMul = nullptr;
  for (uint32_t i = 0; i < coefficients.size(); i++) {
    float coefficient = coefficients[i];
    if (!current) {
      return failure();
    }

    // Rhs of add must be a constant with a single value matching
    // `coefficient`
    FullOp constCoefficient =
        mlir::dyn_cast_or_null<FullOp>(current.getRhs().getDefiningOp());
    if (!constCoefficient) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(constCoefficient, coefficient))) {
      return failure();
    }

    MultiplyOp multiply =
        mlir::dyn_cast_or_null<MultiplyOp>(current.getLhs().getDefiningOp());
    if (!multiply) {
      return failure();
    }

    if (!polynomialArg) {
      polynomialArg = multiply.getRhs();
    } else if (polynomialArg != multiply.getRhs()) {
      return failure();
    }

    current = mlir::dyn_cast_or_null<AddOp>(multiply.getLhs().getDefiningOp());
    finalMul = multiply;
  }

  // Match the final "ax" part of the nested polynomial
  FullOp constA =
      mlir::dyn_cast_or_null<FullOp>(finalMul.getLhs().getDefiningOp());
  if (!constA) {
    return failure();
  }

  if (mlir::failed(matchConstantValue(constA, a))) {
    return failure();
  }

  return success();
}

class ErfApproximationFusionPattern
    : public mlir::OpRewritePattern<ClampScalarOp> {
public:
  ErfApproximationFusionPattern(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  mlir::LogicalResult
  matchAndRewrite(ClampScalarOp clamp,
                  mlir::PatternRewriter &rewriter) const final {
    // This RewritePattern will fuse the following approximation of `erf` to
    // ttir::ErfOp
    //
    // x = clamp(x, -4.0, 4.0)
    //         x * Nested polynomial (x^2)
    // x = ------------------------------------
    //       Different nested polynomial(x^2)
    // x = clamp(x, -1.0 , 1.0)
    //
    // A nested polynomial is one which takes the form
    // f(x) = (((((((ax + b)x + c)x + d)x + ...))))

    // Check if the clamp is between -1.0 and 1.0
    if (mlir::failed(matchFloatAttr(clamp.getMinAttr(), -1.0)) ||
        mlir::failed(matchFloatAttr(clamp.getMaxAttr(), 1.0))) {
      return failure();
    }

    DivOp div = dyn_cast_or_null<DivOp>(clamp.getInput().getDefiningOp());
    if (!div) {
      return failure();
    }

    // The numerator is a nested polynomial which is multiplied by the
    // clamped erf input: x * Nested polynomial (x^2)
    MultiplyOp numerator =
        mlir::dyn_cast_or_null<MultiplyOp>(div.getLhs().getDefiningOp());
    if (!numerator) {
      return failure();
    }

    // The numerator polynomial is multiplied by the clamped erf input on the
    // left hand side.
    ClampScalarOp clampedErfInput =
        dyn_cast_or_null<ClampScalarOp>(numerator.getLhs().getDefiningOp());
    if (!clampedErfInput) {
      return failure();
    }

    // Check if the clamp is between -4.0 and 4.0
    if (mlir::failed(matchFloatAttr(clampedErfInput.getMinAttr(), -4.0)) ||
        mlir::failed(matchFloatAttr(clampedErfInput.getMaxAttr(), 4.0))) {
      return failure();
    }

    AddOp numeratorPolynomial =
        mlir::dyn_cast_or_null<AddOp>(numerator.getRhs().getDefiningOp());
    if (!numeratorPolynomial) {
      return failure();
    }

    Value polynomialArg = nullptr;
    // The numerator is a nested polynomial with constants (in order from
    // closest to the div to furthest):
    SmallVector<float> numeratorPolynomialCoefficients = {
        -1.60960333262415e-02f, -2.95459980854025e-03f, -7.34990630326855e-04f,
        -5.69250639462346e-05f, -2.10102402082508e-06f, 2.77068142495902e-08f};

    float numeratorA = -2.72614225801306e-10f;

    if (mlir::failed(matchNestedPolynomial(numeratorPolynomial, numeratorA,
                                           numeratorPolynomialCoefficients,
                                           polynomialArg))) {
      return failure();
    }

    AddOp denominator =
        mlir::dyn_cast_or_null<AddOp>(div.getRhs().getDefiningOp());
    if (!denominator) {
      return failure();
    }

    // The denominator is a the input sacaled followed by nested polynomial
    // with constants (in order from closest to the div to furthest):
    SmallVector<float> denominatorPolynomialCoefficients = {
        -1.42647390514189e-02f, -7.37332916720468e-03f, -1.68282697438203e-03f,
        -2.13374055278905e-04f};
    float denominatorA = -1.45660718464996e-05f;

    // We pass the same polynomialArg to the denominator because we expect it to
    // be the same as the numerators.
    if (mlir::failed(matchNestedPolynomial(denominator, denominatorA,
                                           denominatorPolynomialCoefficients,
                                           polynomialArg))) {
      return failure();
    }

    // The polynomial argument is the square of the (clamped) erf input, applied
    // via a MultiplyOp (rather than PowOp)
    MultiplyOp polynomialArgOp =
        dyn_cast_or_null<MultiplyOp>(polynomialArg.getDefiningOp());
    if (!polynomialArgOp) {
      return failure();
    }
    if (polynomialArgOp.getLhs() != polynomialArgOp.getRhs()) {
      return failure();
    }
    if (polynomialArgOp.getLhs().getDefiningOp() != clampedErfInput) {
      return failure();
    }

    Value erfInput = clampedErfInput.getInput();

    ErfOp erf = utils::createDPSOp<ErfOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(clamp.getLoc(), "_clamp"),
        mlir::cast<RankedTensorType>(clamp.getType()), erfInput);
    rewriter.replaceOp(clamp, erf);
    return success();
  }
};

class ErfcApproximationFusionPattern : public mlir::OpRewritePattern<WhereOp> {
public:
  ErfcApproximationFusionPattern(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(WhereOp where,
                                PatternRewriter &rewriter) const final {
    // Erfc may be presented as this pattern:
    //
    // erf_approx_small_x(x) = ...
    // erfc_approx_large_x(x) = ...
    // erfc_approx_small_x(x) = 1 - erf_approx_small_x(x)
    // where(|x| < 1, erf_approx_small_x(x), erfc_approx_large_x(x))
    //
    // This fusion pattern will fuse this representation of erfc

    LessThanOp lessThanOne =
        mlir::dyn_cast_or_null<LessThanOp>(where.getFirst().getDefiningOp());
    if (!lessThanOne) {
      return failure();
    }

    FullOp one =
        mlir::dyn_cast_or_null<FullOp>(lessThanOne.getRhs().getDefiningOp());
    if (!one) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(one, 1.0))) {
      return failure();
    }

    AbsOp absX =
        mlir::dyn_cast_or_null<AbsOp>(lessThanOne.getLhs().getDefiningOp());
    if (!absX) {
      return failure();
    }

    Value erfInput = absX.getInput();

    // Confirm that the true branch of the where is erfc_approx_small_x(x) (aka
    // 1 - erf_approx_small_x(x))
    SubtractOp subtract =
        mlir::dyn_cast_or_null<SubtractOp>(where.getSecond().getDefiningOp());
    if (!subtract) {
      return failure();
    }

    Value polynomialArg = nullptr;
    if (mlir::failed(matchErfcSmallInputApproximation(subtract, erfInput,
                                                      polynomialArg))) {
      return failure();
    }

    // Confirm that the false branch of the where is erfc_approx_large_x(x)
    WhereOp largeErfcApproximation =
        mlir::dyn_cast_or_null<WhereOp>(where.getThird().getDefiningOp());
    if (!largeErfcApproximation) {
      return failure();
    }

    if (mlir::failed(matchErfcLargeInputApproximation(largeErfcApproximation,
                                                      erfInput))) {
      return failure();
    }

    ErfOp erf = utils::createDPSOp<ErfOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(where.getLoc(), "_erf"),
        mlir::cast<RankedTensorType>(where.getType()), erfInput);

    SubtractOp oneMinusErf = utils::createDPSOp<SubtractOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(where.getLoc(), "_subtract"),
        mlir::cast<RankedTensorType>(where.getType()), one, erf);
    rewriter.replaceOp(where, oneMinusErf);
    return success();
  }

private:
  LogicalResult matchErfcLargeInputApproximation(WhereOp erfcResult,
                                                 Value erfInput) const {
    // Condition is x < 0
    // True branch of the where is 2 - erfc_approx_clamped
    // False branch of the where is erfc_approx_clamped

    LessThanOp lessThanZero = mlir::dyn_cast_or_null<LessThanOp>(
        erfcResult.getFirst().getDefiningOp());
    if (!lessThanZero) {
      return failure();
    }

    FullOp zero =
        mlir::dyn_cast_or_null<FullOp>(lessThanZero.getRhs().getDefiningOp());
    if (!zero or mlir::failed(matchConstantValue(zero, 0.0))) {
      return failure();
    }

    // True branch of the where is 2 - erfc_approx_clamped
    SubtractOp subtract = mlir::dyn_cast_or_null<SubtractOp>(
        erfcResult.getSecond().getDefiningOp());
    if (!subtract) {
      return failure();
    }

    FullOp two =
        mlir::dyn_cast_or_null<FullOp>(subtract.getLhs().getDefiningOp());
    if (!two or mlir::failed(matchConstantValue(two, 2.0))) {
      return failure();
    }

    // erfc_approx_clamped
    // Condition: -x^2 < -88.72283905206835 (this is the negation of the natural
    // log of the maximum f32 value) True branch: 0 False branch:
    // erfc_approx_large_x
    WhereOp erfcApproxClamped =
        mlir::dyn_cast_or_null<WhereOp>(subtract.getRhs().getDefiningOp());
    if (!erfcApproxClamped) {
      return failure();
    }

    // erfc_approx_clamped condition
    LessThanOp lessThan = mlir::dyn_cast_or_null<LessThanOp>(
        erfcApproxClamped.getFirst().getDefiningOp());
    if (!lessThan) {
      return failure();
    }

    FullOp negMaxLog =
        mlir::dyn_cast_or_null<FullOp>(lessThan.getRhs().getDefiningOp());
    if (!negMaxLog) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(negMaxLog, -88.72283905206835))) {
      return failure();
    }

    NegOp negativeXSquared =
        mlir::dyn_cast_or_null<NegOp>(lessThan.getLhs().getDefiningOp());
    if (!negativeXSquared) {
      return failure();
    }

    // The operand to negativeXSquared ought to be multiply(x, x)
    MultiplyOp multiply = mlir::dyn_cast_or_null<MultiplyOp>(
        negativeXSquared.getInput().getDefiningOp());
    if (!multiply) {
      return failure();
    }

    if (multiply.getLhs() != multiply.getRhs() or
        multiply.getLhs() != erfInput) {
      return failure();
    }

    // True branch of erfc_approx_clamped is 0
    zero = mlir::dyn_cast_or_null<FullOp>(
        erfcApproxClamped.getSecond().getDefiningOp());
    if (!zero or mlir::failed(matchConstantValue(zero, 0.0))) {
      return failure();
    }

    // False branch of erfc_approx_clamped is
    // (e^(-x^2) * (1/|x|)) * Polynomial(1/x^2)
    MultiplyOp falseBranch = mlir::dyn_cast_or_null<MultiplyOp>(
        erfcApproxClamped.getThird().getDefiningOp());
    if (!falseBranch) {
      return failure();
    }

    // Matching (e^(-x^2) * (1/x))
    multiply = mlir::dyn_cast_or_null<MultiplyOp>(
        falseBranch.getLhs().getDefiningOp());
    if (!multiply) {
      return failure();
    }

    // Matching e^(-x^2)
    ExpOp exp =
        mlir::dyn_cast_or_null<ExpOp>(multiply.getLhs().getDefiningOp());
    if (!exp) {
      return failure();
    }

    // The operand to exp ought to be -x^2
    if (exp.getInput().getDefiningOp() != negativeXSquared) {
      return failure();
    }

    // Matching (1/x)
    DivOp div =
        mlir::dyn_cast_or_null<DivOp>(multiply.getRhs().getDefiningOp());
    if (!div) {
      return failure();
    }

    // The numerator ought to be 1
    FullOp one = mlir::dyn_cast_or_null<FullOp>(div.getLhs().getDefiningOp());
    if (!one or mlir::failed(matchConstantValue(one, 1.0))) {
      return failure();
    }

    // The denominator ought to be |x|
    AbsOp absX = mlir::dyn_cast_or_null<AbsOp>(div.getRhs().getDefiningOp());
    if (!absX or absX.getInput() != erfInput) {
      return failure();
    }

    // Matching Polynomial(1/x^2)
    // This polynomial is actually going to be one of two different polynomials
    // based on the value of x
    // Condition: |x| < 2
    // True branch: polynomial #1
    // False branch: polynomial #2

    WhereOp whichPolynomial =
        mlir::dyn_cast_or_null<WhereOp>(falseBranch.getRhs().getDefiningOp());
    if (!whichPolynomial) {
      return failure();
    }
    // Condition: |x| < 2
    LessThanOp absXLessThanTwo = mlir::dyn_cast_or_null<LessThanOp>(
        whichPolynomial.getFirst().getDefiningOp());
    if (!absXLessThanTwo) {
      return failure();
    }

    two = mlir::dyn_cast_or_null<FullOp>(
        absXLessThanTwo.getRhs().getDefiningOp());
    if (!two or mlir::failed(matchConstantValue(two, 2.0))) {
      return failure();
    }

    // True branch
    AddOp trueBranchPolynomial = mlir::dyn_cast_or_null<AddOp>(
        whichPolynomial.getSecond().getDefiningOp());
    if (!trueBranchPolynomial) {
      return failure();
    }

    Value polynomialArg = nullptr;
    SmallVector<float> nestedPolynomialCoefficients = {
        5.638259427386472E-1f,  -2.741127028184656E-1f, 3.404879937665872E-1f,
        -4.944515323274145E-1f, 6.210004621745983E-1f,  -5.824733027278666E-1f,
        3.687424674597105E-1f,  -1.387039388740657E-1f};
    float polynomialA = 2.326819970068386E-2f;
    if (mlir::failed(matchNestedPolynomial(trueBranchPolynomial, polynomialA,
                                           nestedPolynomialCoefficients,
                                           polynomialArg))) {
      return failure();
    }

    // False branch
    AddOp falseBranchPolynomial = mlir::dyn_cast_or_null<AddOp>(
        whichPolynomial.getThird().getDefiningOp());
    if (!falseBranchPolynomial) {
      return failure();
    }
    //   const float kErfcRCoefficients[] = {
    //     -1.047766399936249E+1f, +1.297719955372516E+1f,
    //     -7.495518717768503E+0f, +2.921019019210786E+0f,
    //     -1.015265279202700E+0f, +4.218463358204948E-1f,
    //     -2.820767439740514E-1f, +5.641895067754075E-1f,
    // };
    SmallVector<float> falseBranchNestedPolynomialCoefficients = {
        5.641895067754075E-1f,  -2.820767439740514E-1f, 4.218463358204948E-1f,
        -1.015265279202700E+0f, 2.921019019210786E+0f,  -7.495518717768503E+0f,
        1.297719955372516E+1f};
    float falseBranchPolynomialA = -1.047766399936249E+1f;
    if (mlir::failed(matchNestedPolynomial(
            falseBranchPolynomial, falseBranchPolynomialA,
            falseBranchNestedPolynomialCoefficients, polynomialArg))) {
      return failure();
    }

    // polynomialArg ought to be 1/x^2
    div = mlir::dyn_cast_or_null<DivOp>(polynomialArg.getDefiningOp());
    if (!div) {
      return failure();
    }

    // The numerator ought to be 1
    one = mlir::dyn_cast_or_null<FullOp>(div.getLhs().getDefiningOp());
    if (!one or mlir::failed(matchConstantValue(one, 1.0))) {
      return failure();
    }

    // The denominator ought to be x^2
    multiply = mlir::dyn_cast_or_null<MultiplyOp>(div.getRhs().getDefiningOp());
    if (!multiply) {
      return failure();
    }

    // The left operand ought to be erfInput
    if (multiply.getLhs() != multiply.getRhs() or
        multiply.getLhs() != erfInput) {
      return failure();
    }

    return success();
  }

  LogicalResult matchErfcSmallInputApproximation(SubtractOp erfcResult,
                                                 Value erfInput,
                                                 Value &polynomialArg) const {
    FullOp one =
        mlir::dyn_cast_or_null<FullOp>(erfcResult.getLhs().getDefiningOp());
    if (!one) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(one, 1.0))) {
      return failure();
    }

    // The LHS of this multiply ought to be erfInput
    MultiplyOp multiply =
        mlir::dyn_cast_or_null<MultiplyOp>(erfcResult.getRhs().getDefiningOp());
    if (!multiply) {
      return failure();
    }

    if (multiply.getLhs() != erfInput) {
      return failure();
    }

    // Nested polynomial ends with an AddOp
    AddOp nestedPolynomialResult =
        mlir::dyn_cast_or_null<AddOp>(multiply.getRhs().getDefiningOp());
    if (!nestedPolynomialResult) {
      return failure();
    }

    // Nested polynomial is a nested polynomial with constants (in order from
    // closest to the div to furthest):
    SmallVector<float> nestedPolynomialCoefficients = {
        1.128379165726710E+0f,  -3.761262582423300E-1f, 1.128358514861418E-1f,
        -2.685381193529856E-2f, 5.188327685732524E-3f,  -8.010193625184903E-4f};
    float nestedPolynomialA = +7.853861353153693E-5f;

    if (mlir::failed(matchNestedPolynomial(
            nestedPolynomialResult, nestedPolynomialA,
            nestedPolynomialCoefficients, polynomialArg))) {
      return failure();
    }

    // The polynomialArg ought to be the square of erfInput applied via a
    // MultiplyOp
    MultiplyOp polynomialArgOp =
        mlir::dyn_cast_or_null<MultiplyOp>(polynomialArg.getDefiningOp());
    if (!polynomialArgOp) {
      return failure();
    }

    if (polynomialArgOp.getLhs() != polynomialArgOp.getRhs() or
        polynomialArgOp.getLhs() != erfInput) {
      return failure();
    }

    return success();
  }
};

class GeluFusionPattern : public mlir::OpRewritePattern<MultiplyOp> {
public:
  GeluFusionPattern(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(MultiplyOp multiply,
                                PatternRewriter &rewriter) const final {
    // gelu can be presented as this pattern:
    //
    // gelu(x) = (0.5*x)*(1 + erf(x*rsqrt(2)))
    //
    // This fusion pattern will fuse this representation of gelu

    // The RHS of the multiply shall be 0.5*x

    MultiplyOp multiplyByHalf = dyn_cast_or_null<MultiplyOp>(
        skipTypecast(multiply.getRhs().getDefiningOp()));
    if (!multiplyByHalf) {
      return failure();
    }

    // The LHS of multiplyByHalf will be the input to gelu, the RHS will be a
    // 0.5 constant
    FullOp half = dyn_cast_or_null<FullOp>(
        skipTypecast(multiplyByHalf.getRhs().getDefiningOp()));
    if (!half) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(half, 0.5))) {
      return failure();
    }
    Value geluInput = multiplyByHalf.getLhs();

    // LHS of multiply should be add(erf(multiply(geluInput, rsqrt(2))), 1)

    AddOp plusOne = dyn_cast_or_null<AddOp>(
        skipTypecast(multiply.getLhs().getDefiningOp()));
    if (!plusOne) {
      return failure();
    }

    FullOp one = dyn_cast_or_null<FullOp>(
        skipTypecast(plusOne.getRhs().getDefiningOp()));
    if (!one) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(one, 1.0))) {
      return failure();
    }

    ErfOp erf =
        dyn_cast_or_null<ErfOp>(skipTypecast(plusOne.getLhs().getDefiningOp()));
    if (!erf) {
      return failure();
    }

    MultiplyOp multiplyByRsqrt2 = dyn_cast_or_null<MultiplyOp>(
        skipTypecast(erf.getInput().getDefiningOp()));
    if (!multiplyByRsqrt2) {
      return failure();
    }

    // The LHS of this multiply must be the gelu input
    if (multiplyByRsqrt2.getLhs() != geluInput) {
      return failure();
    }

    RsqrtOp rsqrt2 = dyn_cast_or_null<RsqrtOp>(
        skipTypecast(multiplyByRsqrt2.getRhs().getDefiningOp()));
    if (!rsqrt2) {
      return failure();
    }

    FullOp two = dyn_cast_or_null<FullOp>(
        skipTypecast(rsqrt2.getInput().getDefiningOp()));
    if (!two) {
      return failure();
    }

    if (mlir::failed(matchConstantValue(two, 2.0))) {
      return failure();
    }

    GeluOp gelu = utils::createDPSOp<GeluOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(multiply.getLoc(), "_multiply"),
        mlir::cast<RankedTensorType>(multiply.getType()), geluInput);
    rewriter.replaceOp(multiply, gelu);

    return success();
  }

private:
  // Decompositions will frequently instert typecasts if the original precision
  // is less than f32. We still want to fuse gelu even if its decomposition
  // has typecasts.
  Operation *skipTypecast(Operation *op) const {
    Operation *out = op;
    while (llvm::isa_and_nonnull<TypecastOp>(out)) {
      out = out->getOperand(0).getDefiningOp();
    }
    return out;
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
      patterns.add<ErfApproximationFusionPattern>(&getContext());
      patterns.add<ErfcApproximationFusionPattern>(&getContext());
      patterns.add<GeluFusionPattern>(&getContext());

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir
