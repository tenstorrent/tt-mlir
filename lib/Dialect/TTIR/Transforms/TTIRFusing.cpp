// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFUSING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
// Check if we can fuse conv2d followed by add into conv2d with bias.
class TTIRConv2dWithBias : public mlir::OpRewritePattern<Conv2dOp> {
  using mlir::OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return mlir::failure();
    }

    // Get the other operand of the add op.
    mlir::OpOperand *nonConvOperand = getBiasOperand(srcOp);
    mlir::Value newBias = nonConvOperand->get();

    // Modify conv2d in place to add bias.
    rewriter.modifyOpInPlace(srcOp,
                             [&]() { srcOp.getBiasMutable().assign(newBias); });

    // Replace add op uses with conv2d.
    rewriter.replaceAllOpUsesWith(nonConvOperand->getOwner(), srcOp);

    // The original conv2d op will be removed by DCE since it's no longer
    // used.
    return mlir::success();
  }

private:
  bool isFusable(Conv2dOp srcOp) const {
    // If bias already exists on Conv2d we cannot fuse.
    if (srcOp.getBias().getImpl()) {
      return false;
    }

    // If conv2d has more than one use we cannot fuse.
    // If it's only user is not AddOp we cannot fuse.
    if (!srcOp.getResult().hasOneUse() ||
        !ttmlir::utils::allUsers<AddOp>(srcOp)) {
      return false;
    }

    // If we have this IR:
    // %0 = empty()
    // %1 = conv2d()
    // %2 = add(%1, %0)
    // we want to get type of %0 and check if it is compatible with conv2d bias.
    OpOperand *nonConvOperand = getBiasOperand(srcOp);
    mlir::Operation *otherOperation = nonConvOperand->get().getDefiningOp();
    mlir::RankedTensorType nonConvType =
        mlir::cast<mlir::RankedTensorType>(nonConvOperand->get().getType());

    // If nonConvOperand is not defined by an operation, its a block argument
    // and we can fuse if bias is compatible with its shape.
    if (otherOperation == nullptr) {
      return srcOp.verifyBias(nonConvType.getShape());
    }

    // If we have this IR:
    // %0 = conv2d()
    // %1 = empty()
    // %2 = add(%0, %1)
    // We cannot fuse since %1 comes after conv2d. In this simple case
    // we can move empty above covn2d, but in general case it would
    // require more complex analysis.
    if (!otherOperation->isBeforeInBlock(srcOp)) {
      return false;
    }

    return srcOp.verifyBias(nonConvType.getShape());
  }

  OpOperand *getBiasOperand(Conv2dOp srcOp) const {
    auto addOp = mlir::cast<AddOp>(*srcOp.getResult().getUsers().begin());
    auto otherOperands = ttmlir::utils::getOtherOperands(addOp, srcOp);
    return otherOperands.front();
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
  matchAndRewrite(ReductionOpTy srcOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if the reduction op has exactly one use and it's a reshape op.
    if (!srcOp.getResult().hasOneUse() ||
        !ttmlir::utils::allUsers<ReshapeOp>(srcOp)) {
      return mlir::failure();
    }

    // Get the reshape op that follows the reduction op.
    auto reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    // If the reduction already has keep_dim=true, there's nothing to fuse.
    if (srcOp.getKeepDim()) {
      return mlir::failure();
    }

    // Get the input and output shapes.
    ArrayRef<int64_t> inputShape = srcOp.getInput().getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();

    // Check if the reshape is simply adding back dimensions that were reduced.
    // For this, the reshape output shape should have the same rank as the input
    // shape.
    if (reshapeOutputShape.size() != inputShape.size()) {
      return mlir::failure();
    }

    // Get the reduction dimensions.
    llvm::BitVector reduceDimsMask(inputShape.size(), false);
    if (!srcOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced.
      reduceDimsMask.set();
    } else {
      mlir::ArrayAttr reduceDims = *srcOp.getDimArg();
      for (mlir::Attribute reduceDim : reduceDims) {
        int64_t reduceDimInt =
            mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
        // Handle negative indices
        reduceDimInt = (reduceDimInt + inputShape.size()) % inputShape.size();
        reduceDimsMask.set(reduceDimInt);
      }
    }

    // Check if the reshape output shape has 1s in the reduced dimensions and
    // matches the original shape in the non-reduced dimensions.
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (reduceDimsMask[i]) {
        // Reduced dimension should be 1 in the reshape output.
        if (reshapeOutputShape[i] != 1) {
          return mlir::failure();
        }
      } else {
        // Non-reduced dimension should match the original input shape.
        if (reshapeOutputShape[i] != inputShape[i]) {
          return mlir::failure();
        }
      }
    }

    // Create a new reduction op with keep_dim=true.
    auto newReductionOp = utils::createDPSOp<ReductionOpTy>(
        rewriter, srcOp.getLoc(), reshapeOp.getResult().getType(),
        srcOp.getInput(), rewriter.getBoolAttr(true), srcOp.getDimArgAttr());

    // Replace the reshape op with the result of the new reduction op.
    rewriter.replaceOp(reshapeOp, newReductionOp.getResult());

    // The original reduction op will be removed by DCE since it's no longer
    // used.
    return mlir::success();
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
  // Pattern: div(exp(x), broadcast(sum(exp(x), keep_dim=true)))
  mlir::LogicalResult
  matchAndRewrite(DivOp srcOp, mlir::PatternRewriter &rewriter) const final {

    // Get the numerator (exp) and denominator (broadcast)
    mlir::Value numerator = srcOp.getInputs()[0];
    mlir::Value denominator = srcOp.getInputs()[1];

    // Check that the numerator is an exp operation
    auto expOp = numerator.getDefiningOp<ExpOp>();
    if (!expOp) {
      return mlir::failure();
    }

    // Check that the denominator is a broadcast operation
    auto broadcastOp = denominator.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return mlir::failure();
    }

    // Check that the broadcast input is a sum operation with keep_dim=true
    auto sumOp = broadcastOp.getInput().getDefiningOp<SumOp>();
    if (!sumOp || !sumOp.getKeepDim()) {
      return mlir::failure();
    }

    // Check that the sum input is the same exp operation as the numerator
    if (sumOp.getInput() != expOp.getResult(0)) {
      return mlir::failure();
    }

    // Check correct user counts for each operation in the pattern:
    // - exp should have exactly 2 users (sum and div)
    // - sum should have exactly 1 user (broadcast)
    // - broadcast should have exactly 1 user (div)
    if (ttmlir::utils::countUsers(expOp.getResult(0)) != 2 ||
        !sumOp.getResult().hasOneUse() ||
        !broadcastOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Get the reduction dimension from the sum operation
    if (!sumOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced, which is
      // not what we want for softmax
      return mlir::failure();
    }

    mlir::ArrayAttr reduceDims = *sumOp.getDimArg();
    if (reduceDims.size() != 1) {
      // Softmax reduces along a single dimension
      return mlir::failure();
    }

    int64_t reduceDim = mlir::cast<mlir::IntegerAttr>(reduceDims[0]).getInt();

    // Create a new softmax operation
    auto softmaxOp = utils::createDPSOp<SoftmaxOp>(
        rewriter, srcOp.getLoc(),
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType()),
        expOp.getInputs()[0], reduceDim);

    // Replace the div operation with the softmax operation
    rewriter.replaceOp(srcOp, softmaxOp.getResult());

    return mlir::success();
  }
};

class Conv2dWithMultiply : public mlir::OpRewritePattern<Conv2dOp> {
  using mlir::OpRewritePattern<Conv2dOp>::OpRewritePattern;

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
  matchAndRewrite(Conv2dOp convOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if this pattern is applicable
    if (!isCommutable(convOp)) {
      return mlir::failure();
    }

    // Get the scale factor from the multiply operation
    OpOperand *scaleOperand = getScaleOperand(convOp);
    Value scaleValue = scaleOperand->get();

    // Get the convolution weight
    Value weightValue = convOp.getWeight();

    // Create a reshaped scale factor suitable for multiplying with weights
    Value reshapedScale = createReshapedScale(rewriter, convOp, scaleValue);

    // Create pre-multiplied weights
    Value scaledWeights =
        createScaledWeights(rewriter, convOp, weightValue, reshapedScale);

    // Modify conv2d in place to add scaled weights.
    rewriter.modifyOpInPlace(
        convOp, [&]() { convOp.getWeightMutable().assign(scaledWeights); });

    // Replace the multiply operation uses with conv2d.
    rewriter.replaceAllOpUsesWith(scaleOperand->getOwner(), convOp);

    return mlir::success();
  }

private:
  bool isCommutable(Conv2dOp convOp) const {
    // Convolution must have exactly one use and it must be a multiply operation
    if (!convOp.getResult().hasOneUse() ||
        !ttmlir::utils::allUsers<MultiplyOp>(convOp)) {
      return false;
    }

    // Get the function containing this operation
    mlir::func::FuncOp funcOp = convOp->getParentOfType<mlir::func::FuncOp>();

    // Get all constant parameters in the function
    SmallPtrSet<BlockArgument, 4> constParams =
        ttmlir::utils::populateConstParams(funcOp);

    // Both the weight and scale must be constants
    if (!isConstant(convOp.getWeight(), constParams) ||
        !isConstant(getScaleOperand(convOp)->get(), constParams)) {
      return false;
    }

    // The scale must have the correct shape (1,1,1,out_channels)
    return hasValidScaleShape(convOp, getScaleOperand(convOp));
  }

  /// Check if a value is a constant parameter
  bool isConstant(mlir::Value value,
                  const SmallPtrSet<BlockArgument, 4> &constParams) const {
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
      return constParams.contains(blockArg);
    }
    return false;
  }

  /// Check if the scale has the correct shape (1,1,1,out_channels)
  bool hasValidScaleShape(Conv2dOp convOp, OpOperand *scaleOperand) const {
    int64_t outChannels = convOp.getOutputChannelSize();
    RankedTensorType scaleType =
        mlir::cast<RankedTensorType>(scaleOperand->get().getType());

    // Scale must be a 4D tensor
    if (scaleType.getRank() != 4) {
      return false;
    }

    // Scale must have shape (1,1,1,out_channels)
    return scaleType.getDimSize(0) == 1 && scaleType.getDimSize(1) == 1 &&
           scaleType.getDimSize(2) == 1 &&
           scaleType.getDimSize(3) == outChannels;
  }

  /// Get the scale operand from the multiply operation
  OpOperand *getScaleOperand(Conv2dOp convOp) const {
    MultiplyOp multiplyOp =
        mlir::cast<MultiplyOp>(*convOp.getResult().getUsers().begin());
    return ttmlir::utils::getOtherOperands(multiplyOp, convOp).front();
  }

  /// Create a reshaped scale factor suitable for multiplying with weights
  Value createReshapedScale(mlir::PatternRewriter &rewriter, Conv2dOp convOp,
                            Value scaleValue) const {
    // Get the scale's type
    RankedTensorType scaleType =
        mlir::cast<RankedTensorType>(scaleValue.getType());

    // Create a new shape (out_channels,1,1,1) from (1,1,1,out_channels)
    llvm::SmallVector<int64_t> newShape(scaleType.getShape());
    std::swap(newShape[0], newShape[3]); // Swap first and last dimensions

    // Convert to int32 for the reshape operation
    llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());

    // Create and return the reshape operation
    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
               rewriter, convOp.getLoc(), newShape, scaleType.getElementType(),
               scaleType.getEncoding(), scaleValue,
               rewriter.getI32ArrayAttr(newShapeI32))
        .getResult();
  }

  /// Create pre-multiplied weights
  Value createScaledWeights(mlir::PatternRewriter &rewriter, Conv2dOp convOp,
                            Value weightValue, Value reshapedScale) const {
    // Create a multiplication of the weights by the reshaped scale
    return utils::createDPSOp<MultiplyOp>(
               rewriter, convOp.getLoc(),
               mlir::cast<RankedTensorType>(weightValue.getType()), weightValue,
               reshapedScale)
        .getResult(0);
  }

  /// Create a new convolution with the scaled weights
  Conv2dOp createConvWithScaledWeights(mlir::PatternRewriter &rewriter,
                                       Conv2dOp originalConv,
                                       Value scaledWeights) const {
    // Create a new convolution operation with the scaled weights
    return utils::createDPSOp<Conv2dOp>(
        rewriter, originalConv.getLoc(), originalConv.getResult().getType(),
        originalConv.getInput(), scaledWeights, originalConv.getBias(),
        originalConv.getStride(), originalConv.getPadding(),
        originalConv.getDilation(), originalConv.getGroupsAttr(),
        originalConv.getFlattenedCompatInfo());
  }
};

class TTIRFusingPass : public impl::TTIRFusingBase<TTIRFusingPass> {
public:
  using impl::TTIRFusingBase<TTIRFusingPass>::TTIRFusingBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRConv2dWithBias>(&getContext());

    // Add patterns for each reduction op type
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

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace
} // namespace mlir::tt::ttir
