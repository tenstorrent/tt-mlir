// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {

int64_t
findNewSoftmaxDimByStrideAndReductionLen(llvm::ArrayRef<int64_t> inputShape,
                                         llvm::ArrayRef<int64_t> outputShape,
                                         int64_t softmaxDim) {
  // Stride of new softmax dim in new shape must match stride of old
  // softmax dim in old shape, and same for length.
  //
  // Example 1 (Valid commute):
  //   Input shape      = [2, 3, 4]
  //   Softmax dim      = 2  (stride = 1, reduction length = 4)
  //   Reshape          = [6, 4]
  // The new softmax dim is 1, since output dim 1 still has stride = 1 and
  // reduction length = 4.
  //
  // Example 2 (Invalid Commute)
  //   Input shape      = [2, 3, 4]
  //   Softmax dim      = 1  (stride = 4, reduction length = 3)
  //   Reshape          = [2, 12]
  //   output strides = [12, 1],  Hence, we can't apply softmax because no
  //   dimension in output has stride 4 and reduction len = 3
  int64_t inputDims = inputShape.size();
  int64_t outputDims = outputShape.size();

  if (softmaxDim < 0) {
    softmaxDim += inputDims;
  }

  assert(softmaxDim >= 0 && softmaxDim < inputDims &&
         "Invalid softmax Dimension");

  int64_t softmaxReductionLength = inputShape[softmaxDim];
  int64_t inputStride = 1;
  for (int64_t i = softmaxDim + 1; i < inputDims; ++i) {
    inputStride *= inputShape[i];
  };

  // Commute is valid if
  // 1. Input Stride of softmax dimension is preserved in output, and
  // 2. The number of elements the over which softmax normalizes/reduces should
  // also match
  int64_t outputStride = 1;
  for (int64_t dim = outputDims - 1; dim >= 0; dim--) {
    if (outputStride == inputStride &&
        outputShape[dim] == softmaxReductionLength) {
      return dim;
    }
    outputStride *= outputShape[dim];
  };

  return -1;
}

template <CommuteDirection commuteDirection>
// <TMOp, Op, CommuteDirection>
class TTIRCommuteReshapeThroughSoftmax
    : public TTIRCommuteOpRewritePattern<ReshapeOp, SoftmaxOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, SoftmaxOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // Before commuting reshape op upwards
  //   %a = softmax(%arg0, dim)
  //   %b = reshape(%a)
  // After commuting reshape op upwards
  //   %x = reshape(%arg0)
  //   %y = softmax(%x, UpdatedDim)
  void performCommuteUpwardsRewrite(SoftmaxOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto softmaxInputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputReshapeType =
        cast<RankedTensorType>(reshapeUser.getResult().getType());

    auto softmaxInputShape = softmaxInputType.getShape();
    auto outputReshapeShape = outputReshapeType.getShape();
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());
    // We are mapping the softmax axis from softmax input to reshape user output
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        softmaxInputShape, outputReshapeShape, softmaxDim);
    assert(newSoftmaxDim != -1 &&
           "Invalid Softmax Dimension, isCommuteUpwardsFavorable should have "
           "handled this");

    // Create a new Reshape before the Softmax op
    auto newInputReshapeType = RankedTensorType::get(
        outputReshapeType.getShape(), softmaxInputType.getElementType(),
        outputReshapeType.getEncoding());
    SmallVector<int32_t> newReshapeShape(outputReshapeShape.begin(),
                                         outputReshapeShape.end());

    auto newInputReshape = ReshapeOp::create(
        rewriter, reshapeUser->getLoc(), newInputReshapeType, op.getInput(),
        rewriter.getI32ArrayAttr(newReshapeShape));

    auto newSoftmaxDimAttr = rewriter.getSI32IntegerAttr(newSoftmaxDim);

    // Create a new Softmax Op with updated dimension attribute
    auto newSoftmaxOp = SoftmaxOp::create(
        rewriter, op->getLoc(), outputReshapeType, newInputReshape.getResult(),
        newSoftmaxDimAttr, op.getNumericStableAttr());

    // All users must be identical TMs.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(reshapeUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    for (auto *user : users) {
      rewriter.replaceOp(user, newSoftmaxOp.getResult());
    };
  }

  // Consider the following IR pseudocode:
  // Before commuting reshape op downwards
  //   %x = reshape(%arg0)
  //   %y = softmax(%x, dim)
  // After commuting reshape op downwards
  //   %a = softmax(%arg0, updatedDim)
  //   %b = reshape(%a)
  void
  performCommuteDownwardsRewrite(SoftmaxOp op, ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    auto reshapeOperandOutputShape =
        reshapeOperand.getResult().getType().getShape();
    auto reshapeOperandInputShape =
        reshapeOperand.getInput().getType().getShape();
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());
    // We are mapping the softmax axis from reshape operand output to reshape
    // operand input
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        reshapeOperandOutputShape, reshapeOperandInputShape, softmaxDim);
    assert(newSoftmaxDim != -1 &&
           "Invalid Softmax Dimension, isCommuteDownwardsFavorable should have "
           "handled this");

    // Create the Softmax Op on the Reshape Operand's input
    auto newSoftmaxInputType =
        cast<RankedTensorType>(reshapeOperand.getInput().getType());

    auto newSoftmaxDimAttr = rewriter.getSI32IntegerAttr(newSoftmaxDim);

    // Create new Softmax Op with updated dimension attribute
    auto newSoftmaxOp = SoftmaxOp::create(
        rewriter, op->getLoc(), newSoftmaxInputType, reshapeOperand.getInput(),
        newSoftmaxDimAttr, op.getNumericStableAttr());

    // Create new reshape Op
    auto originalSoftmaxOpType =
        cast<RankedTensorType>(op.getResult().getType());
    auto originalSoftmaxOpShape = originalSoftmaxOpType.getShape();
    SmallVector<int32_t> reshapeTargetShape(originalSoftmaxOpShape.begin(),
                                            originalSoftmaxOpShape.end());
    auto newReshapeOp = ReshapeOp::create(
        rewriter, reshapeOperand->getLoc(), op.getType(),
        newSoftmaxOp.getResult(), rewriter.getI32ArrayAttr(reshapeTargetShape));

    rewriter.replaceOp(op, newReshapeOp.getResult());
  }

private:
  bool isCommuteUpwardsViable(SoftmaxOp op,
                              ReshapeOp reshapeUser) const override {
    return true;
  }

  bool isCommuteUpwardsFavorable(SoftmaxOp op,
                                 ReshapeOp reshapeUser) const override {
    // Check if the softmax dimension was not split or merged by reshape, if it
    // does then we can't commute
    auto softmaxInputShape =
        cast<RankedTensorType>(op.getInput().getType()).getShape();
    auto outputReshapeShape =
        cast<RankedTensorType>(reshapeUser.getResult().getType()).getShape();
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());

    // We are mapping the softmax axis from softmax input to reshape user output
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        softmaxInputShape, outputReshapeShape, softmaxDim);
    if (newSoftmaxDim == -1) {
      return false;
    }

    // Otherwise, We should always commute a reshape above softmax op if all
    // users are an identical TMs/reshapes. This includes the case where there
    // is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SoftmaxOp op,
                                ReshapeOp reshapeOperand) const override {
    return true;
  }

  bool isCommuteDownwardsFavorable(SoftmaxOp op,
                                   ReshapeOp reshapeOperand) const override {
    // We need to ensure the softmax dim on the reshape is not the axis
    // which gets split or merged from the input
    // For example consider
    // %x -- Input
    // %y = reshape(%x) : [2,3,4] -> [2,12]
    // %z = softmax(%y, dim=1) // reduces over 12
    // If we commute the reshape down the softmax
    // softmax(%x, dim = ?)
    // No axis where reduction length is 12.
    auto reshapeOperandOutputShape =
        reshapeOperand.getResult().getType().getShape();
    auto reshapeOperandInputShape =
        reshapeOperand.getInput().getType().getShape();
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());

    // We are mapping the softmax axis from reshape operand output to reshape
    // operand input
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        reshapeOperandOutputShape, reshapeOperandInputShape, softmaxDim);
    if (newSoftmaxDim == -1) {
      return false;
    }

    // If the above condition satisfies, then commuting a reshape downwards
    // through a softmax op is always favorable as
    // softmax only has one operand, and here we know it is a reshape.
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateSoftmaxCommutePatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns) {
  patterns.insert<TTIRCommuteReshapeThroughSoftmax<commuteDirection>>(ctx);
}

template void populateSoftmaxCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

template void populateSoftmaxCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
