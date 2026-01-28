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
  // Maps a softmax axis across a reshape by matching both:
  //   (1) the reduction length of the axis, and
  //   (2) its contiguous stride.
  // Consider a Tensor with shape = [2, 3, 4]
  //   i=0,j=0,k=0
  //   i=0,j=0,k=1
  //   i=0,j=0,k=2
  //   i=0,j=0,k=3
  //   i=0,j=1,k=0
  //   ...
  // strides = [3*4, 4, 1]

  // Example 1 (Valid commute):
  //   Input shape      = [2, 3, 4]
  //   Softmax dim      = 2  (stride = 1, reduction length = 4)
  //   Reshape          = [6, 4]
  // The new softmax dim is 1, since output dim 1 still has stride = 1 and
  // reduction length = 4. Example 2 (Invalid Commute)
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
  // Assuming the shape products fit in int64_t
  int64_t inputStride = 1;
  for (int64_t i = softmaxDim + 1; i < inputDims; ++i) {
    inputStride *= inputShape[i];
  };

  int64_t outputStride = 1;
  SmallVector<int64_t> outputStrides(outputDims);
  for (int64_t i = 0; i < outputDims; ++i) {
    int64_t dim = outputDims - 1 - i;
    outputStrides[dim] = outputStride;
    outputStride *= outputShape[dim];
  };

  // Commute is valid if
  // 1. Input Stride of softmax dimension is preserved in output, and
  // 2. The number of elements the over which softmax normalizes/reduces should
  // also match
  for (int64_t i = 0; i < outputDims; ++i) {
    if (outputStrides[i] == inputStride &&
        outputShape[i] == softmaxReductionLength) {
      return i;
    }
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
  // %0 = softmax(%arg0 : tensor<1x160x160x128xbf16>) <{dimension = 3}>
  // %1 = reshape(%0) <{shape = [1 : i32, 25600 : i32, 128 : i32]}>
  // This method will transform this IR into:
  // %0 = reshape(%arg0 : tensor<1x160x160x128xbf16>) <{shape = [1 : i32, 25600
  // : i32, 128 : i32]}> %1 = softmax(%0) <{dimension = 2}>
  void performCommuteUpwardsRewrite(SoftmaxOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto softmaxInputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputReshapeType =
        cast<RankedTensorType>(reshapeUser.getResult().getType());

    // Get the updated softmax dim
    auto softmaxInputShape = softmaxInputType.getShape(); // Softmax Input
    auto outputReshapeShape =
        outputReshapeType.getShape(); // Reshape User output
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());
    // We are mapping the softmax axis from softmax input to reshape user output
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        softmaxInputShape, outputReshapeShape, softmaxDim);
    assert(newSoftmaxDim != -1 &&
           "Invalid Softmax Dimension, isCommuteUpwardsFavorable should have "
           "handled this");

    // Create a new Reshape before the Softmax op
    auto newReshapeInputType = RankedTensorType::get(
        outputReshapeType.getShape(), softmaxInputType.getElementType(),
        outputReshapeType.getEncoding());
    SmallVector<int32_t> newReshapeShape(outputReshapeShape.begin(),
                                         outputReshapeShape.end());

    auto newInputReshape = rewriter.create<ReshapeOp>(
        reshapeUser->getLoc(), newReshapeInputType, op.getInput(),
        rewriter.getI32ArrayAttr(newReshapeShape));

    // Softmax expects a signed integer attribute (si32)
    auto si32Ty =
        IntegerType::get(rewriter.getContext(), 32,
                         mlir::IntegerType::SignednessSemantics::Signed);
    auto newSoftmaxDimAttr = rewriter.getIntegerAttr(si32Ty, newSoftmaxDim);

    // Create a new Softmax Op with updated dimension attribute
    auto newSoftmaxOp = rewriter.create<SoftmaxOp>(
        op->getLoc(), outputReshapeType, newInputReshape.getResult(),
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
  // %0 = reshape(%arg0 : tensor<1x160x160x128xbf16>) <{shape = [1 : i32, 25600
  // : i32, 128 : i32]}> %1 = softmax(%0) <{dimension = 2}> This method will
  // transform this IR into: %0 = softmax(%arg0 : tensor<1x160x160x128xbf16>)
  // <{dimension = 3}> %1 = reshape(%0) <{shape = [1 : i32, 25600 : i32, 128 :
  // i32]}>
  void
  performCommuteDownwardsRewrite(SoftmaxOp op, ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    // Get the updated softmax dim
    auto reshapeOperandOutputShape = reshapeOperand.getResult()
                                         .getType()
                                         .getShape(); // Reshape Operand Output
    auto reshapeOperandInputShape =
        reshapeOperand.getInput().getType().getShape(); // Reshape Operand Input
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

    // Softmax expects a signed integer attribute (si32)
    auto si32Ty =
        IntegerType::get(rewriter.getContext(), 32,
                         mlir::IntegerType::SignednessSemantics::Signed);
    auto newSoftmaxDimAttr = rewriter.getIntegerAttr(si32Ty, newSoftmaxDim);

    // Create new Softmax Op with updated dimension attribute
    auto newSoftmaxOp = rewriter.create<SoftmaxOp>(
        op->getLoc(), newSoftmaxInputType, reshapeOperand.getInput(),
        newSoftmaxDimAttr, op.getNumericStableAttr());

    // Create new reshape Op
    // Note: For the new reshape op, type is the softmax op
    // Before
    //   %x = reshape(%arg0)
    //   %y = softmax(%x, dim)
    // After commuting downwards
    //   %a = softmax(%arg0, updatedDim)
    //   %b = reshape(%a)
    // Hence, output type of reshape should match %y (original softmax op)
    auto originalSoftmaxOpType =
        cast<RankedTensorType>(op.getResult().getType());
    auto originalSoftmaxOpShape = originalSoftmaxOpType.getShape();
    SmallVector<int32_t> reshapeTargetShape(originalSoftmaxOpShape.begin(),
                                            originalSoftmaxOpShape.end());
    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOperand->getLoc(), op.getType(), newSoftmaxOp.getResult(),
        rewriter.getI32ArrayAttr(reshapeTargetShape));

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
    auto softmaxInputShape = cast<RankedTensorType>(op.getInput().getType())
                                 .getShape(); // Softmax Input
    auto outputReshapeShape =
        cast<RankedTensorType>(reshapeUser.getResult().getType())
            .getShape(); // Reshape User output
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());

    // We are mapping the softmax axis from softmax input to reshape user output
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        softmaxInputShape, outputReshapeShape, softmaxDim);
    if (newSoftmaxDim == -1) {
      // Couldn't find the same stride and reduction dim length in reshape
      // As softmax dimension was merged or split by reshape afterwards.
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
    auto reshapeOperandOutputShape = reshapeOperand.getResult()
                                         .getType()
                                         .getShape(); // Reshape Operand Output
    auto reshapeOperandInputShape =
        reshapeOperand.getInput().getType().getShape(); // Reshape Operand Input
    int64_t softmaxDim = static_cast<int64_t>(op.getDimension());

    // We are mapping the softmax axis from reshape operand output to reshape
    // operand input
    int64_t newSoftmaxDim = findNewSoftmaxDimByStrideAndReductionLen(
        reshapeOperandOutputShape, reshapeOperandInputShape, softmaxDim);
    if (newSoftmaxDim == -1) {
      // Couldn't find the same stride and reduction dim length in reshape
      // As softmax dimension was merged or split by reshape afterwards.
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
