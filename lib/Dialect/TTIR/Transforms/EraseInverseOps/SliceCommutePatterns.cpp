// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include <cassert>

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughSlice
    : public TTIRCommuteOpRewritePattern<PermuteOp, SliceStaticOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, SliceStaticOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // %0 = slice(%arg0) <{
  //      begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = permute(%1) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  void performCommuteUpwardsRewrite(SliceStaticOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {

    RankedTensorType sliceOperandType = op.getInput().getType();

    RankedTensorType newPermuteType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(sliceOperandType.getShape(),
                                        permuteUser.getPermutation()),
        sliceOperandType.getElementType(), sliceOperandType.getEncoding());

    PermuteOp newPerm = utils::createDPSOp<PermuteOp>(
        rewriter, op->getLoc(), newPermuteType, op.getInput(),
        permuteUser.getPermutation());

    SmallVector<Attribute> newSliceStarts = ttmlir::utils::applyPermutation(
        op.getBegins().getValue(), permuteUser.getPermutation());
    SmallVector<Attribute> newSliceEnds = ttmlir::utils::applyPermutation(
        op.getEnds().getValue(), permuteUser.getPermutation());
    SmallVector<Attribute> newSliceSteps = ttmlir::utils::applyPermutation(
        op.getStep().getValue(), permuteUser.getPermutation());

    RankedTensorType newSliceType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        permuteUser.getPermutation()),
        op.getType().getElementType(), op.getType().getEncoding());

    SliceStaticOp newSlice = utils::createDPSOp<SliceStaticOp>(
        rewriter, op->getLoc(), newSliceType, newPerm,
        rewriter.getArrayAttr(newSliceStarts),
        rewriter.getArrayAttr(newSliceEnds),
        rewriter.getArrayAttr(newSliceSteps));

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
      rewriter.replaceOp(user, newSlice);
    }
  }

  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}
  //
  // This method will transform this into:
  // %0 = slice(%arg0) <{
  //      begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = permute(%1) <{permutation = array<i64: 0, 2, 3, 1>}>
  void
  performCommuteDownwardsRewrite(SliceStaticOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    SmallVector<int64_t> inversePermutation =
        ttmlir::utils::inversePermutation(permuteOperand.getPermutation());
    RankedTensorType newSliceType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        inversePermutation),
        op.getType().getElementType(), op.getType().getEncoding());

    SmallVector<Attribute> newSliceStarts = ttmlir::utils::applyPermutation(
        op.getBegins().getValue(), inversePermutation);
    SmallVector<Attribute> newSliceEnds = ttmlir::utils::applyPermutation(
        op.getEnds().getValue(), inversePermutation);
    SmallVector<Attribute> newSliceSteps = ttmlir::utils::applyPermutation(
        op.getStep().getValue(), inversePermutation);

    SliceStaticOp newSlice = utils::createDPSOp<SliceStaticOp>(
        rewriter, op->getLoc(), newSliceType, permuteOperand.getInput(),
        rewriter.getArrayAttr(newSliceStarts),
        rewriter.getArrayAttr(newSliceEnds),
        rewriter.getArrayAttr(newSliceSteps));

    RankedTensorType newPermuteType = RankedTensorType::get(
        op.getType().getShape(), newSlice.getType().getElementType(),
        newSlice.getType().getEncoding());
    PermuteOp newPerm = utils::createDPSOp<PermuteOp>(
        rewriter, op->getLoc(), newPermuteType, newSlice,
        permuteOperand.getPermutation());

    rewriter.replaceOp(op, newPerm);
  }

private:
  bool isCommuteUpwardsViable(SliceStaticOp op, PermuteOp) const override {
    // Commuting a permute through a slice is always viable
    return true;
  }

  bool isCommuteUpwardsFavorable(SliceStaticOp op, PermuteOp) const override {
    // We should always commute a permute above a slice if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SliceStaticOp op, PermuteOp) const override {
    // Commuting a permute through a slice is always viable
    return true;
  }

  bool isCommuteDownwardsFavorable(SliceStaticOp op,
                                   PermuteOp permuteOperand) const override {
    // Commuting downwards is favorable if the all other operands a satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      if (checkIdenticalTms(op->getOperand(i).getDefiningOp(),
                            permuteOperand) ||
          ttcore::valueTracesToConstantArgs(op->getOperand(i))) {
        continue;
      }
      return false;
    }
    return true;
  }
};

template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughSlice
    : public TTIRCommuteOpRewritePattern<ReshapeOp, SliceStaticOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, SliceStaticOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(SliceStaticOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    ReshapeOp newReshape = deslicedReshapeOp(op, reshapeUser, rewriter);

    SliceStaticOp newSlice =
        reshapedSliceStaticOp(op, newReshape.getResult(), rewriter);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(reshapeUser, user) &&
             "shouldCommute should have ensured this is true");
      rewriter.replaceOp(user, newSlice);
    }
  }

  void
  performCommuteDownwardsRewrite(SliceStaticOp op, ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    SliceStaticOp newSlice =
        reshapedSliceStaticOp(op, reshapeOperand.getInput(), rewriter);

    // The reshape should produce the same output type as the original slice
    SmallVector<int32_t> reshapeTargetShape(op.getType().getShape());
    ReshapeOp newReshape = utils::createDPSOp<ReshapeOp>(
        rewriter, op->getLoc(), op.getType(), newSlice,
        rewriter.getI32ArrayAttr(reshapeTargetShape));
    rewriter.replaceOp(op, newReshape);
  }

private:
  int64_t getRightmostReshapeDimRTL(ReshapeOp reshapeOp) const {
    ArrayRef<int64_t> inputShape = reshapeOp.getInput().getType().getShape();
    ArrayRef<int64_t> outputShape = reshapeOp.getResult().getType().getShape();
    auto inputRank = inputShape.size();
    auto outputRank = outputShape.size();
    int64_t minRank = std::min(inputRank, outputRank);

    // Find the rightmost changed dimension (counting from right to left).
    // Return the right-to-left position (0 = rightmost, ndims-1 = leftmost).
    for (int64_t i = 0; i < minRank; i++) {
      if (inputShape[inputRank - 1 - i] != outputShape[outputRank - 1 - i]) {
        llvm::outs() << "rightmostReshapeDimRTL: " << i << "\n";
        return i;
      }
    }

    // If all overlapping dimensions are identical, return -1
    return -1;
  }

  int64_t getLeftmostSlicedDimRTL(SliceStaticOp sliceOp) const {
    ArrayRef<int64_t> inputShape = sliceOp.getInput().getType().getShape();
    auto begins = sliceOp.getBegins().getValue();
    auto ends = sliceOp.getEnds().getValue();
    auto steps = sliceOp.getStep().getValue();
    int64_t ndims = begins.size();

    // Find the leftmost sliced dimension (counting from right to left).
    // Return the right-to-left position (0 = rightmost, ndims-1 = leftmost).
    int64_t leftmostSlicedDim = -1;
    for (int64_t i = ndims - 1; i >= 0; --i) {
      int32_t begin = cast<IntegerAttr>(begins[i]).getInt();
      int32_t end = cast<IntegerAttr>(ends[i]).getInt();
      int32_t step = cast<IntegerAttr>(steps[i]).getInt();
      int64_t dimSize = inputShape[i];
      int32_t adjustedEnd = (end < 0) ? (end + dimSize) : end;
      int32_t adjustedBegin = (begin < 0) ? (begin + dimSize) : begin;

      // Tensor is sliced along a dimension if it doesn't have the full range,
      // which can be:
      // - begin != 0
      // - end != dimSize
      // - step != 1

      if (adjustedBegin != 0 || adjustedEnd != dimSize || step != 1) {
        leftmostSlicedDim = ndims - 1 - i;
        break;
      }
    }
    llvm::outs() << "leftmostSlicedDim: " << leftmostSlicedDim << "\n";
    // If no slicing is applied, return -1
    return leftmostSlicedDim;
  }

  SliceStaticOp reshapedSliceStaticOp(SliceStaticOp op,
                                      TypedValue<RankedTensorType> input,
                                      PatternRewriter &rewriter) const {
    ArrayRef<int64_t> shape = input.getType().getShape();
    int64_t ndims = shape.size();

    int64_t leftmostSlicedDimRTL = getLeftmostSlicedDimRTL(op);
    assert(leftmostSlicedDimRTL != -1 && "Expected valid slicing dimension");

    // Update ends and output shape to match new input shape:
    // - ends = dim size
    // - output shape = new shape
    // for dims to the left of leftmost sliced dim

    auto originalEnds = op.getEnds().getValue();
    auto originalOutputShape = op.getType().getShape();
    int64_t originalNdims = originalEnds.size();

    SmallVector<Attribute> newEnds(ndims);
    SmallVector<int64_t> newOutputShape(ndims);

    for (int64_t i = 0; i < ndims; ++i) {
      if (i > leftmostSlicedDimRTL) {
        // For dims to the left of leftmost sliced dim, set ends=dim size
        newEnds[ndims - 1 - i] = IntegerAttr::get(
            IntegerType::get(op->getContext(), 32), shape[ndims - 1 - i]);
        newOutputShape[ndims - 1 - i] = shape[ndims - 1 - i];
      } else {
        // For other dims, copy original values
        newEnds[ndims - 1 - i] = originalEnds[originalNdims - 1 - i];
        newOutputShape[ndims - 1 - i] =
            originalOutputShape[originalNdims - 1 - i];
      }
    }

    // Update begins and steps to match new ends:
    // - begins = 0
    // - steps = 1
    // for dims to the left of leftmost sliced dim

    auto originalBegins = op.getBegins().getValue();
    auto originalSteps = op.getStep().getValue();

    SmallVector<Attribute> newBegins(ndims);
    SmallVector<Attribute> newSteps(ndims);

    for (int64_t i = 0; i < ndims; ++i) {
      if (i > leftmostSlicedDimRTL) {
        // For dims to the left of the leftmost sliced dim, set begins=0,
        // steps=1
        newBegins[ndims - 1 - i] =
            IntegerAttr::get(IntegerType::get(op->getContext(), 32), 0);
        newSteps[ndims - 1 - i] =
            IntegerAttr::get(IntegerType::get(op->getContext(), 32), 1);
      } else {
        // For other dims, copy original values
        newBegins[ndims - 1 - i] = originalBegins[originalNdims - 1 - i];
        newSteps[ndims - 1 - i] = originalSteps[originalNdims - 1 - i];
      }
    }

    RankedTensorType newSliceType =
        RankedTensorType::get(newOutputShape, op.getType().getElementType(),
                              op.getType().getEncoding());

    SliceStaticOp newSlice = utils::createDPSOp<SliceStaticOp>(
        rewriter, op->getLoc(), newSliceType, input,
        rewriter.getArrayAttr(newBegins), rewriter.getArrayAttr(newEnds),
        rewriter.getArrayAttr(newSteps));

    return newSlice;
  }

  ReshapeOp deslicedReshapeOp(SliceStaticOp op, ReshapeOp reshapeUser,
                              PatternRewriter &rewriter) const {
    SmallVector<int64_t> targetShape;
    ArrayRef<int64_t> originalReshapeShape =
        reshapeUser.getResult().getType().getShape();
    ArrayRef<int64_t> sliceInputShape = op.getInput().getType().getShape();
    int64_t ndims = sliceInputShape.size();

    int64_t leftmostSlicedDimRTL = getLeftmostSlicedDimRTL(op);
    assert(leftmostSlicedDimRTL != -1 && "Expected valid slicing dimension");
    int64_t leftmostSlicedDimLTR = ndims - 1 - leftmostSlicedDimRTL;

    for (int64_t i = 0; i < ndims; ++i) {
      if (i < leftmostSlicedDimLTR) {
        targetShape.push_back(originalReshapeShape[i]);
      } else {
        targetShape.push_back(sliceInputShape[i]);
      }
    }

    auto targetType = RankedTensorType::get(
        targetShape, op.getInput().getType().getElementType(),
        op.getInput().getType().getEncoding());
    SmallVector<int32_t> targetShapeInt32(targetShape.begin(),
                                          targetShape.end());
    return utils::createDPSOp<ReshapeOp>(
        rewriter, op->getLoc(), targetType, op.getInput(),
        rewriter.getI32ArrayAttr(targetShapeInt32));
  }

  bool isCommuteUpwardsViable(SliceStaticOp op,
                              ReshapeOp reshapeUser) const override {
    // Reshape can commute if its rightmost reshape dim (rightmost changing dim)
    // is to the right of leftmost slicing dim (rightmost slicing dim), counting
    // from right to left
    int64_t rightmostReshapeDimRTL = getRightmostReshapeDimRTL(reshapeUser);
    int64_t leftmostSlicedDimRTL = getLeftmostSlicedDimRTL(op);
    if (rightmostReshapeDimRTL == -1 || leftmostSlicedDimRTL == -1) {
      return false;
    }
    // Larger right-to-left position means further left
    return rightmostReshapeDimRTL > leftmostSlicedDimRTL;
  }

  bool isCommuteUpwardsFavorable(SliceStaticOp op, ReshapeOp) const override {
    // We should always commute a reshape above a slice if all users are an
    // identical reshape. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SliceStaticOp op,
                                ReshapeOp reshapeOperand) const override {
    // Reshape can commute if its rightmost reshape dim (rightmost changing dim)
    // is to the right of leftmost slicing dim (rightmost slicing dim), counting
    // from right to left
    int64_t rightmostReshapeDimRTL = getRightmostReshapeDimRTL(reshapeOperand);
    int64_t leftmostSlicedDimRTL = getLeftmostSlicedDimRTL(op);
    if (rightmostReshapeDimRTL == -1 || leftmostSlicedDimRTL == -1) {
      return false;
    }
    // Larger right-to-left position means further left
    return rightmostReshapeDimRTL > leftmostSlicedDimRTL;
  }

  bool isCommuteDownwardsFavorable(SliceStaticOp op,
                                   ReshapeOp reshapeOperand) const override {
    // Commuting downwards is favorable if all other operands satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      if (checkIdenticalTms(op->getOperand(i).getDefiningOp(),
                            reshapeOperand) ||
          ttcore::valueTracesToConstantArgs(op->getOperand(i))) {
        continue;
      }
      return false;
    }
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateSliceCommutePatterns(MLIRContext *ctx,
                                  RewritePatternSet &patterns) {
  patterns.insert<TTIRCommutePermuteThroughSlice<commuteDirection>>(ctx);
  patterns.insert<TTIRCommuteReshapeThroughSlice<commuteDirection>>(ctx);
}

template void populateSliceCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateSliceCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
