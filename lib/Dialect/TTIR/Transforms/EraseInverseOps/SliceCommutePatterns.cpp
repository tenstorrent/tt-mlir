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
        reshapedSliceStaticOp(op, newReshape.getResult(),
                              getHighestReshapeDim(reshapeUser), rewriter);

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
        reshapedSliceStaticOp(op, reshapeOperand.getInput(),
                              getHighestReshapeDim(reshapeOperand), rewriter);

    // The reshape should produce the same output type as the original slice
    SmallVector<int32_t> reshapeTargetShape(op.getType().getShape());
    ReshapeOp newReshape = utils::createDPSOp<ReshapeOp>(
        rewriter, op->getLoc(), op.getType(), newSlice,
        rewriter.getI32ArrayAttr(reshapeTargetShape));
    rewriter.replaceOp(op, newReshape);
  }

private:
  int64_t getHighestReshapeDim(ReshapeOp reshapeOp) const {
    ArrayRef<int64_t> inputShape = reshapeOp.getInput().getType().getShape();
    ArrayRef<int64_t> outputShape = reshapeOp.getResult().getType().getShape();
    int64_t minRank = std::min(inputShape.size(), outputShape.size());

    // Find the highest dimension that changes, starting from the end
    for (int64_t i = minRank - 1; i >= 0; --i) {
      if (inputShape[i] != outputShape[i]) {
        return i;
      }
    }
    // If all dimensions match up to minRank, return the highest dimension
    // that exists in one but not the other
    if (inputShape.size() != outputShape.size()) {
      return std::max(static_cast<int64_t>(inputShape.size()),
                      static_cast<int64_t>(outputShape.size())) -
             1;
    }
    return -1; // Shapes are identical (shouldn't happen with valid reshape)
  }

  int64_t getLowestSlicingDim(SliceStaticOp sliceOp) const {
    // Tensor is sliced along a dimension if it doesn't have the full range,
    // which can be:
    // - begin != 0
    // - end != dimSize
    // - step != 1

    ArrayRef<int64_t> inputShape = sliceOp.getInput().getType().getShape();
    auto begins = sliceOp.getBegins().getValue();
    auto ends = sliceOp.getEnds().getValue();
    auto steps = sliceOp.getStep().getValue();

    int64_t lowestSlicingDim = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(begins.size()); ++i) {
      int32_t begin = cast<IntegerAttr>(begins[i]).getInt();
      int32_t end = cast<IntegerAttr>(ends[i]).getInt();
      int32_t step = cast<IntegerAttr>(steps[i]).getInt();
      int64_t dimSize = inputShape[i];
      int32_t adjustedEnd = (end < 0) ? (end + dimSize) : end;
      int32_t adjustedBegin = (begin < 0) ? (begin + dimSize) : begin;

      if (adjustedBegin != 0 || adjustedEnd != dimSize || step != 1) {
        lowestSlicingDim = i;
        break;
      }
    }
    return lowestSlicingDim;
  }

  SliceStaticOp reshapedSliceStaticOp(SliceStaticOp op,
                                      TypedValue<RankedTensorType> input,
                                      int64_t highestChangingDim,
                                      PatternRewriter &rewriter) const {
    // For dims higher than the highest changing dim, begins are 0, ends are
    // dimSize, step is 1.
    // We need to update ends and output shape with new dimSizes for those dims,
    // while dims at or below highest changing dim stay unchanged.
    ArrayRef<int64_t> shape = input.getType().getShape();
    auto originalEnds = op.getEnds().getValue();
    SmallVector<Attribute> newEnds;
    size_t ndims = shape.size();
    newEnds.reserve(ndims);
    for (size_t i = 0; i < ndims; ++i) {
      if (static_cast<int64_t>(i) <= highestChangingDim) {
        newEnds.push_back(
            IntegerAttr::get(IntegerType::get(op->getContext(), 32), shape[i]));
      } else {
        newEnds.push_back(originalEnds[i]);
      }
    }

    SmallVector<int64_t> newOutputShape;
    auto originalOutputShape = op.getType().getShape();
    for (size_t i = 0; i < ndims; ++i) {
      if (static_cast<int64_t>(i) <= highestChangingDim) {
        newOutputShape.push_back(shape[i]);
      } else {
        newOutputShape.push_back(originalOutputShape[i]);
      }
    }

    RankedTensorType newSliceType =
        RankedTensorType::get(newOutputShape, op.getType().getElementType(),
                              op.getType().getEncoding());

    SliceStaticOp newSlice = utils::createDPSOp<SliceStaticOp>(
        rewriter, op->getLoc(), newSliceType, input, op.getBegins(),
        rewriter.getArrayAttr(newEnds), op.getStep());

    return newSlice;
  }

  ReshapeOp deslicedReshapeOp(SliceStaticOp op, ReshapeOp reshapeUser,
                              PatternRewriter &rewriter) const {
    SmallVector<int64_t> targetShape;
    ArrayRef<int64_t> reshapeShape =
        reshapeUser.getResult().getType().getShape();
    ArrayRef<int64_t> sliceInputShape = op.getInput().getType().getShape();

    auto highestReshapeDim = getHighestReshapeDim(reshapeUser);
    size_t ndims = reshapeShape.size();
    for (size_t i = 0; i < ndims; ++i) {
      if (static_cast<int64_t>(i) <= highestReshapeDim) {
        targetShape.push_back(reshapeShape[i]);
      } else {
        targetShape.push_back(
            sliceInputShape[i]); // working only if reshape doesnt change ndim.
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
    // Reshape can commute if its highest changing dim is lower than lowest
    // slicing dim
    int64_t highestChangingDim = getHighestReshapeDim(reshapeUser);
    int64_t lowestSlicingDim = getLowestSlicingDim(op);
    if (highestChangingDim == -1 || lowestSlicingDim == -1) {
      return false;
    }
    return highestChangingDim < lowestSlicingDim;
  }

  bool isCommuteUpwardsFavorable(SliceStaticOp op, ReshapeOp) const override {
    // We should always commute a reshape above a slice if all users are an
    // identical reshape. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SliceStaticOp op,
                                ReshapeOp reshapeOperand) const override {
    // Reshape can commute if its highest changing dim is lower than lowest
    // slicing dim
    int64_t highestChangingDim = getHighestReshapeDim(reshapeOperand);
    int64_t lowestSlicingDim = getLowestSlicingDim(op);
    if (highestChangingDim == -1 || lowestSlicingDim == -1) {
      return false;
    }
    return highestChangingDim < lowestSlicingDim;
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
