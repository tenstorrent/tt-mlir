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
  // %1 = permute(%0) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  void performCommuteUpwardsRewrite(SliceStaticOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    auto perm = permuteUser.getPermutation();
    RankedTensorType inputType = op.getInput().getType();

    // Create permute on the unsliced input.
    RankedTensorType newPermuteType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(inputType.getShape(), perm),
        inputType.getElementType(), inputType.getEncoding());
    PermuteOp newPerm = rewriter.create<PermuteOp>(
        permuteUser->getLoc(), newPermuteType, op.getInput(), perm);

    // Create slice with permuted parameters.
    auto newBegins =
        ttmlir::utils::applyPermutation(op.getBegins().getValue(), perm);
    auto newEnds =
        ttmlir::utils::applyPermutation(op.getEnds().getValue(), perm);
    auto newSteps =
        ttmlir::utils::applyPermutation(op.getStep().getValue(), perm);

    RankedTensorType newSliceType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(), perm),
        op.getType().getElementType(), op.getType().getEncoding());
    SliceStaticOp newSlice = rewriter.create<SliceStaticOp>(
        op->getLoc(), newSliceType, newPerm, rewriter.getArrayAttr(newBegins),
        rewriter.getArrayAttr(newEnds), rewriter.getArrayAttr(newSteps));

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      rewriter.replaceOp(user, newSlice);
    }
  }

  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  //
  // This method will transform this into:
  // %0 = slice(%arg0) <{
  //      begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = permute(%0) <{permutation = array<i64: 0, 2, 3, 1>}>
  void
  performCommuteDownwardsRewrite(SliceStaticOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    auto perm = permuteOperand.getPermutation();
    auto invPerm = ttmlir::utils::inversePermutation(perm);

    // Create slice on the unpermuted input.
    auto newBegins =
        ttmlir::utils::applyPermutation(op.getBegins().getValue(), invPerm);
    auto newEnds =
        ttmlir::utils::applyPermutation(op.getEnds().getValue(), invPerm);
    auto newSteps =
        ttmlir::utils::applyPermutation(op.getStep().getValue(), invPerm);

    RankedTensorType newSliceType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(), invPerm),
        op.getType().getElementType(), op.getType().getEncoding());
    SliceStaticOp newSlice = rewriter.create<SliceStaticOp>(
        op->getLoc(), newSliceType, permuteOperand.getInput(),
        rewriter.getArrayAttr(newBegins), rewriter.getArrayAttr(newEnds),
        rewriter.getArrayAttr(newSteps));

    // Restore the permutation after the slice.
    PermuteOp newPerm = rewriter.create<PermuteOp>(
        permuteOperand->getLoc(), op.getType(), newSlice, perm);

    rewriter.replaceOp(op, newPerm);
  }

private:
  bool isCommuteUpwardsViable(SliceStaticOp, PermuteOp) const override {
    return true;
  }

  bool isCommuteUpwardsFavorable(SliceStaticOp op, PermuteOp) const override {
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SliceStaticOp, PermuteOp) const override {
    return true;
  }

  bool isCommuteDownwardsFavorable(SliceStaticOp, PermuteOp) const override {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Reshape through Slice
//===----------------------------------------------------------------------===//

template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughSlice
    : public TTIRCommuteOpRewritePattern<ReshapeOp, SliceStaticOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, SliceStaticOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // %0 = slice(%arg0 : tensor<1x160x160x128xbf16>) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = reshape(%0) <{shape = [1 : i32, 1 : i32, 25600 : i32, 64 : i32]}>
  //
  // This method will transform this into:
  // %0 = reshape(%arg0 : tensor<1x160x160x128xbf16>)
  //      <{shape = [1 : i32, 1 : i32, 25600 : i32, 128 : i32]}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 1 : i32, 25600 : i32, 64 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  //
  // Uses findMatchingDimRTL to map each sliced dimension through the reshape.
  void performCommuteUpwardsRewrite(SliceStaticOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto dimMap = computeDimMapping(op, reshapeUser);
    auto sliceInput = op.getInput();
    auto reshapeShape = reshapeUser.getResult().getType().getShape();

    // Create reshape on unsliced input - use original input dims for mapped
    // dims.
    SmallVector<int64_t> newReshapeShape(reshapeShape);
    for (auto [sliceDim, reshapeDim] : dimMap) {
      newReshapeShape[reshapeDim] = sliceInput.getType().getShape()[sliceDim];
    }

    ReshapeOp newReshape = rewriter.create<ReshapeOp>(
        reshapeUser->getLoc(),
        RankedTensorType::get(newReshapeShape,
                              sliceInput.getType().getElementType(),
                              sliceInput.getType().getEncoding()),
        sliceInput,
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(newReshapeShape.begin(),
                                                      newReshapeShape.end())));

    // Create slice with mapped dimensions.
    SliceStaticOp newSlice =
        createMappedSlice(op, dimMap, newReshape.getResult(), rewriter);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      rewriter.replaceOp(user, newSlice);
    }
  }

  // Consider the following IR pseudocode:
  // %0 = reshape(%arg0 : tensor<1x1x25600x128xbf16>)
  //      <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  //
  // This method will transform this into:
  // %0 = slice(%arg0 : tensor<1x1x25600x128xbf16>) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 1 : i32, 25600 : i32, 64 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = reshape(%0) <{shape = [1 : i32, 160 : i32, 160 : i32, 64 : i32]}>
  void
  performCommuteDownwardsRewrite(SliceStaticOp op, ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    auto dimMap = computeInverseDimMapping(op, reshapeOperand);

    SliceStaticOp newSlice =
        createMappedSlice(op, dimMap, reshapeOperand.getInput(), rewriter);

    ReshapeOp newReshape = rewriter.create<ReshapeOp>(
        reshapeOperand->getLoc(), op.getType(), newSlice,
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(
            op.getType().getShape().begin(), op.getType().getShape().end())));

    rewriter.replaceOp(op, newReshape);
  }

private:
  // Returns true if the dimension has non-trivial slicing.
  bool isDimSliced(SliceStaticOp op, int64_t dim) const {
    auto shape = op.getInput().getType().getShape();
    int32_t begin = cast<IntegerAttr>(op.getBegins()[dim]).getInt();
    int32_t end = cast<IntegerAttr>(op.getEnds()[dim]).getInt();
    int32_t step = cast<IntegerAttr>(op.getStep()[dim]).getInt();
    int64_t size = shape[dim];

    int32_t adjBegin = begin < 0 ? begin + size : begin;
    int32_t adjEnd = end < 0 ? end + size : end;
    return adjBegin != 0 || adjEnd != size || step != 1;
  }

  // For upwards commute: maps sliced dims to their positions in reshape output.
  // Returns pairs of (sliceDim, reshapeOutputDim).
  SmallVector<std::pair<int64_t, int64_t>>
  computeDimMapping(SliceStaticOp op, ReshapeOp reshape) const {
    SmallVector<std::pair<int64_t, int64_t>> mapping;
    int64_t inRank = op.getInput().getType().getRank();
    int64_t outRank = reshape.getResult().getType().getRank();

    for (int64_t dim = 0; dim < inRank; ++dim) {
      if (!isDimSliced(op, dim)) {
        continue;
      }
      int64_t dimRTL = inRank - 1 - dim;
      int64_t outRTL = utils::findMatchingDimRTL(reshape, dimRTL);
      if (outRTL != -1) {
        mapping.emplace_back(dim, outRank - 1 - outRTL);
      }
    }
    return mapping;
  }

  // For downwards commute: maps sliced dims to their positions in reshape
  // input.
  SmallVector<std::pair<int64_t, int64_t>>
  computeInverseDimMapping(SliceStaticOp op, ReshapeOp reshape) const {
    SmallVector<std::pair<int64_t, int64_t>> mapping;
    int64_t sliceRank = op.getInput().getType().getRank();
    int64_t inRank = reshape.getInput().getType().getRank();

    for (int64_t dim = 0; dim < sliceRank; ++dim) {
      if (!isDimSliced(op, dim)) {
        continue;
      }
      int64_t dimRTL = sliceRank - 1 - dim;
      // Find input dim that maps to this output dim.
      for (int64_t inDim = 0; inDim < inRank; ++inDim) {
        int64_t inRTL = inRank - 1 - inDim;
        if (utils::findMatchingDimRTL(reshape, inRTL) == dimRTL) {
          mapping.emplace_back(dim, inDim);
          break;
        }
      }
    }
    return mapping;
  }

  // Creates a slice op with parameters mapped to the new input shape.
  SliceStaticOp createMappedSlice(SliceStaticOp op,
                                  ArrayRef<std::pair<int64_t, int64_t>> dimMap,
                                  TypedValue<RankedTensorType> input,
                                  PatternRewriter &rewriter) const {
    auto shape = input.getType().getShape();
    int64_t rank = shape.size();

    // Initialize with identity slice (no slicing).
    SmallVector<int32_t> begins(rank), ends(rank), steps(rank);
    SmallVector<int64_t> outShape(rank);
    for (int64_t i = 0; i < rank; ++i) {
      begins[i] = 0;
      ends[i] = shape[i];
      steps[i] = 1;
      outShape[i] = shape[i];
    }

    // Apply original slicing parameters to mapped dimensions.
    for (auto [srcDim, dstDim] : dimMap) {
      begins[dstDim] = cast<IntegerAttr>(op.getBegins()[srcDim]).getInt();
      ends[dstDim] = cast<IntegerAttr>(op.getEnds()[srcDim]).getInt();
      steps[dstDim] = cast<IntegerAttr>(op.getStep()[srcDim]).getInt();
      outShape[dstDim] = op.getType().getShape()[srcDim];
    }

    return rewriter.create<SliceStaticOp>(
        op->getLoc(),
        RankedTensorType::get(outShape, op.getType().getElementType(),
                              op.getType().getEncoding()),
        input, rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
        rewriter.getI32ArrayAttr(steps));
  }

  // Checks if all sliced dimensions can be mapped through the reshape.
  bool canCommute(SliceStaticOp op, ReshapeOp reshape, bool isUpwards) const {
    auto mapping = isUpwards ? computeDimMapping(op, reshape)
                             : computeInverseDimMapping(op, reshape);
    int64_t slicedCount = 0;
    int64_t rank = op.getInput().getType().getRank();
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (isDimSliced(op, dim)) {
        ++slicedCount;
      }
    }
    return static_cast<int64_t>(mapping.size()) == slicedCount;
  }

  bool isCommuteUpwardsViable(SliceStaticOp op,
                              ReshapeOp reshapeUser) const override {
    return canCommute(op, reshapeUser, /*isUpwards=*/true);
  }

  bool isCommuteUpwardsFavorable(SliceStaticOp op, ReshapeOp) const override {
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SliceStaticOp op,
                                ReshapeOp reshapeOperand) const override {
    return canCommute(op, reshapeOperand, /*isUpwards=*/false);
  }

  bool isCommuteDownwardsFavorable(SliceStaticOp, ReshapeOp) const override {
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
