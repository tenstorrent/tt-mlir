// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Utils.h"
#include <mlir/IR/BuiltinTypes.h>

namespace mlir::tt::ttir {

static SmallVector<int64_t> getContiguousStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

// This function calculates the theoretical strides of a tensor after it has
// been broadcasted. Put simply - if a dimension is broadcasted, the stride
// of that dimension becomes 0 as you do not have to travel along the buffer
// which stores the tensor data to retrieve the next segment of data along
// that dimension. All other strides remain the same.
static SmallVector<int64_t>
getStrideAfterBroadcast(ArrayRef<int64_t> originalShape,
                        ArrayRef<int64_t> broadcastShape) {
  SmallVector<int64_t> strides = getContiguousStrides(originalShape);
  SmallVector<int64_t> newStrides(originalShape.size(), 0);
  for (uint64_t i = 0; i < originalShape.size(); i++) {
    // Want to set the stride to 0 if the boolean expression is false,
    // strides[i] if true.
    newStrides[i] = (originalShape[i] == broadcastShape[i]) * strides[i];
  }
  return newStrides;
}

std::optional<SmallVector<int64_t>>
getStrideAfterBroadcastReshape(ArrayRef<int64_t> originalShape,
                               ArrayRef<int64_t> broadcastShape,
                               ArrayRef<int64_t> finalShape) {
  auto stridesAfterBroadcast =
      getStrideAfterBroadcast(originalShape, broadcastShape);

  // The following algorithm is based upon the implementation of pytorch's
  // `view` op implementation. Specifically the helper that computes the new
  // stride.
  //
  // Source:
  // https://github.com/pytorch/pytorch/blob/842a072fd3d219aca538435d4e956053e76817df/aten/src/ATen/TensorUtils.cpp#L364
  SmallVector<int64_t> newStrides(finalShape.size(), 0);

  int64_t viewD = finalShape.size() - 1;
  int64_t tensorNumel = 1;
  int64_t viewNumel = 1;
  int64_t chunkBaseStride = stridesAfterBroadcast.back();

  for (int64_t tensorD = broadcastShape.size() - 1; tensorD >= 0; tensorD--) {
    tensorNumel *= broadcastShape[tensorD];
    if (tensorD == 0 ||
        (broadcastShape[tensorD - 1] != 1 &&
         stridesAfterBroadcast[tensorD - 1] != tensorNumel * chunkBaseStride)) {

      while (viewD >= 0 &&
             (viewNumel < tensorNumel || finalShape[viewD] == 1)) {
        newStrides[viewD] = viewNumel * chunkBaseStride;
        viewNumel *= finalShape[viewD];
        viewD--;
      }

      if (viewNumel != tensorNumel) {
        return std::nullopt;
      }

      if (tensorD > 0) {
        chunkBaseStride = stridesAfterBroadcast[tensorD - 1];
        tensorNumel = 1;
        viewNumel = 1;
      }
    }
  }
  if (viewD != -1) {
    return std::nullopt;
  }
  return newStrides;
}

class TTIRCommuteTransposesAboveBroadcast
    : public TTIRCommuteRewritePattern<ttir::TransposeOp, ttir::BroadcastOp> {
public:
  using TTIRCommuteRewritePattern<ttir::TransposeOp,
                                  ttir::BroadcastOp>::TTIRCommuteRewritePattern;

  void performCommuteRewrite(ttir::BroadcastOp op, ArrayRef<Value> operands,
                             ArrayRef<Operation *> users,
                             PatternRewriter &rewriter) const override {

    auto transpose = cast<ttir::TransposeOp>(users[0]);
    Value operand = operands[0];
    auto tmResultType =
        cast<RankedTensorType>(transpose->getResult(0).getType());

    auto resultShape = tmResultType.getShape();

    SmallVector<int64_t> newShape(resultShape);

    // Commuting a transpose above a broadcast requires us to swap the broadcast
    // dimensions according to the transpose dimensions.
    SmallVector<int64_t> newBroadcastDimensions(op.getBroadcastDimensions());
    std::swap(newBroadcastDimensions[transpose.getDim0()],
              newBroadcastDimensions[transpose.getDim1()]);

    auto newTranspose = ttmlir::utils::createDPSOp<ttir::TransposeOp>(
        rewriter, op->getLoc(), newShape, tmResultType.getElementType(),
        tmResultType.getEncoding(), operand, transpose.getDim0(),
        transpose.getDim1());

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));

    auto newBroadcast = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newTranspose,
        newBroadcastDimensions);

    rewriter.replaceOp(transpose, newBroadcast);
  }

private:
  LogicalResult
  matchCommutePattern(ttir::BroadcastOp op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // We will match a broacast -> transpose sequence if at least one user is a
    // transpose.
    for (Operation *user : users) {
      if (isa<ttir::TransposeOp>(user)) {
        return success();
      }
    }
    return failure();
  }

  LogicalResult shouldCommute(ttir::BroadcastOp op, ArrayRef<Value> operands,
                              ArrayRef<Operation *> users) const override {
    // We should always commute a transpose above a broadcast if it is the only
    // user of the broadcast. For now this is the only case we will handle.
    // matchCommutePattern will have already confirmed that this user is a
    // transpose and it can be commuted above the broadcast.
    return success(users.size() == 1);
  }
};

class TTIRCommuteReshapeAboveBroadcast
    : public TTIRCommuteRewritePattern<ttir::ReshapeOp, ttir::BroadcastOp> {
public:
  using TTIRCommuteRewritePattern<ttir::ReshapeOp,
                                  ttir::BroadcastOp>::TTIRCommuteRewritePattern;
  void performCommuteRewrite(ttir::BroadcastOp op, ArrayRef<Value> operands,
                             ArrayRef<Operation *> users,
                             PatternRewriter &rewriter) const override {

    Value operand = operands[0];
    auto reshape = cast<ttir::ReshapeOp>(users[0]);
    auto originalShape = cast<RankedTensorType>(operand.getType()).getShape();
    auto broadcastShape = op.getResult().getType().getShape();

    auto tmResultType = reshape.getResult().getType();

    auto resultShape = tmResultType.getShape();

    std::optional<SmallVector<int64_t>> finalStrides =
        getStrideAfterBroadcastReshape(originalShape, broadcastShape,
                                       resultShape);

    assert(finalStrides.has_value() &&
           "matchCommutePattern should have ensured that this is possible.");

    // All dimensions with stride 0 will be broadcasted. So the new reshape
    // should have the same shape as the desired output - except with all
    // broadcasted dimensions as `1`.
    //
    // The broadcasted dimensions should all be `1` except for the dimensions
    // which we actually want to broadcast.
    SmallVector<int64_t> newReshapeShape(resultShape);
    SmallVector<int64_t> newBroadcastDimensions(resultShape);
    for (uint64_t i = 0; i < resultShape.size(); i++) {
      if (finalStrides.value()[i] == 0) {
        newReshapeShape[i] = 1;
      } else {
        newBroadcastDimensions[i] = 1;
      }
    }

    // Now that we know which shape the reshape should have and which broadcast
    // dimensions the broadcast should have, we can generate the new ops.
    auto newTMResultType =
        RankedTensorType::get(newReshapeShape, tmResultType.getElementType(),
                              tmResultType.getEncoding());

    auto newReshape = ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, op->getLoc(), newTMResultType, operand,
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(newReshapeShape.begin(),
                                                      newReshapeShape.end())));

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));
    auto newBroadcast = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newReshape,
        newBroadcastDimensions);

    rewriter.replaceOp(reshape, newBroadcast);
  }

private:
  LogicalResult
  matchCommutePattern(ttir::BroadcastOp op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // We will match a broacast -> reshape sequence if at least one user is a
    // reshape. There are some cases where the specific reshape cannot be
    // commuted above a specific broadcast, so we must check if it is possible
    // here.

    // The following points are true about about reshaping and broadcasting
    // tensors that have stride attributes:
    //
    // 1. You can always reshape a contiguous tensor by editing the strides.
    // 2. You can always broadcast a tensor (contiguous or not) by editing
    // the strides.
    // 3. You can only SOMETIMES reshape a non-contiguous tensor by editing
    // the strides alone.
    //
    // We can always assume that the input tensor to this broadcast -> reshape
    // sequence is contiguous, since TTIR ops do not edit strides. If the
    // reshape which follows the broadcast shuffles broadcasted data into the
    // same axes as original data, then we cannot move the reshape before
    // the broadcast and get the same result. This is because the broadcast
    // op does not have the ability to shuffle original data into broadcasted
    // data. If our desired result requires us to do that, then the broadcast
    // cannot come after the reshape.
    //
    // We can check if the reshape can commute above the broadcast by checking
    // whether or not the reshape can be done by editing strides alone (point
    // #3).

    for (Operation *user : users) {
      if (auto reshape = dyn_cast<ttir::ReshapeOp>(user)) {

        Value operand = operands[0];
        auto tmResultType = reshape.getResult().getType();

        auto resultShape = tmResultType.getShape();

        auto originalShape =
            cast<RankedTensorType>(operand.getType()).getShape();
        auto broadcastShape = op.getResult().getType().getShape();

        std::optional<SmallVector<int64_t>> finalStrides =
            getStrideAfterBroadcastReshape(originalShape, broadcastShape,
                                           resultShape);
        // finalStrides will be nullopt if the broadcast -> reshape sequence
        // cannot be performed by editing strides alone. This means we cannot
        // commute the reshape above the broadcast either.
        if (!finalStrides) {
          // We want to continue in case another user reshape which
          // can commute above the broadcast is found.
          continue;
        }

        // A reshape which can commute above the broadcast has been found.
        return success();
      }
    }

    return failure();
  }

  LogicalResult shouldCommute(ttir::BroadcastOp op, ArrayRef<Value> operands,
                              ArrayRef<Operation *> users) const override {
    // We should always commute a reshape above a broadcast if it is the only
    // user of the broadcast. For now we only handle this case.
    // matchCommutePattern will have already confirmed that this user is a
    // reshape and it can be commuted above the broadcast.
    return success(users.size() == 1);
  }
};

class TTIRCommutePermuteAboveBroadcast
    : public TTIRCommuteRewritePattern<ttir::PermuteOp, ttir::BroadcastOp> {
public:
  using TTIRCommuteRewritePattern<ttir::PermuteOp,
                                  ttir::BroadcastOp>::TTIRCommuteRewritePattern;

  void performCommuteRewrite(ttir::BroadcastOp op, ArrayRef<Value> operands,
                             ArrayRef<Operation *> users,
                             PatternRewriter &rewriter) const override {
    auto permute = cast<ttir::PermuteOp>(users[0]);
    Value operand = operands[0];
    auto operandShape = cast<RankedTensorType>(operand.getType()).getShape();
    auto tmResultType = permute.getResult().getType();

    // Commuting a broadcast above a permute requires us to permute which dims
    // are broadcasted.
    auto permutation = permute.getPermutation();
    SmallVector<int64_t> newShape =
        ttmlir::utils::applyPermutation(operandShape, permutation);
    SmallVector<int64_t> newBroadcastDimensions =
        ttmlir::utils::applyPermutation(op.getBroadcastDimensions(),
                                        permutation);

    auto newTMResultType = RankedTensorType::get(
        newShape, tmResultType.getElementType(), tmResultType.getEncoding());
    auto newPermute = ttmlir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, op->getLoc(), newTMResultType, operand, permutation);

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));
    auto newBroadcast = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newPermute,
        newBroadcastDimensions);

    rewriter.replaceOp(permute, newBroadcast);
  }

private:
  LogicalResult
  matchCommutePattern(ttir::BroadcastOp op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // We will match a broacast -> permute sequence if at least one user is a
    // permute
    for (Operation *user : users) {
      if (isa<ttir::PermuteOp>(user)) {
        return success();
      }
    }
    return failure();
  }

  LogicalResult shouldCommute(ttir::BroadcastOp op, ArrayRef<Value> operands,
                              ArrayRef<Operation *> users) const override {
    // We should always commute a permute above a broadcast if it is the only
    // user of the broadcast. For now this is the only case we will handle.
    // matchCommutePattern will have already confirmed that this user is a
    // permute and it can be commuted above the broadcast.
    return success(users.size() == 1);
  }
};

void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  patterns
      .add<TTIRCommuteTransposesAboveBroadcast,
           TTIRCommuteReshapeAboveBroadcast, TTIRCommutePermuteAboveBroadcast>(
          ctx);
}

} // namespace mlir::tt::ttir
