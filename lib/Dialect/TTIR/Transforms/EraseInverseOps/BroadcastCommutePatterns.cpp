// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::tt::ttir {

SmallVector<int64_t> getContiguousStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

SmallVector<int64_t> getStrideAfterBroadcast(ArrayRef<int64_t> originalShape,
                                             ArrayRef<int64_t> broadcastShape) {
  SmallVector<int64_t> strides = getContiguousStrides(originalShape);
  SmallVector<int64_t> newStrides(originalShape.size(), 0);
  for (int64_t i = originalShape.size() - 1; i >= 0; i--) {
    if (originalShape[i] == broadcastShape[i]) {
      newStrides[i] = strides[i];
    } else {
      newStrides[i] = 0;
    }
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
    ttir::TransposeOp newTranspose;
    SmallVector<int64_t> newBroadcastDimensions;

    // Commuting a transpose above a broadcast requires us to swap the broadcast
    // dimensions according to the transpose dimensions.
    for (int32_t i = 0;
         i < static_cast<int32_t>(op.getBroadcastDimensions().size()); i++) {
      if (i == transpose.getDim0()) {
        newBroadcastDimensions.push_back(
            op.getBroadcastDimensions()[transpose.getDim1()]);
      } else if (i == transpose.getDim1()) {
        newBroadcastDimensions.push_back(
            op.getBroadcastDimensions()[transpose.getDim0()]);
      } else {
        newBroadcastDimensions.push_back(op.getBroadcastDimensions()[i]);
      }
    }

    auto newTMResultType =
        tmResultType.cloneWith(newShape, tmResultType.getElementType());

    auto transposeDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newTMResultType.getShape(),
        newTMResultType.getElementType());
    newTranspose = rewriter.create<ttir::TransposeOp>(
        op->getLoc(), newTMResultType, operand, transposeDPS,
        transpose.getDim0(), transpose.getDim1());

    auto broadcastDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), resultShape, tmResultType.getElementType());
    assert(newBroadcastDimensions.size() <= (uint32_t)tmResultType.getRank());
    auto newBroadcast = rewriter.create<ttir::BroadcastOp>(
        op->getLoc(), tmResultType, newTranspose, broadcastDPS,
        newBroadcastDimensions);

    rewriter.replaceOp(transpose, newBroadcast);
  }

private:
  LogicalResult
  matchCommutePattern(ttir::BroadcastOp op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // We will match a broacast -> transpose sequence if at least one user is a
    // transpose
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
    auto broadcastShape =
        cast<RankedTensorType>(op.getResult().getType()).getShape();

    auto tmResultType = cast<RankedTensorType>(reshape->getResult(0).getType());

    auto resultShape = tmResultType.getShape();
    auto finalShape = resultShape;

    std::optional<SmallVector<int64_t>> finalStrides =
        getStrideAfterBroadcastReshape(originalShape, broadcastShape,
                                       finalShape);

    assert(finalStrides.has_value() &&
           "matchCommutePattern should have ensured that this is possible.");

    // All dimensions with stride 0 will be broadcasted. So the new reshape
    // should have the same shape as the desired output - except with all
    // broadcasted dimensions as `1`.
    SmallVector<int64_t> newReshapeShape(finalShape);
    for (uint64_t i = 0; i < finalShape.size(); i++) {
      if (finalStrides.value()[i] == 0) {
        newReshapeShape[i] = 1;
      }
    }

    // The broadcasted dimensions should all be `1` except for the dimensions
    // which we actually want to broadcast.
    SmallVector<int64_t> newBroadcastDimensions(finalShape);
    for (uint64_t i = 0; i < finalShape.size(); i++) {
      if (finalStrides.value()[i] != 0) {
        newBroadcastDimensions[i] = 1;
      }
    }

    // Now that we know which shape the reshape should have and which broadcast
    // dimensions the broadcast should have, we can generate the new ops.
    auto newTMResultType =
        tmResultType.cloneWith(newReshapeShape, tmResultType.getElementType());

    auto reshapeDps = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newTMResultType.getShape(),
        newTMResultType.getElementType());

    auto newReshape = rewriter.create<ttir::ReshapeOp>(
        op->getLoc(), newTMResultType, operand, reshapeDps,
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(newReshapeShape.begin(),
                                                      newReshapeShape.end())));

    auto broadcastDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), resultShape, tmResultType.getElementType());
    assert(newBroadcastDimensions.size() <= (uint32_t)tmResultType.getRank());
    auto newBroadcast = rewriter.create<ttir::BroadcastOp>(
        op->getLoc(), tmResultType, newReshape, broadcastDPS,
        newBroadcastDimensions);

    rewriter.replaceOp(reshape, newBroadcast);
  }

private:
  LogicalResult
  matchCommutePattern(ttir::BroadcastOp op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // We will match a broacast -> reshape sequence if at least one user is a
    // reshape There are some cases where the specific reshape cannot be
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
    // same subspace(s) as real data, then we cannot move the reshape before
    // the broadcast and get the same result. This is because the broadcast
    // op does not have the ability to shuffle real data into broadcasted data.
    // If our desired result requires us to do that, then the broadcast cannot
    // come after the reshape.
    //
    // We can check if the reshape can commute above the broadcast by checking
    // whether or not the reshape can be done by editing strides alone (point
    // #3).

    for (Operation *user : users) {
      if (auto reshape = dyn_cast_or_null<ttir::ReshapeOp>(user)) {

        Value operand = operands[0];
        auto tmResultType =
            cast<RankedTensorType>(reshape->getResult(0).getType());

        auto resultShape = tmResultType.getShape();

        auto originalShape =
            cast<RankedTensorType>(operand.getType()).getShape();
        auto broadcastShape =
            cast<RankedTensorType>(op.getResult().getType()).getShape();
        auto finalShape = resultShape;

        std::optional<SmallVector<int64_t>> finalStrides =
            getStrideAfterBroadcastReshape(originalShape, broadcastShape,
                                           finalShape);
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
    auto tmResultType = cast<RankedTensorType>(permute->getResult(0).getType());

    auto resultShape = tmResultType.getShape();

    SmallVector<int64_t> newShape(operandShape);
    SmallVector<int64_t> newBroadcastDimensions;

    // Commuting a broadcast above a permute requires us to permute which dims
    // are broadcasted.
    auto permutation = permute.getPermutation();
    for (int64_t i = 0; i < static_cast<int64_t>(permutation.size()); i++) {
      newShape[i] = operandShape[permutation[i]];
    }

    for (uint32_t i = 0; i < op.getBroadcastDimensions().size(); i++) {
      newBroadcastDimensions.push_back(
          op.getBroadcastDimensions()[permutation[i]]);
    }

    auto newTMResultType =
        tmResultType.cloneWith(newShape, tmResultType.getElementType());

    auto permuteDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newTMResultType.getShape(),
        newTMResultType.getElementType());
    auto newPermute = rewriter.create<ttir::PermuteOp>(
        op->getLoc(), newTMResultType, operand, permuteDPS, permutation);

    auto broadcastDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), resultShape, tmResultType.getElementType());
    assert(newBroadcastDimensions.size() <= (uint32_t)tmResultType.getRank());
    auto newBroadcast = rewriter.create<ttir::BroadcastOp>(
        op->getLoc(), tmResultType, newPermute, broadcastDPS,
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
