// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "cstdint"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <llvm/Support/LogicalResult.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

template <typename TMOpType>
class TTIRCommuteTmsAboveElementwiseRewriter
    : public TTIRCommuteRewritePattern<TMOpType, Operation *> {
public:
  using TTIRCommuteRewritePattern<TMOpType,
                                  Operation *>::TTIRCommuteRewritePattern;

  void performCommuteRewrite(Operation *op, ArrayRef<Value> operands,
                             ArrayRef<Operation *> users,
                             PatternRewriter &rewriter) const override {
    // SmallVector<Operation *> users(op->getUsers());
    Operation *user = users[0];
    // SmallVector<Value> operands(op->getOperands());
    auto oldEltwiseType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newEltwiseType = cast<RankedTensorType>(user->getResult(0).getType())
                              .clone(oldEltwiseType.getElementType());

    SmallVector<mlir::tensor::EmptyOp> newTMDPSOperands;
    SmallVector<TMOpType> newTMs;
    SmallVector<Type> newTMResultTypes;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {

      // The new TM will have the same shape as before, but if the eltwise op
      // was a typecast, it will have the element type of the original operand
      // of the eltwise. So we need to generate a new type for the TM keeping
      // this in mind.
      auto operandType = cast<RankedTensorType>(operands[operandIdx].getType());
      auto oldTMResultType =
          cast<RankedTensorType>(user->getResult(0).getType());
      newTMResultTypes.push_back(
          oldTMResultType.clone(operandType.getElementType()));
      newTMDPSOperands.push_back(rewriter.create<tensor::EmptyOp>(
          op->getLoc(), newEltwiseType.getShape(),
          operandType.getElementType()));

      TMOpType newTM = cast<TMOpType>(rewriter.clone(*user));
      newTM->setOperand(newTM->getNumOperands() - 1,
                        newTMDPSOperands[operandIdx]);
      newTMs.push_back(newTM);
    }

    mlir::tensor::EmptyOp newEltwiseDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType());
    Operation *newEltwise = rewriter.clone(*op);

    // Do not want to put a clone on the DPS operand
    for (uint32_t operandIdx = 0; operandIdx < newEltwise->getNumOperands() - 1;
         operandIdx++) {
      newTMs[operandIdx]->setOperand(0, operands[operandIdx]);
      newTMs[operandIdx]->setOperand(1, newTMDPSOperands[operandIdx]);
      newTMs[operandIdx]->getResult(0).setType(newTMResultTypes[operandIdx]);
      newEltwise->setOperand(operandIdx, newTMs[operandIdx]->getResult(0));
    }
    newEltwise->setOperand(newEltwise->getNumOperands() - 1,
                           newEltwiseDPS->getResult(0));
    newEltwise->getResult(0).setType(newEltwiseType);

    // This only works when all the users are an identical TM
    // In the future this function may be called when this is not
    // the case, and we'll need to insert user clones on the
    // user edges that do not have an inverse on them.
    assert(succeeded(checkAllUsersAreIdenticalTms(users)) &&
           "TODO: Implement for commuting through eltewise when not all users "
           "are the same TM");
    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }
};

template <typename TMOpType>
class TTIRCommuteTmsAboveElementwiseUnaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter<TMOpType> {
public:
  using TTIRCommuteTmsAboveElementwiseRewriter<
      TMOpType>::TTIRCommuteTmsAboveElementwiseRewriter;

private:
  LogicalResult shouldCommute(Operation *op, ArrayRef<Value> operands,
                              ArrayRef<Operation *> users) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    return success(succeeded(checkAllUsersAreIdenticalTms(users)));
  }

  LogicalResult
  matchCommutePattern(Operation *op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // This pattern matches if at least one user is TMOpType and the op
    // is elementwise unary
    for (Operation *user : users) {
      if (isa<TMOpType>(user)) {
        return success(op->hasTrait<ElementwiseUnary::Trait>());
      }
    }
    return failure();
  }
};

template <typename TMOpType>
class TTIRCommuteTmsAboveElementwiseBinaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter<TMOpType> {
public:
  using TTIRCommuteTmsAboveElementwiseRewriter<
      TMOpType>::TTIRCommuteTmsAboveElementwiseRewriter;

private:
  LogicalResult shouldCommute(Operation *op, ArrayRef<Value> operands,
                              ArrayRef<Operation *> users) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    return success(succeeded(checkAllUsersAreIdenticalTms(users)));
  }

  LogicalResult
  matchCommutePattern(Operation *op, ArrayRef<Value> operands,
                      ArrayRef<Operation *> users) const override {
    // This pattern matches if at least one user is TMOpType and the op
    // is elementwise unary
    for (Operation *user : users) {
      if (isa<TMOpType>(user)) {
        return success(op->hasTrait<ElementwiseBinary::Trait>());
      }
    }
    return failure();
  }
};

namespace {

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

  // The following algorith is based upon the implementation of pytorch's
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

    SmallVector<int64_t> newReshapeShape(finalShape);

    for (uint64_t i = 0; i < finalShape.size(); i++) {
      if (finalStrides.value()[i] == 0) {
        newReshapeShape[i] = 1;
      }
    }

    SmallVector<int64_t> newBroadcastDimensions(finalShape);
    for (uint64_t i = 0; i < finalShape.size(); i++) {
      if (finalStrides.value()[i] != 0) {
        newBroadcastDimensions[i] = 1;
      }
    }

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

    if (newReshapeShape[0] == 1 && newReshapeShape[1] == 2048) {
      int x = 2;
      (void)x;
    }
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
    // 1. You can always reshape a contiguous tensor by editing the strides
    // 2. You can always broadcast a tesnsor by editing the strides
    // 3. You can NOT always reshape a tensor that is not contiguous (only
    // sometimes)
    //
    // We can always assume that the input tensor to this broadcast -> reshape
    // sequence is contiguous, since TTIR ops do not edit strides. If the
    // reshape which follows the broadcast shuffles broadcasted data into the
    // same subspace(s) as real data, then we cannot move the reshape before
    // the broadcast and get the same result; as it would be impossible to
    // have real data shuffled into the same subspace(s) as broadcasted data
    // because the broadcast operation cannot do that alone.
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

    SmallVector<int64_t> newShape(resultShape);
    SmallVector<int64_t> newBroadcastDimensions;

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
} // namespace

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;
  void runOnOperation() final {
    RewritePatternSet commutePatterns(&getContext());
    commutePatterns
        .add<TTIRCommuteTmsAboveElementwiseUnaryRewriter<ttir::TransposeOp>,
             TTIRCommuteTmsAboveElementwiseUnaryRewriter<ttir::PermuteOp>,
             TTIRCommuteTmsAboveElementwiseUnaryRewriter<ttir::ReshapeOp>,
             TTIRCommuteTmsAboveElementwiseBinaryRewriter<ttir::TransposeOp>,
             TTIRCommuteTmsAboveElementwiseBinaryRewriter<ttir::PermuteOp>,
             TTIRCommuteTmsAboveElementwiseBinaryRewriter<ttir::ReshapeOp>>(
            &getContext());
    commutePatterns.add<TTIRCommuteTransposesAboveBroadcast,
                        TTIRCommuteReshapeAboveBroadcast,
                        TTIRCommutePermuteAboveBroadcast>(&getContext());
    FrozenRewritePatternSet commutePatternSet(std::move(commutePatterns));

    // We want to commute all TMs upwards as much as possible so they are are
    // placed back to back Then we can erase back to back inverses. The
    // implemented folding patterns for TransposeOp, PermuteOp. and ReshapeOp
    // will automatically erase back to back inverses during the pass.
    // see TTIROps.cpp for the folding patterns.
    //
    // Because there are multiple TMs we wish to commute and erase, we must
    // continuously run the commute and erase patterns until the graph stops
    // changing. This is because erasing a pair of TMs may free up a path
    // for another pair of TMs to be erased.
    GreedyRewriteConfig rewriteConfig = GreedyRewriteConfig();
    bool changed = false;
    do {
      if (failed(applyPatternsGreedily(getOperation(), commutePatternSet,
                                       rewriteConfig, &changed))) {
        signalPassFailure();
        return;
      }
    } while (changed);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace mlir::tt::ttir
