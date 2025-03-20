// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Utils.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

LogicalResult checkAllUsersAreIdenticalTms(SmallVector<Operation *> users) {
  Operation *firstUser = users[0];
  for (auto *user : users) {
    if (user->getAttrDictionary() != firstUser->getAttrDictionary()) {
      return failure();
    }
  }
  return success(
      isa<ttir::TransposeOp, ttir::PermuteOp, ttir::ReshapeOp>(firstUser));
}

template <typename... TmTypes>
LogicalResult checkAtLeastOneUserIsTm(SmallVector<Operation *> users) {
  for (auto *user : users) {
    if (isa<TmTypes...>(user)) {
      return success();
    }
  }
  return failure();
}

namespace {
class TTIRCommuteTmsAboveElementwiseRewriter : public RewritePattern {
public:
  TTIRCommuteTmsAboveElementwiseRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult match(Operation *op) const override {
    if (failed(checkTrait(op))) {
      // The op should support implicit broadcast to fold them.
      return failure();
    }
    return shouldCommute(op);
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users(op->getUsers());
    if (isa<ttir::TransposeOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::TransposeOp>(op, users, op->getOperands(),
                                                  rewriter);
    } else if (isa<ttir::PermuteOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::PermuteOp>(op, users, op->getOperands(),
                                                rewriter);
    } else if (isa<ttir::ReshapeOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::ReshapeOp>(op, users, op->getOperands(),
                                                rewriter);
    } else {
      llvm_unreachable("users[0] must be one of ttir::TransposeOp, "
                       "ttir::PermuteOp, ttir::ReshapeOp");
    }
  }

  LogicalResult checkAllOperandsHaveSameShape(ValueRange operands) const {
    RankedTensorType firstOperandType =
        cast<RankedTensorType>(operands[0].getType());
    for (Value operand : operands) {
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
      if (operandType.getShape() != firstOperandType.getShape()) {
        return failure();
      }
    }
    return success();
  }

private:
  LogicalResult virtual shouldCommute(Operation *op) const {
    llvm_unreachable("shouldCommute must be overridden");
  };

  LogicalResult virtual checkTrait(Operation *op) const {
    llvm_unreachable("checkTrait must be overridden");
  };

  template <typename TMOpType>
  void commuteTmsThroughEltwise(Operation *op, SmallVector<Operation *> users,
                                ValueRange operands,
                                PatternRewriter &rewriter) const {
    Operation *user = users[0];
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
    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }
};

class TTIRCommuteTmsAboveElementwiseUnaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter {
public:
  TTIRCommuteTmsAboveElementwiseUnaryRewriter(MLIRContext *ctx)
      : TTIRCommuteTmsAboveElementwiseRewriter(ctx) {}

private:
  LogicalResult checkTrait(Operation *op) const override {
    return success(op->hasTrait<ElementwiseUnary::Trait>());
  }

  LogicalResult shouldCommute(Operation *op) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(succeeded(checkAllUsersAreIdenticalTms(users)));
  };
};

class TTIRCommuteTmsAboveElementwiseBinaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter {
public:
  TTIRCommuteTmsAboveElementwiseBinaryRewriter(MLIRContext *ctx)
      : TTIRCommuteTmsAboveElementwiseRewriter(ctx) {}

private:
  LogicalResult checkTrait(Operation *op) const override {
    return success(op->hasTrait<ElementwiseBinary::Trait>());
  }

  LogicalResult shouldCommute(Operation *op) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(succeeded(checkAllUsersAreIdenticalTms(users)));
  };
};
} // namespace

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

class TTIRCommuteTmsAboveBroadcast
    : public OpRewritePattern<ttir::BroadcastOp> {
  using OpRewritePattern<ttir::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users(op->getUsers());
    if (failed(checkAtLeastOneUserIsTm<ttir::TransposeOp, ttir::PermuteOp,
                                       ttir::ReshapeOp>(users))) {
      return failure();
    }

    Operation *originalTM = users[0];
    Value operand = op->getOperand(0);
    auto tmResultType =
        cast<RankedTensorType>(originalTM->getResult(0).getType());

    auto operandShape = cast<RankedTensorType>(operand.getType()).getShape();
    auto resultShape = tmResultType.getShape();

    SmallVector<int64_t> newShape(resultShape);
    Operation *newTM;
    SmallVector<int64_t> newBroadcastDimensions;
    if (auto transpose = dyn_cast_or_null<ttir::TransposeOp>(originalTM)) {
      newShape[transpose.getDim0()] = operandShape[transpose.getDim0()];
      newShape[transpose.getDim1()] = operandShape[transpose.getDim1()];

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
      newTM = rewriter.create<ttir::TransposeOp>(
          op->getLoc(), newTMResultType, operand, transposeDPS,
          transpose.getDim0(), transpose.getDim1());

    } else if (auto permute = dyn_cast_or_null<ttir::PermuteOp>(originalTM)) {
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
      newTM = rewriter.create<ttir::PermuteOp>(
          op->getLoc(), newTMResultType, operand, permuteDPS, permutation);

    } else if (auto reshape = dyn_cast_or_null<ttir::ReshapeOp>(originalTM)) {
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
      //

      auto originalShape = cast<RankedTensorType>(operand.getType()).getShape();
      auto broadcastShape =
          cast<RankedTensorType>(op.getResult().getType()).getShape();
      auto finalShape = resultShape;

      std::optional<SmallVector<int64_t>> finalStrides =
          getStrideAfterBroadcastReshape(originalShape, broadcastShape,
                                         finalShape);
      if (!finalStrides) {
        return rewriter.notifyMatchFailure(
            op, "Cannot commute reshape above broadcast");
      }

      SmallVector<int64_t> newReshapeShape(finalShape);

      for (uint64_t i = 0; i < finalShape.size(); i++) {
        if (finalStrides.value()[i] == 0) {
          newReshapeShape[i] = 1;
        }
      }

      SmallVector<int64_t> newBroadcastDims(finalShape);
      for (uint64_t i = 0; i < finalShape.size(); i++) {
        if (finalStrides.value()[i] != 0) {
          newBroadcastDims[i] = 1;
        }
      }

      auto newTMResultType = tmResultType.cloneWith(
          newReshapeShape, tmResultType.getElementType());

      auto reshapeDps = rewriter.create<tensor::EmptyOp>(
          op->getLoc(), newTMResultType.getShape(),
          newTMResultType.getElementType());

      newTM = rewriter.create<ttir::ReshapeOp>(
          op->getLoc(), newTMResultType, operand, reshapeDps,
          rewriter.getI32ArrayAttr(SmallVector<int32_t>(
              newReshapeShape.begin(), newReshapeShape.end())));

      newBroadcastDimensions = newBroadcastDims;

    } else {
      llvm_unreachable("newTM must be one of ttir::TransposeOp, "
                       "ttir::PermuteOp, ttir::ReshapeOp");
    }

    auto broadcastDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), resultShape, tmResultType.getElementType());
    assert(newBroadcastDimensions.size() <= (uint32_t)tmResultType.getRank());
    auto newBroadcast = rewriter.create<ttir::BroadcastOp>(
        op->getLoc(), tmResultType, newTM->getResult(0), broadcastDPS,
        newBroadcastDimensions);

    rewriter.replaceOp(originalTM, newBroadcast);
    return success();
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
    commutePatterns.add<ttir::TTIRCommuteTmsAboveElementwiseUnaryRewriter>(
        &getContext());
    commutePatterns.add<ttir::TTIRCommuteTmsAboveElementwiseBinaryRewriter>(
        &getContext());
    commutePatterns.add<TTIRCommuteTmsAboveBroadcast>(&getContext());
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
