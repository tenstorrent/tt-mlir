// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::tt::ttir {

LogicalResult checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  if (users.size() == 0) {
    return failure();
  }

  Operation *firstUser = users[0];
  for (auto *user : users) {
    if (user->getAttrDictionary() != firstUser->getAttrDictionary()) {
      return failure();
    }
  }
  return success(isa<TransposeOp, PermuteOp, ReshapeOp>(firstUser));
}

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

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns) {
  patterns.add<TTIRCommuteTmsAboveElementwiseUnaryRewriter<TransposeOp>,
               TTIRCommuteTmsAboveElementwiseUnaryRewriter<PermuteOp>,
               TTIRCommuteTmsAboveElementwiseUnaryRewriter<ReshapeOp>,
               TTIRCommuteTmsAboveElementwiseBinaryRewriter<TransposeOp>,
               TTIRCommuteTmsAboveElementwiseBinaryRewriter<PermuteOp>,
               TTIRCommuteTmsAboveElementwiseBinaryRewriter<ReshapeOp>>(ctx);
}

} // namespace mlir::tt::ttir
