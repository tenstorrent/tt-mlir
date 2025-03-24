// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Utils.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttir {

static LogicalResult checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  if (users.size() == 0) {
    return success();
  }

  Operation *firstUser = users[0];
  for (auto *user : users) {
    if (user->getAttrDictionary() != firstUser->getAttrDictionary()) {
      return failure();
    }
  }
  return success(firstUser->hasTrait<ttir::TensorManipulation::Trait>());
}
namespace {
template <typename TMOpType, typename ElementwiseInterfaceType>
class TTIRCommuteTmsAboveElementwiseRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<TMOpType,
                                                  ElementwiseInterfaceType> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      TMOpType, ElementwiseInterfaceType>::TTIRCommuteOpInterfaceRewritePattern;

  void performCommuteRewrite(Operation *op, TMOpType tmUser,
                             PatternRewriter &rewriter) const override {

    auto eltwise = cast<ElementwiseInterfaceType>(op);
    llvm::SmallVector<Operation *> users(eltwise->getUsers());
    auto oldEltwiseType =
        cast<RankedTensorType>(eltwise->getResult(0).getType());
    auto oldTMResultType = tmUser.getResult().getType();

    auto newEltwiseType = RankedTensorType::get(oldTMResultType.getShape(),
                                                oldEltwiseType.getElementType(),
                                                oldTMResultType.getEncoding());

    SmallVector<mlir::tensor::EmptyOp> newTMDPSOperands;
    SmallVector<Value> newEltwiseOperands;
    SmallVector<RankedTensorType> newTMResultTypes;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {

      // The new TM will have the same shape as before, but if the eltwise op
      // was a typecast, it will have the element type of the original operand
      // of the eltwise. So we need to generate a new type for the TM keeping
      // this in mind.
      auto operandType =
          cast<RankedTensorType>(eltwise->getOperand(operandIdx).getType());

      newTMResultTypes.push_back(
          oldTMResultType.clone(operandType.getElementType()));

      auto dpsOperand = rewriter.create<tensor::EmptyOp>(
          op->getLoc(), newEltwiseType.getShape(),
          operandType.getElementType());
      newTMDPSOperands.push_back(dpsOperand);

      auto newTM = rewriter.create<TMOpType>(
          op->getLoc(), newTMResultTypes[operandIdx],
          ValueRange({eltwise->getOperand(operandIdx), dpsOperand}),
          tmUser->getAttrs());

      newEltwiseOperands.push_back(newTM);
    }

    newEltwiseOperands.push_back(rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType()));

    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, newEltwiseType, op->getAttrs());

    // This only works when all the users are an identical TM
    // In the future this function may be called when this is not
    // the case, and we'll need to insert user clones on the
    // user edges that do not have an inverse on them.
    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }
};
} // namespace

namespace {
template <typename TMOpType>
class TTIRCommuteTmsAboveElementwiseUnaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter<TMOpType,
                                                    ElementwiseUnary> {
public:
  using TTIRCommuteTmsAboveElementwiseRewriter<
      TMOpType, ElementwiseUnary>::TTIRCommuteTmsAboveElementwiseRewriter;

private:
  LogicalResult shouldCommute(Operation *op, TMOpType) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(users.size() > 0 &&
                   succeeded(checkAllUsersAreIdenticalTms(users)));
  }

  LogicalResult matchCommutePattern(Operation *op,
                                    TMOpType tmUser) const override {
    // We can always commute a TM above an elementwise op
    return success();
  }
};
} // namespace

namespace {
template <typename TMOpType>
class TTIRCommuteTmsAboveElementwiseBinaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter<TMOpType,
                                                    ElementwiseBinary> {
public:
  using TTIRCommuteTmsAboveElementwiseRewriter<
      TMOpType, ElementwiseBinary>::TTIRCommuteTmsAboveElementwiseRewriter;

private:
  LogicalResult shouldCommute(Operation *op, TMOpType) const override {
    // In some cases there may be an implicit broadcast on one of the operands.
    // That is there is no broadcast op on one of the operands but a broadcast
    // is required to execute the op nonetheless. We do not handle this yet. So
    // we will make sure the operand types are identical.
    auto firstOperandType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto secondOperandType =
        cast<RankedTensorType>(op->getOperand(1).getType());

    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(firstOperandType == secondOperandType && users.size() > 0 &&
                   succeeded(checkAllUsersAreIdenticalTms(users)));
  }

  LogicalResult matchCommutePattern(Operation *op,
                                    TMOpType tmUser) const override {
    // We can always commute a TM above an elementwise op
    return success();
  }
};
} // namespace

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
