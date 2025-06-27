// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <typename TMOpType, typename ElementwiseInterfaceType>
class TTIRCommuteTmsAboveElementwiseRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<TMOpType,
                                                  ElementwiseInterfaceType> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      TMOpType, ElementwiseInterfaceType>::TTIRCommuteOpInterfaceRewritePattern;

  void performCommuteRewrite(ElementwiseInterfaceType op, TMOpType tmUser,
                             PatternRewriter &rewriter) const override {

    llvm::SmallVector<Operation *> users(op->getUsers());
    auto oldEltwiseType = cast<RankedTensorType>(op->getResult(0).getType());
    auto oldTMResultType = tmUser.getResult().getType();

    auto newEltwiseType = RankedTensorType::get(oldTMResultType.getShape(),
                                                oldEltwiseType.getElementType(),
                                                oldTMResultType.getEncoding());

    SmallVector<Value> newEltwiseOperands;
    SmallVector<RankedTensorType> newTMResultTypes;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {

      // The new TM will have the same shape as before, but if the eltwise op
      // was a typecast, it will have the element type of the original operand
      // of the eltwise. So we need to generate a new type for the TM keeping
      // this in mind.
      auto operandType =
          cast<RankedTensorType>(op->getOperand(operandIdx).getType());

      newTMResultTypes.push_back(
          oldTMResultType.clone(operandType.getElementType()));

      mlir::Location newLoc = ttmlir::utils::appendLocationSuffix(
          tmUser->getLoc(), "_tm" + std::to_string(operandIdx));
      auto newTM = ttir::utils::createDPSOp<TMOpType>(
          rewriter, newLoc, newTMResultTypes[operandIdx],
          op->getOperand(operandIdx), tmUser->getAttrs());

      newEltwiseOperands.push_back(newTM);
    }

    newEltwiseOperands.push_back(rewriter.create<ttir::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType(), newEltwiseType.getEncoding()));

    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, newEltwiseType, op->getAttrs());

    // This only works when all the users are an identical TM
    // In the future this function may be called when this is not
    // the case, and we'll need to insert user clones on the
    // user edges that do not have an inverse on them.
    for (auto *user : users) {
      assert(checkIdenticalTms(tmUser, user) &&
             "shouldCommute should have ensured this is true");
    }

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
  bool isCommuteViable(ElementwiseUnary op, TMOpType tmUser) const override {
    // We can always commute a TM above an elementwise op.
    return true;
  }

  bool isCommuteFavorable(ElementwiseUnary op, TMOpType) const override {
    // If all users of an elementwise unary op are identical tms, then it is
    // always favorable to commute them above it.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
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
  bool isCommuteViable(ElementwiseBinary op, TMOpType tmUser) const override {
    // We can always commute a TM above an elementwise op
    return true;
  }

  bool isCommuteFavorable(ElementwiseBinary op, TMOpType) const override {
    // In some cases there may be an implicit broadcast on one of the operands.
    // That is there is no broadcast op on one of the operands but a broadcast
    // is required to execute the op nonetheless. We do not handle this yet. So
    // we will make sure the operand types are identical.
    auto firstOperandType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto secondOperandType =
        cast<RankedTensorType>(op->getOperand(1).getType());

    // If all users of an elementwise binary op are identical tms, then it is
    // typically favorable to commute them above it. In some cases we may not
    // be able to erase/consteval one or both of the commuted operand TMs.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users) &&
           firstOperandType == secondOperandType;
  }
};
} // namespace

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns) {
  patterns.add<TTIRCommuteTmsAboveElementwiseUnaryRewriter<PermuteOp>,
               TTIRCommuteTmsAboveElementwiseUnaryRewriter<ReshapeOp>,
               TTIRCommuteTmsAboveElementwiseBinaryRewriter<PermuteOp>,
               TTIRCommuteTmsAboveElementwiseBinaryRewriter<ReshapeOp>>(ctx);
}

} // namespace mlir::tt::ttir
