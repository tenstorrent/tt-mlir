// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttir {

namespace {
template <typename TMOpType, typename ElementwiseInterfaceType,
          CommuteDirection direction>
class TTIRCommuteTmsThroughElementwiseRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<
          TMOpType, ElementwiseInterfaceType, direction> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      TMOpType, ElementwiseInterfaceType,
      direction>::TTIRCommuteOpInterfaceRewritePattern;

  void performCommuteAboveRewrite(ElementwiseInterfaceType op, TMOpType tmUser,
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

      auto newTM = ttir::utils::createDPSOp<TMOpType>(
          rewriter, op->getLoc(), newTMResultTypes[operandIdx],
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

  void performCommuteBelowRewrite(ElementwiseInterfaceType op,
                                  TMOpType tmOperand,
                                  PatternRewriter &rewriter) const override {
    // TODO: implement this
    RankedTensorType newEltwiseType =
        cast<RankedTensorType>(op->getResult(0).getType())
            .clone(tmOperand.getInput().getType().getShape());
    // For each of the other operands we must generate an inverse TM
    // Do not want to do anything to the DPS operand (last one)
    SmallVector<Value> newEltwiseOperands;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {
      Value operand = op->getOperand(operandIdx);

      if (operand.getDefiningOp() == tmOperand) {
        newEltwiseOperands.push_back(tmOperand.getInput());
        continue;
      }
      newEltwiseOperands.push_back(
          getInverseTM(tmOperand, operand, rewriter)->getResult(0));
    }

    newEltwiseOperands.push_back(rewriter.create<ttir::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType(), newEltwiseType.getEncoding()));

    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, newEltwiseType, op->getAttrs());

    TMOpType newUserTM = ttir::utils::createDPSOp<TMOpType>(
        rewriter, op->getLoc(),
        cast<RankedTensorType>(tmOperand->getResult(0).getType()),
        newEltwise->getResult(0), tmOperand->getAttrs());

    if (failed(newUserTM.verify())) {
      llvm_unreachable("Failed to verify new TM");
    }

    rewriter.replaceOp(op, newUserTM);
  }
};
} // namespace

namespace {
template <typename TMOpType, CommuteDirection direction>
class TTIRCommuteTmsThroughElementwiseUnaryRewriter
    : public TTIRCommuteTmsThroughElementwiseRewriter<
          TMOpType, ElementwiseUnary, direction> {
public:
  using TTIRCommuteTmsThroughElementwiseRewriter<
      TMOpType, ElementwiseUnary,
      direction>::TTIRCommuteTmsThroughElementwiseRewriter;

private:
  bool isCommuteAboveViable(ElementwiseUnary op,
                            TMOpType tmUser) const override {
    // We can always commute a TM above an elementwise op.
    return true;
  }

  bool isCommuteBelowViable(ElementwiseUnary op,
                            TMOpType tmOperand) const override {
    // We can always commute a TM below an elementwise op.
    // TODO: commute logic not implemented yet, thus it is not viable
    return true;
  }

  bool isCommuteAboveFavorable(ElementwiseUnary op, TMOpType) const override {
    // If all users of an elementwise unary op are identical tms, then it is
    // always favorable to commute them above it.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteBelowFavorable(ElementwiseUnary op, TMOpType) const override {
    // It is always favorable to commute a TM below an elementwise unary op as
    // there are no other operands.
    return true;
  }
};
} // namespace

namespace {
template <typename TMOpType, CommuteDirection direction>
class TTIRCommuteTmsThroughElementwiseBinaryRewriter
    : public TTIRCommuteTmsThroughElementwiseRewriter<
          TMOpType, ElementwiseBinary, direction> {
public:
  using TTIRCommuteTmsThroughElementwiseRewriter<
      TMOpType, ElementwiseBinary,
      direction>::TTIRCommuteTmsThroughElementwiseRewriter;

private:
  bool isCommuteAboveViable(ElementwiseBinary op,
                            TMOpType tmUser) const override {
    // We can always commute a TM above an elementwise op
    return true;
  }

  bool isCommuteBelowViable(ElementwiseBinary op,
                            TMOpType tmOperand) const override {
    // We can always commute a TM below an elementwise op
    // TODO: commute logic not implemented yet, thus it is not viable
    return true;
  }

  bool isCommuteAboveFavorable(ElementwiseBinary op, TMOpType) const override {
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

  bool isCommuteBelowFavorable(ElementwiseBinary op,
                               TMOpType tmOperand) const override {
    // If the other operand is an identical TM then it is always favourable to
    // commute below it as both can effectively be commuted at once.
    TMOpType firstOperand = op->getOperand(0).getDefiningOp<TMOpType>();
    TMOpType secondOperand = op->getOperand(1).getDefiningOp<TMOpType>();

    if (checkIdenticalTms(firstOperand, secondOperand)) {
      return true;
    }

    // If the other operand is not an identical TM except it lies on a
    // consteval-able path then commuting is essentially free as the inserted TM
    // will be absorbed into a consteval graph and the TM will be
    // consteval-able.

    // TODO: Implement the logic described above
    Value otherOperand = op->getOperand(0).getDefiningOp() == tmOperand
                             ? op->getOperand(1)
                             : op->getOperand(0);
    return this->valueTracesToConstantArgs(otherOperand);
  }
};
} // namespace

void populateElementwiseCommuteAbovePatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             mlir::func::FuncOp funcOp) {
  patterns.add<
      TTIRCommuteTmsThroughElementwiseUnaryRewriter<TransposeOp,
                                                    CommuteDirection::ABOVE>,
      TTIRCommuteTmsThroughElementwiseUnaryRewriter<PermuteOp,
                                                    CommuteDirection::ABOVE>,
      TTIRCommuteTmsThroughElementwiseUnaryRewriter<ReshapeOp,
                                                    CommuteDirection::ABOVE>,
      TTIRCommuteTmsThroughElementwiseBinaryRewriter<TransposeOp,
                                                     CommuteDirection::ABOVE>,
      TTIRCommuteTmsThroughElementwiseBinaryRewriter<PermuteOp,
                                                     CommuteDirection::ABOVE>,
      TTIRCommuteTmsThroughElementwiseBinaryRewriter<ReshapeOp,
                                                     CommuteDirection::ABOVE>>(
      ctx, funcOp);
}

void populateElementwiseCommuteBelowPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             mlir::func::FuncOp funcOp) {
  patterns.add<
      TTIRCommuteTmsThroughElementwiseUnaryRewriter<TransposeOp,
                                                    CommuteDirection::BELOW>,
      TTIRCommuteTmsThroughElementwiseUnaryRewriter<PermuteOp,
                                                    CommuteDirection::BELOW>,
      TTIRCommuteTmsThroughElementwiseUnaryRewriter<ReshapeOp,
                                                    CommuteDirection::BELOW>,
      TTIRCommuteTmsThroughElementwiseBinaryRewriter<TransposeOp,
                                                     CommuteDirection::BELOW>,
      TTIRCommuteTmsThroughElementwiseBinaryRewriter<PermuteOp,
                                                     CommuteDirection::BELOW>,
      TTIRCommuteTmsThroughElementwiseBinaryRewriter<ReshapeOp,
                                                     CommuteDirection::BELOW>>(
      ctx, funcOp);
}

} // namespace mlir::tt::ttir
