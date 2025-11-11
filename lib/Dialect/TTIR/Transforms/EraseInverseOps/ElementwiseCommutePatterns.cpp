// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <typename TMOpType, typename ElementwiseInterfaceType,
          CommuteDirection commuteDirection>
class TTIRCommuteTmsThroughElementwiseRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<
          TMOpType, ElementwiseInterfaceType, commuteDirection> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      TMOpType, ElementwiseInterfaceType,
      commuteDirection>::TTIRCommuteOpInterfaceRewritePattern;

  void performCommuteUpwardsRewrite(ElementwiseInterfaceType op,
                                    TMOpType tmUser,
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
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
    }

    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }

  void
  performCommuteDownwardsRewrite(ElementwiseInterfaceType op,
                                 TMOpType tmOperand,
                                 PatternRewriter &rewriter) const override {
    RankedTensorType newEltwiseType =
        cast<RankedTensorType>(op->getResult(0).getType())
            .clone(tmOperand.getInput().getType().getShape());
    // For each of the other operands we must generate an inverse TM
    // Do not want to do anything to the DPS operand
    SmallVector<Value> newEltwiseOperands;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands();
         operandIdx++) {

      if (auto asDpsOp =
              dyn_cast<DestinationStyleOpInterface>(op.getOperation())) {
        if (asDpsOp.isDpsInit(&asDpsOp->getOpOperand(operandIdx))) {
          continue;
        }
      }
      Value operand = op->getOperand(operandIdx);

      if (operand.getDefiningOp() == tmOperand) {
        newEltwiseOperands.push_back(tmOperand.getInput());
        continue;
      }
      newEltwiseOperands.push_back(getInverseTM(tmOperand, operand, rewriter));
    }

    newEltwiseOperands.push_back(rewriter.create<ttir::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType(), newEltwiseType.getEncoding()));

    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, newEltwiseType, op->getAttrs());

    RankedTensorType newTMType =
        cast<RankedTensorType>(op->getResult(0).getType())
            .clone(tmOperand.getType().getShape());
    TMOpType newUserTM = ttir::utils::createDPSOp<TMOpType>(
        rewriter, op->getLoc(), newTMType, newEltwise->getResult(0),
        tmOperand->getAttrs());

    rewriter.replaceOp(op, newUserTM);
  }

private:
  bool isCommuteUpwardsViable(ElementwiseInterfaceType op,
                              TMOpType) const override {
    // We can always commute a TM above an elementwise op
    return true;
  }

  bool isCommuteUpwardsFavorable(ElementwiseInterfaceType op,
                                 TMOpType) const override {
    // If all users of an elementwise binary op are identical tms, then it is
    // typically favorable to commute them above it. In some cases we may not
    // be able to erase/consteval one or both of the commuted operand TMs.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ElementwiseInterfaceType op,
                                TMOpType) const override {
    // We can always commute a TM below an elementwise op
    return true;
  }

  bool isCommuteDownwardsFavorable(ElementwiseInterfaceType op,
                                   TMOpType tmOperand) const override {
    // Commuting downwards is favorable if the all other operands a satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      if (auto asDpsOp =
              dyn_cast<DestinationStyleOpInterface>(op.getOperation())) {
        if (asDpsOp.isDpsInit(&asDpsOp->getOpOperand(i))) {
          continue;
        }
      }
      if (checkIdenticalTms(op->getOperand(i).getDefiningOp(), tmOperand) ||
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
void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns) {
  patterns.add<
      TTIRCommuteTmsThroughElementwiseRewriter<PermuteOp, ElementwiseUnary,
                                               commuteDirection>,
      TTIRCommuteTmsThroughElementwiseRewriter<ReshapeOp, ElementwiseUnary,
                                               commuteDirection>,
      TTIRCommuteTmsThroughElementwiseRewriter<PermuteOp, ElementwiseBinary,
                                               commuteDirection>,
      TTIRCommuteTmsThroughElementwiseRewriter<ReshapeOp, ElementwiseBinary,
                                               commuteDirection>>(ctx);
}

template void populateElementwiseCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateElementwiseCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
