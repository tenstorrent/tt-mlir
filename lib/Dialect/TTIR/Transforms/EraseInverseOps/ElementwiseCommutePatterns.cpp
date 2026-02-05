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
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands();
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
      auto newTM = rewriter.create<TMOpType>(
          newLoc, newTMResultTypes[operandIdx], op->getOperand(operandIdx),
          tmUser->getAttrs());

      newEltwiseOperands.push_back(newTM);
    }

    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, newEltwiseType, op->getAttrs());

    // This only works when all the users are an identical TM
    // In the future this function may be called when this is not
    // the case, and we'll need to insert user clones on the
    // user edges that do not have an inverse on them.
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(tmUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

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

    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");
    SmallVector<Value> newEltwiseOperands;
    for (auto operand : op->getOperands()) {
      if (operand.getDefiningOp() == tmOperand) {
        newEltwiseOperands.push_back(tmOperand.getInput());
        continue;
      }
      newEltwiseOperands.push_back(getInverseTM(tmOperand, operand, rewriter));
    }

    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, newEltwiseType, op->getAttrs());

    RankedTensorType newTMType =
        cast<RankedTensorType>(op->getResult(0).getType())
            .clone(tmOperand.getType().getShape());
    TMOpType newUserTM = rewriter.create<TMOpType>(op->getLoc(), newTMType,
                                                   newEltwise->getResult(0),
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

  // Walk up the def chain through single-operand ops to check for broadcast.
  bool hasBroadcastInChain(Value operand) const {
    Operation *defOp = operand.getDefiningOp();
    while (defOp) {
      if (isa<ttir::BroadcastOp>(defOp)) {
        return true;
      }
      if (defOp->getNumOperands() == 1) {
        defOp = defOp->getOperand(0).getDefiningOp();
      } else {
        break;
      }
    }
    return false;
  }

  bool isCommuteDownwardsFavorable(ElementwiseInterfaceType op,
                                   TMOpType tmOperand) const override {
    // Cost model for commuting tmOperand below the elementwise op.
    //
    // Baseline delta is 0: removing tmOperand (-1) and adding a result
    // TM after the new elementwise (+1) cancel out.
    //
    // Per other-operand costs:
    //   Identical TM            : -1  (the existing TM gets erased)
    //   Consteval                :  0  (inverse TM is const-folded away)
    //   Consteval + broadcast
    //     when TMOpType=Reshape  : +1  (see below)
    //   Non-consteval            : +1  (inverse TM persists at runtime)
    //
    // Reshape through broadcast penalty:
    //   When a consteval operand goes through a broadcast and the commuted
    //   TM is a reshape, the inverse reshape is inserted after the
    //   broadcast, creating a broadcast→reshape chain that materializes a
    //   large constant in DRAM.
    //
    //   Example — before commute (original IR):
    //     %a0 = ...                                    : (1,16,1,32,1,128)
    //     %a1 = reshape %a0                            : (1,16,1,32,128)
    //     %b0 = ...                                    : (32,128)
    //     %b1 = reshape %b0                            : (1,1,1,32,128)
    //     %b2 = broadcast %b1                          : (1,16,1,32,128)
    //     %c  = add(%a1, %b2)                          : (1,16,1,32,128)
    //
    //   After commute (%a1's reshape is pushed below add):
    //     %a0 = ...                                    : (1,16,1,32,1,128)
    //     // reshape on %a0 removed — commuted below
    //     %b0 = ...                                    : (32,128)
    //     %b1 = reshape %b0                            : (1,32,128)
    //     %b2 = broadcast %b1                          : (16,32,128)
    //     %b3 = reshape %b2 (inverse reshape)          : (1,16,1,32,1,128)
    //     %c0 = add(%a0, %b3)                          : (1,16,1,32,1,128)
    //     %c1 = reshape %c0 (commuted from %a1)        : (1,16,1,32,128)
    //
    //   Net effect: %a1's reshape didn't disappear — it moved to %c1.
    //   Meanwhile %b gained an extra inverse reshape (%b3), increasing
    //   total op count by 1.  Although the entire %b chain is consteval,
    //   the larger broadcast output (%b2) now materializes a bigger
    //   constant in DRAM, increasing DRAM pressure.
    //
    //   Permute does not have this problem: broadcast→permute preserves
    //   rank and dim structure, consistent with BroadcastCommutePatterns
    //   which allows permute through broadcast unconditionally.
    //
    // Favorable when netDelta <= 0 (no worse after commute; neutral
    // commutes may enable downstream inverse-pair cancellation).

    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");

    int netDelta = 0;

    for (auto operand : op->getOperands()) {
      if (operand.getDefiningOp() == tmOperand) {
        continue;
      }

      if (checkIdenticalTms(operand.getDefiningOp(), tmOperand)) {
        netDelta -= 1; // Identical TM will be erased
        continue;
      }

      if (ttcore::valueTracesToConstantArgs(operand)) {
        // Penalize only reshape through broadcast: the inverse reshape
        // after broadcast creates a broadcast→reshape chain that
        // increases DRAM pressure.  Permute through broadcast is fine.
        if constexpr (std::is_same_v<TMOpType, ReshapeOp>) {
          if (hasBroadcastInChain(operand)) {
            netDelta += 1;
          }
        }
        continue;
      }

      // Non-consteval, non-identical: inverse TM stays at runtime
      netDelta += 1;
    }

    return netDelta <= 0;
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
