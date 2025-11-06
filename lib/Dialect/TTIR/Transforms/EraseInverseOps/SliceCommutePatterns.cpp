// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughSlice
    : public TTIRCommuteOpRewritePattern<PermuteOp, SliceStaticOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, SliceStaticOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // %0 = slice(%arg0) <{
  //      begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = permute(%1) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  void performCommuteUpwardsRewrite(SliceStaticOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {

    RankedTensorType sliceOperandType = op.getInput().getType();

    RankedTensorType newPermuteType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(sliceOperandType.getShape(),
                                        permuteUser.getPermutation()),
        sliceOperandType.getElementType(), sliceOperandType.getEncoding());

    PermuteOp newPerm = utils::createDPSOp<PermuteOp>(
        rewriter, op->getLoc(), newPermuteType, op.getInput(),
        permuteUser.getPermutation());

    SmallVector<Attribute> newSliceStarts = ttmlir::utils::applyPermutation(
        op.getBegins().getValue(), permuteUser.getPermutation());
    SmallVector<Attribute> newSliceEnds = ttmlir::utils::applyPermutation(
        op.getEnds().getValue(), permuteUser.getPermutation());
    SmallVector<Attribute> newSliceSteps = ttmlir::utils::applyPermutation(
        op.getStep().getValue(), permuteUser.getPermutation());

    RankedTensorType newSliceType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        permuteUser.getPermutation()),
        op.getType().getElementType(), op.getType().getEncoding());

    SliceStaticOp newSlice = utils::createDPSOp<SliceStaticOp>(
        rewriter, op->getLoc(), newSliceType, newPerm,
        rewriter.getArrayAttr(newSliceStarts),
        rewriter.getArrayAttr(newSliceEnds),
        rewriter.getArrayAttr(newSliceSteps));

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
      rewriter.replaceOp(user, newSlice);
    }
  }

  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = slice(%0) <{
  //      begins = [0 : i32, 0 : i32, 0 : i32, 80 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}
  //
  // This method will transform this into:
  // %0 = slice(%arg0) <{
  //      begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32],
  //      ends = [1 : i32, 160 : i32, 160 : i32, 160 : i32],
  //      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // %1 = permute(%1) <{permutation = array<i64: 0, 2, 3, 1>}>
  void
  performCommuteDownwardsRewrite(SliceStaticOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    SmallVector<int64_t> inversePermutation =
        ttmlir::utils::inversePermutation(permuteOperand.getPermutation());
    RankedTensorType newSliceType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        inversePermutation),
        op.getType().getElementType(), op.getType().getEncoding());

    SmallVector<Attribute> newSliceStarts = ttmlir::utils::applyPermutation(
        op.getBegins().getValue(), inversePermutation);
    SmallVector<Attribute> newSliceEnds = ttmlir::utils::applyPermutation(
        op.getEnds().getValue(), inversePermutation);
    SmallVector<Attribute> newSliceSteps = ttmlir::utils::applyPermutation(
        op.getStep().getValue(), inversePermutation);

    SliceStaticOp newSlice = utils::createDPSOp<SliceStaticOp>(
        rewriter, op->getLoc(), newSliceType, permuteOperand.getInput(),
        rewriter.getArrayAttr(newSliceStarts),
        rewriter.getArrayAttr(newSliceEnds),
        rewriter.getArrayAttr(newSliceSteps));

    RankedTensorType newPermuteType = RankedTensorType::get(
        op.getType().getShape(), newSlice.getType().getElementType(),
        newSlice.getType().getEncoding());
    PermuteOp newPerm = utils::createDPSOp<PermuteOp>(
        rewriter, op->getLoc(), newPermuteType, newSlice,
        permuteOperand.getPermutation());

    rewriter.replaceOp(op, newPerm);
  }

private:
  bool isCommuteUpwardsViable(SliceStaticOp op, PermuteOp) const override {
    // Commuting a permute through a slice is always viable
    return true;
  }

  bool isCommuteUpwardsFavorable(SliceStaticOp op, PermuteOp) const override {
    // We should always commute a permute above a slice if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(SliceStaticOp op, PermuteOp) const override {
    // Commuting a permute through a slice is always viable
    return true;
  }

  bool isCommuteDownwardsFavorable(SliceStaticOp op,
                                   PermuteOp permuteOperand) const override {
    // Commuting downwards is favorable if the all other operands a satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      if (checkIdenticalTms(op->getOperand(i).getDefiningOp(),
                            permuteOperand) ||
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
void populateSliceCommutePatterns(MLIRContext *ctx,
                                  RewritePatternSet &patterns) {
  patterns.insert<TTIRCommutePermuteThroughSlice<commuteDirection>>(ctx);
}

template void populateSliceCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateSliceCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
