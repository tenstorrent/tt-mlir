// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughConcat
    : public TTIRCommuteOpRewritePattern<PermuteOp, ConcatOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, ConcatOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // %0 = concat(%arg0, %arg1) <{dim = 0 : si32}>
  // %1 = permute(%0) <{permutation = array<i64: 1, 0>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 1, 0>}>
  // %1 = permute(%arg1) <{permutation = array<i64: 1, 0>}>
  // %2 = concat(%0, %1) <{dim = 1 : si32}>
  void performCommuteUpwardsRewrite(ConcatOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    // We want to index shapes with the concat dim so we must ensure we are
    // using the positive index
    int32_t currentConcatDim =
        op.getDim() < 0 ? op.getResult().getType().getRank() + op.getDim()
                        : op.getDim();

    const int64_t *newConcatDimLoc =
        std::find(permuteUser.getPermutation().begin(),
                  permuteUser.getPermutation().end(), currentConcatDim);
    assert(newConcatDimLoc != permuteUser.getPermutation().end() &&
           "Concat dim specifies a dimension which is non existent in the "
           "permute, this should be impossible.");

    const int64_t newConcatDim =
        newConcatDimLoc - permuteUser.getPermutation().begin();

    SmallVector<Value> newConcatOperands;

    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");
    for (auto operand : op->getOperands()) {
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());

      RankedTensorType permuteOperandType = RankedTensorType::get(
          ttmlir::utils::applyPermutation(operandType.getShape(),
                                          permuteUser.getPermutation()),
          operandType.getElementType());

      newConcatOperands.push_back(
          rewriter
              .create<PermuteOp>(op->getLoc(), permuteOperandType, operand,
                                 permuteUser.getPermutation())
              ->getResult(0));
    }

    RankedTensorType newConcatType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        permuteUser.getPermutation()),
        op.getType().getElementType());
    ConcatOp newConcat = rewriter.create<ConcatOp>(
        op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    // All users must be identical TMs.
    // We must not reference `permuteUser` during/after replacements, as it will
    // be erased on its turn.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(permuteUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    for (auto *user : users) {
      rewriter.replaceOp(user, newConcat);
    }
  }

  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 1, 0>}>
  // %1 = permute(%arg1) <{permutation = array<i64: 1, 0>}>
  // %2 = concat(%0, %1) <{dim = 1 : si32}>
  //
  // This method will transform this into
  // %0 = concat(%arg0, %arg1) <{dim = 0 : si32}>
  // %1 = permute(%0) <{permutation = array<i64: 1, 0>}>
  void
  performCommuteDownwardsRewrite(ConcatOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    // Create inverse permutes for each of the other concat operands
    SmallVector<Value> newConcatOperands;
    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");
    for (auto operand : op->getOperands()) {
      if (operand.getDefiningOp() == permuteOperand) {
        newConcatOperands.push_back(permuteOperand.getInput());
        continue;
      }

      // We are adding an inverse permute operand to all the other concat
      // operands as they may not be and identical op to permuteOperand. If it
      // is, this inserted operand will be folded against the existing permute
      // to produce a single permute. That permute will be an identity
      // permutation and will be folded again to nothing.
      auto newOperand = getInverseTM(permuteOperand, operand, rewriter);
      newConcatOperands.push_back(newOperand.getResult());
    }

    // We want to index shapes with the concat dim so we must ensure we are
    // using the positive index
    int32_t currentConcatDim =
        op.getDim() < 0 ? op.getResult().getType().getRank() + op.getDim()
                        : op.getDim();

    RankedTensorType newConcatType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(
            op.getType().getShape(),
            ttmlir::utils::inversePermutation(permuteOperand.getPermutation())),
        op.getType().getElementType());
    int64_t newConcatDim = permuteOperand.getPermutation()[currentConcatDim];
    ConcatOp newConcat = rewriter.create<ConcatOp>(
        op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    RankedTensorType newPermuteType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(newConcatType.getShape(),
                                        permuteOperand.getPermutation()),
        newConcatType.getElementType());
    PermuteOp newPerm =
        rewriter.create<PermuteOp>(op->getLoc(), newPermuteType, newConcat,
                                   permuteOperand.getPermutation());

    rewriter.replaceOp(op, newPerm);
  }

private:
  bool isCommuteUpwardsViable(ConcatOp op, PermuteOp) const override {
    // We can always commute a permute above a concat.
    return true;
  }

  bool isCommuteUpwardsFavorable(ConcatOp op,
                                 PermuteOp permuteUser) const override {
    SmallVector<Operation *> users(op->getUsers());
    if (users.empty() || !checkAllUsersAreIdenticalTms(users)) {
      return false;
    }
    // Net TM count change from commuting one group of identical post-concat
    // permutes upward through an N-input concat:
    //   +N  new per-input permutes (one per concat input)
    //   -U  post-concat permutes removed  (U = users.size(), all identical)
    //   -2K cancellations  (K inputs already carrying the identical permute,
    //                        or on consteval paths)
    // Net = N - U - 2K.
    // When Net <= 0 the commute is always beneficial or neutral — allow it.
    // When Net > 0 the commute adds ops with no guaranteed cancellation.
    // Only block it when we can confirm inputs will never gain matching
    // permutes: specifically when any user feeds directly into a flattened
    // channel-last conv2d prep chain (permute → reshape → conv2d with
    // flattenedCompatInfo set).  Those inputs arrive in NCHW layout and will
    // never already carry the post-flatten NHWC permute.
    int64_t numOperands = static_cast<int64_t>(op->getNumOperands());
    int64_t numUsers = static_cast<int64_t>(users.size());
    int64_t cancelCount = 0;
    for (Value operand : op->getOperands()) {
      if (checkIdenticalTms(operand.getDefiningOp(), permuteUser) ||
          ttcore::valueTracesToConstantArgs(operand)) {
        ++cancelCount;
      }
    }
    if (numOperands - numUsers - 2 * cancelCount > 0) {
      for (Operation *user : users) {
        auto permOp = dyn_cast<PermuteOp>(user);
        if (!permOp || !permOp->hasOneUse()) {
          continue;
        }
        auto reshapeOp = dyn_cast<ReshapeOp>(*permOp->getUsers().begin());
        if (!reshapeOp || !reshapeOp->hasOneUse()) {
          continue;
        }
        auto conv2dOp = dyn_cast<Conv2dOp>(*reshapeOp->getUsers().begin());
        if (conv2dOp && conv2dOp.getFlattenedCompatInfo() != nullptr) {
          return false;
        }
      }
    }
    return true;
  }

  bool isCommuteDownwardsViable(ConcatOp op, PermuteOp) const override {
    // We can always commute a permute below a concat.
    return true;
  }

  bool isCommuteDownwardsFavorable(ConcatOp op,
                                   PermuteOp permuteOperand) const override {
    // Commuting downwards is favorable if the all other operands satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");
    for (auto operand : op->getOperands()) {
      if (checkIdenticalTms(operand.getDefiningOp(), permuteOperand) ||
          ttcore::valueTracesToConstantArgs(operand)) {
        continue;
      }
      return false;
    }
    return true;
  }
};
} // namespace

namespace {

template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughConcat
    : public TTIRCommuteOpRewritePattern<ReshapeOp, ConcatOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, ConcatOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // arg0 shape: [1, 64, 64, 1]
  // arg1 shape: [1, 64, 64, 1]
  // %0 = concat(%arg0, %arg1) <{dim = 3 : si32}>
  // %1 = reshape"(%0) <{shape = [1: i32, 4096: i32, 2: i32]}>
  //
  // This method will transform this into:
  // arg0 shape: [1, 64, 64, 1]
  // arg1 shape: [1, 64, 64, 1]
  // %0 = reshape(%arg0) <{shape = [1 : i32, 4096 : i32, 1 : i32]}>
  // %1 = reshape(%arg1) <{shape = [1 : i32, 4096 : i32, 1 : i32]}>
  // %2 = concat(%0, %1) <{dim = 2 : si32}>
  void performCommuteUpwardsRewrite(ConcatOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    SmallVector<Value> newConcatOperands;
    int64_t newConcatDim = this->retrieveReshapeUserConcatDim(op, reshapeUser);
    assert(newConcatDim != -1 && "isCommuteUpwardsViable should have confirmed "
                                 "that this value is not -1");

    // We want to index shapes with the concat dim so we must ensure we are
    // using the positive index
    int32_t currentConcatDim =
        op.getDim() < 0 ? op.getResult().getType().getRank() + op.getDim()
                        : op.getDim();

    ArrayRef<int64_t> newConcatShape = reshapeUser.getType().getShape();
    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");
    for (auto operand : op->getOperands()) {
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
      SmallVector<int32_t> newOperandShape(newConcatShape);
      newOperandShape[newConcatDim] = operandType.getShape()[currentConcatDim];
      RankedTensorType newOperandType = RankedTensorType::get(
          SmallVector<int64_t>(newOperandShape.begin(), newOperandShape.end()),
          operandType.getElementType());
      newConcatOperands.push_back(rewriter.create<ReshapeOp>(
          op->getLoc(), newOperandType, operand,
          rewriter.getI32ArrayAttr(newOperandShape)));
    }

    RankedTensorType newConcatType =
        RankedTensorType::get(newConcatShape, op.getType().getElementType());
    ConcatOp newConcat = rewriter.create<ConcatOp>(
        op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    // All users must be identical TMs.
    // We must not reference `reshapeUser` during/after replacements, as it will
    // be erased on its turn.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(reshapeUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    for (auto *user : users) {
      rewriter.replaceOp(user, newConcat);
    }
  }

  void
  performCommuteDownwardsRewrite(ConcatOp op, ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    // TODO(@LPanosTT): implement this
    llvm_unreachable("Not implemented, this should not be called.");
  }

private:
  int64_t retrieveReshapeUserConcatDim(ConcatOp op,
                                       ReshapeOp reshapeUser) const {
    // We want to index shapes with the concat dim so we must ensure we are
    // using the positive index
    int32_t currentConcatDim =
        op.getDim() < 0 ? op.getResult().getType().getRank() + op.getDim()
                        : op.getDim();
    int64_t concatDimSize = op.getType().getShape()[currentConcatDim];
    int64_t volumeLeft =
        calculateShapeVolumeUpToDim(op.getType().getShape(), currentConcatDim);
    int64_t volumeRight = calculateShapeVolumeFromDim(op.getType().getShape(),
                                                      currentConcatDim + 1);

    ArrayRef<int64_t> reshapeShape = reshapeUser.getType().getShape();
    for (size_t dim = 0; dim < reshapeShape.size(); dim++) {
      if (reshapeShape[dim] == concatDimSize &&
          calculateShapeVolumeUpToDim(reshapeShape, dim) == volumeLeft &&
          calculateShapeVolumeFromDim(reshapeShape, dim + 1) == volumeRight) {
        return dim;
      }
    }
    return -1;
  }

  int64_t calculateShapeVolumeUpToDim(ArrayRef<int64_t> shape,
                                      int64_t dim) const {
    int64_t volume = 1;
    for (int64_t i = 0; i < dim; i++) {
      volume *= shape[i];
    }
    return volume;
  }

  int64_t calculateShapeVolumeFromDim(ArrayRef<int64_t> shape,
                                      int64_t dim) const {
    int64_t volume = 1;
    for (size_t i = dim; i < shape.size(); i++) {
      volume *= shape[i];
    }
    return volume;
  }

  bool isCommuteUpwardsViable(ConcatOp op,
                              ReshapeOp reshapeUser) const override {
    // We can commute a reshape above a concat op if there exists a
    // dimension in the reshaped shape that is identical to the size
    // of the concat dimension AND the volume on either side of that
    // dimension is identical before and after the reshape.
    if (retrieveReshapeUserConcatDim(op, reshapeUser) != -1) {
      return true;
    }
    return false;
  }

  bool isCommuteUpwardsFavorable(ConcatOp op, ReshapeOp) const override {
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ConcatOp op, ReshapeOp) const override {
    // We can commute a reshape below a concat op if the shape BEFORE the
    // reshape has a dimension with the same size as the concat dimension,
    // AND the volume on either side of that dimension is identical before
    // and after the reshape.

    // TODO(@LPanosTT): commute logic not implemented, thus the commute is not
    // viable
    return false;
  }

  bool isCommuteDownwardsFavorable(ConcatOp op,
                                   ReshapeOp reshapeOperand) const override {
    // Commuting downwards is favorable if the all other operands a satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    assert(!isa<DestinationStyleOpInterface>(op.getOperation()) &&
           "DPS ops are not supported");
    for (auto operand : op->getOperands()) {
      if (checkIdenticalTms(operand.getDefiningOp(), reshapeOperand) ||
          ttcore::valueTracesToConstantArgs(operand)) {
        continue;
      }
      return false;
    }
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateConcatCommutePatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns) {
  patterns.insert<TTIRCommutePermuteThroughConcat<commuteDirection>>(ctx);
  patterns.insert<TTIRCommuteReshapeThroughConcat<commuteDirection>>(ctx);
}

template void populateConcatCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateConcatCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
