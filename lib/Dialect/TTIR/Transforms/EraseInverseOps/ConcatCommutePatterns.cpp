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

    for (OpOperand &opOperand : op->getOpOperands()) {
      // Skip DPS operands
      if (op.isDpsInit(&opOperand)) {
        continue;
      }
      Value operand = opOperand.get();
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());

      RankedTensorType permuteOperandType = RankedTensorType::get(
          ttmlir::utils::applyPermutation(operandType.getShape(),
                                          permuteUser.getPermutation()),
          operandType.getElementType());

      newConcatOperands.push_back(
          utils::createDPSOp<PermuteOp>(rewriter, op->getLoc(),
                                        permuteOperandType, operand,
                                        permuteUser.getPermutation())
              ->getResult(0));
    }

    RankedTensorType newConcatType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        permuteUser.getPermutation()),
        op.getType().getElementType());
    ConcatOp newConcat = utils::createDPSOp<ConcatOp>(
        rewriter, op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
      rewriter.replaceOp(user, newConcat);
    }
  }

  // Consider the follwing IR pseudocode:
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
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (op.isDpsInit(&opOperand)) {
        continue;
      }

      Value operand = opOperand.get();
      if (operand.getDefiningOp() == permuteOperand) {
        newConcatOperands.push_back(permuteOperand.getOperand(0));
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
    ConcatOp newConcat = utils::createDPSOp<ConcatOp>(
        rewriter, op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    RankedTensorType newPermuteType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(newConcatType.getShape(),
                                        permuteOperand.getPermutation()),
        newConcatType.getElementType());
    PermuteOp newPerm = utils::createDPSOp<PermuteOp>(
        rewriter, op->getLoc(), newPermuteType, newConcat,
        permuteOperand.getPermutation());

    rewriter.replaceOp(op, newPerm);
  }

private:
  bool isCommuteUpwardsViable(ConcatOp op, PermuteOp) const override {
    // We can always commute a permute above a concat.
    return true;
  }

  bool isCommuteUpwardsFavorable(ConcatOp op, PermuteOp) const override {
    // We should always commute a permute above a concat if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
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

namespace {

template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughConcat
    : public TTIRCommuteOpRewritePattern<ReshapeOp, ConcatOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, ConcatOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the follwing IR pseudocode:
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
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (op.isDpsInit(&opOperand)) {
        continue;
      }
      Value operand = opOperand.get();
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
      SmallVector<int32_t> newOperandShape(newConcatShape);
      newOperandShape[newConcatDim] = operandType.getShape()[currentConcatDim];
      RankedTensorType newOperandType = RankedTensorType::get(
          SmallVector<int64_t>(newOperandShape.begin(), newOperandShape.end()),
          operandType.getElementType());
      newConcatOperands.push_back(utils::createDPSOp<ReshapeOp>(
          rewriter, op->getLoc(), newOperandType, operand,
          rewriter.getI32ArrayAttr(newOperandShape)));
    }

    RankedTensorType newConcatType =
        RankedTensorType::get(newConcatShape, op.getType().getElementType());
    ConcatOp newConcat = utils::createDPSOp<ConcatOp>(
        rewriter, op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(reshapeUser, user) &&
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
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
    // We should always commute a reshape above a concat if all users are an
    // identical reshape. This includes the case where there is one user.
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

    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      if (checkIdenticalTms(op->getOperand(i).getDefiningOp(),
                            reshapeOperand) ||
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
