// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughConcat
    : public TTIRCommuteOpRewritePattern<PermuteOp, ConcatOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, ConcatOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(ConcatOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {

    const int64_t *newConcatDimLoc =
        std::find(permuteUser.getPermutation().begin(),
                  permuteUser.getPermutation().end(), op.getDim());
    assert(newConcatDimLoc != permuteUser.getPermutation().end() &&
           "Concat dim specifies a dimension which is non existent in the "
           "permute, this should be impossible.");

    const int64_t newConcatDim =
        newConcatDimLoc - permuteUser.getPermutation().begin();

    SmallVector<Value> newConcatOperands;

    for (size_t i = 0; i < op->getNumOperands(); i++) {
      // Skip DPS operands
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      Value operand = op->getOperand(i);
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());

      RankedTensorType permuteOperandType = RankedTensorType::get(
          ttmlir::utils::applyPermutation(operandType.getShape(),
                                          permuteUser.getPermutation()),
          operandType.getElementType(), operandType.getEncoding());

      newConcatOperands.push_back(
          utils::createDPSOp<PermuteOp>(rewriter, op->getLoc(),
                                        permuteOperandType, op->getOperand(i),
                                        permuteUser.getPermutation())
              ->getResult(0));
    }

    RankedTensorType newConcatType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(op.getType().getShape(),
                                        permuteUser.getPermutation()),
        op.getType().getElementType(), op.getType().getEncoding());
    ConcatOp newConcat = utils::createDPSOp<ConcatOp>(
        rewriter, op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "shouldCommute should have ensured this is true");
    }

    for (auto *user : users) {
      rewriter.replaceOp(user, newConcat);
    }
  }

  void
  performCommuteDownwardsRewrite(ConcatOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    // Create inverse permutes for each of the other concat operands
    SmallVector<Value> newConcatOperands;
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }

      Value operand = op->getOperand(i);
      if (operand.getDefiningOp() == permuteOperand) {
        newConcatOperands.push_back(permuteOperand.getOperand(0));
        continue;
      }
      auto *newOperand = getInverseTM(permuteOperand, operand, rewriter);
      newConcatOperands.push_back(newOperand->getResult(0));
    }

    RankedTensorType newConcatType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(
            op.getType().getShape(),
            ttmlir::utils::inversePermutation(permuteOperand.getPermutation())),
        op.getType().getElementType(), op.getType().getEncoding());
    int64_t newConcatDim = permuteOperand.getPermutation()[op.getDim()];
    ConcatOp newConcat = utils::createDPSOp<ConcatOp>(
        rewriter, op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    RankedTensorType newPermuteType = RankedTensorType::get(
        ttmlir::utils::applyPermutation(newConcatType.getShape(),
                                        permuteOperand.getPermutation()),
        newConcatType.getElementType(), newConcatType.getEncoding());
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
          valueTracesToConstantArgs(op->getOperand(i))) {
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

  void performCommuteUpwardsRewrite(ConcatOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    SmallVector<Value> newConcatOperands;
    int64_t newConcatDim = this->retrieveReshapeUserConcatDim(op, reshapeUser);
    assert(newConcatDim != -1 && "isCommuteUpwardsViable should have confirmed "
                                 "that this value is not -1");

    ArrayRef<int64_t> newConcatShape = reshapeUser.getType().getShape();
    for (int64_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      Value operand = op->getOperand(i);
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
      SmallVector<int32_t> newOperandShape(newConcatShape);
      newOperandShape[newConcatDim] = operandType.getShape()[op.getDim()];
      RankedTensorType newOperandType = RankedTensorType::get(
          SmallVector<int64_t>(newOperandShape.begin(), newOperandShape.end()),
          operandType.getElementType(), operandType.getEncoding());
      newConcatOperands.push_back(utils::createDPSOp<ReshapeOp>(
          rewriter, op->getLoc(), newOperandType, operand,
          rewriter.getI32ArrayAttr(newOperandShape)));
    }

    RankedTensorType newConcatType =
        RankedTensorType::get(newConcatShape, op.getType().getElementType(),
                              op.getType().getEncoding());
    ConcatOp newConcat = utils::createDPSOp<ConcatOp>(
        rewriter, op->getLoc(), newConcatType, newConcatOperands, newConcatDim);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(reshapeUser, user) &&
             "shouldCommute should have ensured this is true");
    }

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
    int64_t concatDimSize = op.getType().getShape()[op.getDim()];
    int64_t volumeLeft =
        calculateShapeVolumeUpToDim(op.getType().getShape(), op.getDim());
    int64_t volumeRight =
        calculateShapeVolumeFromDim(op.getType().getShape(), op.getDim() + 1);

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
    // identical concat. This includes the case where there is one user.
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
          valueTracesToConstantArgs(op->getOperand(i))) {
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
