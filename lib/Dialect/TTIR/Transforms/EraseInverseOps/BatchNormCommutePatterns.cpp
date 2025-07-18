// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughBatchNorm
    : public TTIRCommuteOpRewritePattern<ttir::ReshapeOp, ttir::BatchNormOp,
                                         commuteDirection> {

public:
  using TTIRCommuteOpRewritePattern<
      ttir::ReshapeOp, ttir::BatchNormOp,
      commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(ttir::BatchNormOp op,
                                    ttir::ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    int32_t newBatchNormDim = findNewBatchNormDim(op, reshapeUser);

    if (newBatchNormDim == -1) {
      llvm_unreachable("isCommuteUpwardsViable should have returned false if "
                       "we cannot find a new batch dimension");
    }

    ttir::ReshapeOp newReshapeOperand =
        ttir::utils::createDPSOp<ttir::ReshapeOp>(
            rewriter, reshapeUser.getLoc(), reshapeUser.getResult().getType(),
            op.getOperand(), reshapeUser.getShape());

    ttir::BatchNormOp newBatchNorm =
        ttir::utils::createDPSOp<ttir::BatchNormOp>(
            rewriter, op.getLoc(), reshapeUser.getType(), newReshapeOperand,
            op.getScale(), op.getOffset(), op.getMean(), op.getVariance(),
            op.getEpsilon(), newBatchNormDim, op.getTraining());
    rewriter.replaceOp(reshapeUser, newBatchNorm);
  }

  void
  performCommuteDownwardsRewrite(ttir::BatchNormOp op,
                                 ttir::ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    int32_t newBatchNormDim = findNewBatchNormDim(op, reshapeOperand);

    if (newBatchNormDim == -1) {
      llvm_unreachable("isCommuteDownwardsViable should have returned false if "
                       "we cannot find a new batch dimension");
    }

    // scale, offset, mean, variance
    ttir::BatchNormOp newBatchNorm =
        ttir::utils::createDPSOp<ttir::BatchNormOp>(
            rewriter, op.getLoc(), reshapeOperand.getInput().getType(),
            reshapeOperand.getInput(), op.getScale(), op.getOffset(),
            op.getMean(), op.getVariance(), op.getEpsilon(), newBatchNormDim,
            op.getTraining());

    ttir::ReshapeOp newReshapeUser = ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, reshapeOperand.getLoc(), reshapeOperand.getResult().getType(),
        newBatchNorm, reshapeOperand.getShape());

    rewriter.replaceOp(op, newReshapeUser);
  }

private:
  bool isCommuteUpwardsViable(ttir::BatchNormOp op,
                              ttir::ReshapeOp reshapeUser) const override {
    // If at least one dim in reshape shape satisfies BOTH of the
    // following conditions, the reshape can be commuted above the batch
    // normalization:
    //  1. The dimension of the reshape shape has the same
    //     size as the dimension of normalization currently
    //  2. That dimension has the same volume to its left
    //     and right

    return findNewBatchNormDim(op, reshapeUser) != -1;
  }

  bool isCommuteUpwardsFavorable(ttir::BatchNormOp op,
                                 ttir::ReshapeOp) const override {
    // If all users of a batch normalization op are identical tms, then it is
    // favourable.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ttir::BatchNormOp op,
                                ttir::ReshapeOp reshapeOperand) const override {

    // If the operand is not the input activation than we cannot
    // commute.
    if (op.getOperand() != reshapeOperand.getResult()) {
      return false;
    }

    // If at least one dim in reshape shape satisfies BOTH of the
    // following conditions, the reshape can be commuted above the batch
    // normalization:
    //  1. The dimension of the reshape shape has the same
    //     size as the dimension of normalization currently
    //  2. That dimension has the same volume to its left
    //     and right

    return findNewBatchNormDim(op, reshapeOperand) != -1;
  }

  bool isCommuteDownwardsFavorable(ttir::BatchNormOp op,
                                   ttir::ReshapeOp) const override {

    // Commuting downwards is always favorable as the other operands will
    // not have any TMs placed on them
    return true;
  }

  uint64_t calculateLeftVolume(const ArrayRef<int64_t> &shape,
                               int64_t dim) const {
    return std::accumulate(shape.begin(), shape.begin() + dim, 1,
                           std::multiplies<int64_t>());
  }

  uint64_t calculateRightVolume(const ArrayRef<int64_t> &shape,
                                int64_t dim) const {
    return std::accumulate(shape.begin() + dim + 1, shape.end(), 1,
                           std::multiplies<int64_t>());
  }

  // This will return the new batch dim should the reshape be commuted through
  // the batch norm. If no such dim exists, it will return -1 - this implies
  // that the reshape cannot be commuted.
  int32_t findNewBatchNormDim(ttir::BatchNormOp op,
                              ttir::ReshapeOp reshapeOp) const {
    uint64_t rightVolumeToMatch =
        calculateRightVolume(op.getType().getShape(), op.getDimension());
    uint64_t leftVolumeToMatch =
        calculateLeftVolume(op.getType().getShape(), op.getDimension());

    for (int32_t dim = 0; dim < reshapeOp.getType().getRank(); ++dim) {
      if (reshapeOp.getType().getDimSize(dim) == op.getType().getDimSize(dim)) {
        if (calculateLeftVolume(reshapeOp.getType().getShape(), dim) ==
                leftVolumeToMatch &&
            calculateRightVolume(reshapeOp.getType().getShape(), dim) ==
                rightVolumeToMatch) {
          return dim;
        }
      }
    }
    return -1;
  }
};
} // namespace

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughBatchNorm
    : public TTIRCommuteOpRewritePattern<ttir::PermuteOp, ttir::BatchNormOp,
                                         commuteDirection> {

public:
  using TTIRCommuteOpRewritePattern<
      ttir::PermuteOp, ttir::BatchNormOp,
      commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(ttir::BatchNormOp op,
                                    ttir::PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    int32_t newBatchNormDim =
        std::find(permuteUser.getPermutation().begin(),
                  permuteUser.getPermutation().end(), op.getDimension()) -
        permuteUser.getPermutation().begin();

    ttir::PermuteOp newPermuteOperand =
        ttir::utils::createDPSOp<ttir::PermuteOp>(
            rewriter, permuteUser.getLoc(), permuteUser.getResult().getType(),
            op.getOperand(), permuteUser.getPermutation());

    ttir::BatchNormOp newBatchNorm =
        ttir::utils::createDPSOp<ttir::BatchNormOp>(
            rewriter, op.getLoc(), permuteUser.getType(), newPermuteOperand,
            op.getScale(), op.getOffset(), op.getMean(), op.getVariance(),
            op.getEpsilon(), newBatchNormDim, op.getTraining());
    rewriter.replaceOp(permuteUser, newBatchNorm);
  }

  void
  performCommuteDownwardsRewrite(ttir::BatchNormOp op,
                                 ttir::PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    int32_t newBatchNormDim =
        permuteOperand.getPermutation()[op.getDimension()];

    ttir::BatchNormOp newBatchNorm =
        ttir::utils::createDPSOp<ttir::BatchNormOp>(
            rewriter, op.getLoc(), permuteOperand.getInput().getType(),
            permuteOperand.getInput(), op.getScale(), op.getOffset(),
            op.getMean(), op.getVariance(), op.getEpsilon(), newBatchNormDim,
            op.getTraining());

    ttir::PermuteOp newPermuteUser = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, permuteOperand.getLoc(), permuteOperand.getResult().getType(),
        newBatchNorm, permuteOperand.getPermutation());
    rewriter.replaceOp(op, newPermuteUser);
  }

private:
  bool isCommuteUpwardsViable(ttir::BatchNormOp,
                              ttir::PermuteOp) const override {
    // We can always commute a permute above a batch normalization
    return true;
  }

  bool isCommuteUpwardsFavorable(ttir::BatchNormOp op,
                                 ttir::PermuteOp) const override {
    // If all users of a batch normalization op are identical tms, then it is
    // favorable.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ttir::BatchNormOp op,
                                ttir::PermuteOp permuteOperand) const override {

    // If the operand is not the input activation than we cannot
    // commute.
    if (op.getOperand() != permuteOperand.getInput()) {
      return false;
    }

    // Otherwise, it is always favourable to commute a permute below a batch
    // normalization
    return true;
  }

  bool isCommuteDownwardsFavorable(ttir::BatchNormOp,
                                   ttir::PermuteOp) const override {

    // Commuting downwards is always favorable as the other operands will
    // not have any TMs placed on them
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateBatchNormCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  patterns.add<TTIRCommutePermuteThroughBatchNorm<commuteDirection>>(ctx);
}

template void populateBatchNormCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateBatchNormCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
} // namespace mlir::tt::ttir
