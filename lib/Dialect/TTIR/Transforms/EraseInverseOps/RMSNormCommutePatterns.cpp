// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughRMSNorm
    : public TTIRCommuteOpRewritePattern<ReshapeOp, RMSNormOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, RMSNormOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // %0 = rms_norm(%arg0) {normalized_shape = [2048]}
  // %1 = reshape(%0) <{shape = [32, 2048]}>
  //
  // This method will transform this into:
  // %0 = reshape(%arg0) <{shape = [32, 2048]}>
  // %1 = rms_norm(%0) {normalized_shape = [2048]}
  //
  // The reshape is moved above rms_norm. If there was already an input reshape,
  // consecutive reshapes may be folded. The key constraint is that the last
  // dimension (normalization dimension) must remain unchanged by the reshape.
  void performCommuteUpwardsRewrite(RMSNormOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputReshapeType = cast<RankedTensorType>(reshapeUser.getType());

    // Create a new input shape that matches the output reshape shape
    SmallVector<int32_t> newInputShape(outputReshapeType.getShape());
    auto newInputType = RankedTensorType::get(outputReshapeType.getShape(),
                                              inputType.getElementType(),
                                              inputType.getEncoding());

    // Create reshape before rms_norm
    auto newInputReshape =
        rewriter.create<ReshapeOp>(op.getLoc(), newInputType, op.getInput(),
                                   rewriter.getI32ArrayAttr(newInputShape));

    auto newRmsNorm = rewriter.create<RMSNormOp>(
        op.getLoc(), outputReshapeType, newInputReshape, op.getWeight(),
        op.getBias(), op.getNormalizedShapeAttr(), op.getEpsilonAttr());

    // All users must be identical TMs. We must not reference `reshapeUser`
    // during/after replacements, as it will be erased on its turn.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(reshapeUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    for (auto *user : users) {
      rewriter.replaceOp(user, newRmsNorm);
    }
  }

private:
  bool isCommuteUpwardsViable(RMSNormOp op,
                              ReshapeOp reshapeUser) const override {
    // RMSNorm normalizes along the last dim, so it must be preserved by the
    // reshape.
    return utils::preservesDim(reshapeUser, -1);
  }

  bool isCommuteUpwardsFavorable(RMSNormOp op,
                                 ReshapeOp reshapeUser) const override {
    // We should commute if all users are identical reshapes.
    // This will move the reshape above rms_norm, potentially allowing
    // it to cancel with other reshapes via other commute patterns.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(RMSNormOp, ReshapeOp) const override {
    return false;
  }

  bool isCommuteDownwardsFavorable(RMSNormOp, ReshapeOp) const override {
    return false;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughDistributedRMSNorm
    : public TTIRCommuteOpRewritePattern<ReshapeOp, DistributedRMSNormOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, DistributedRMSNormOp,
      commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  //
  // %0 = distributed_rms_norm(%input, %weight, %residual) {cluster_axis, eps}
  // %1 = reshape(%0) <{shape = [...]}>
  //
  // This method will transform this into:
  //
  // %a = reshape(%input) <{shape = [...]}>
  // %b = reshape(%residual) <{shape = [...]}>   (only if residual is present)
  // %c = distributed_rms_norm(%a, %weight, %b) {cluster_axis, eps}
  //
  // The reshape is moved above distributed_rms_norm. Weight is shaped by the
  // normalized (last) dimension and is not reshaped. The residual must track
  // the input shape, so it is reshaped alongside. The key constraint is that
  // the last dimension (normalization dimension) must remain unchanged by the
  // reshape.
  void performCommuteUpwardsRewrite(DistributedRMSNormOp op,
                                    ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputReshapeType = cast<RankedTensorType>(reshapeUser.getType());
    ArrayRef<int64_t> newShape = outputReshapeType.getShape();
    SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());

    auto newInputType = RankedTensorType::get(
        newShape, inputType.getElementType(), inputType.getEncoding());
    auto newInputReshape = rewriter.create<ReshapeOp>(
        op.getLoc(), newInputType, op.getInput(),
        rewriter.getI32ArrayAttr(newShapeI32));

    Value newResidual = nullptr;
    if (op.getResidual()) {
      auto residualType = cast<RankedTensorType>(op.getResidual().getType());
      auto newResidualType = RankedTensorType::get(
          newShape, residualType.getElementType(), residualType.getEncoding());
      newResidual = rewriter.create<ReshapeOp>(
          op.getLoc(), newResidualType, op.getResidual(),
          rewriter.getI32ArrayAttr(newShapeI32));
    }

    auto newDistributedRmsNorm = rewriter.create<DistributedRMSNormOp>(
        op.getLoc(), outputReshapeType, newInputReshape.getResult(),
        op.getWeight(), newResidual, op.getClusterAxisAttr(),
        op.getEpsilonAttr());

    // All users must be identical TMs. We must not reference `reshapeUser`
    // during/after replacements, as it will be erased on its turn.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(reshapeUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    for (auto *user : users) {
      rewriter.replaceOp(user, newDistributedRmsNorm.getResult());
    }
  }

  // Consider the following IR pseudocode:
  //
  // %a = reshape(%orig_input) <{shape = [...]}>
  // %0 = distributed_rms_norm(%a, %weight, %residual) {cluster_axis, eps}
  //
  // This method will transform this into:
  //
  // %r = reshape(%residual) <{shape = inverse}>  (only if residual is present)
  // %0 = distributed_rms_norm(%orig_input, %weight, %r) {cluster_axis, eps}
  // %1 = reshape(%0) <{shape = [...]}>
  //
  // The reshape on the input operand is moved below distributed_rms_norm. The
  // last dimension (normalization dim) must be preserved by the reshape. The
  // residual, if present, is reshaped inversely so it continues to match the
  // input shape fed to distributed_rms_norm.
  void
  performCommuteDownwardsRewrite(DistributedRMSNormOp op,
                                 ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    auto origInput = reshapeOperand.getInput();
    auto origInputType = cast<RankedTensorType>(origInput.getType());
    ArrayRef<int64_t> origInputShape = origInputType.getShape();
    SmallVector<int32_t> origInputShapeI32(origInputShape.begin(),
                                           origInputShape.end());

    Value newResidual = nullptr;
    if (op.getResidual()) {
      auto residualType = cast<RankedTensorType>(op.getResidual().getType());
      auto newResidualType = RankedTensorType::get(
          origInputShape, residualType.getElementType(),
          residualType.getEncoding());
      newResidual = rewriter.create<ReshapeOp>(
          op.getLoc(), newResidualType, op.getResidual(),
          rewriter.getI32ArrayAttr(origInputShapeI32));
    }

    auto newOpType = RankedTensorType::get(origInputShape,
                                           origInputType.getElementType(),
                                           origInputType.getEncoding());
    auto newDistributedRmsNorm = rewriter.create<DistributedRMSNormOp>(
        op.getLoc(), newOpType, origInput, op.getWeight(), newResidual,
        op.getClusterAxisAttr(), op.getEpsilonAttr());

    auto originalType = cast<RankedTensorType>(op.getType());
    ArrayRef<int64_t> originalShape = originalType.getShape();
    SmallVector<int32_t> reshapeTargetShape(originalShape.begin(),
                                            originalShape.end());
    auto newReshape = rewriter.create<ReshapeOp>(
        reshapeOperand->getLoc(), op.getType(),
        newDistributedRmsNorm.getResult(),
        rewriter.getI32ArrayAttr(reshapeTargetShape));

    rewriter.replaceOp(op, newReshape.getResult());
  }

private:
  bool
  isCommuteUpwardsViable(DistributedRMSNormOp op,
                         ReshapeOp reshapeUser) const override {
    // Distributed RMSNorm normalizes along the last dim; it must be preserved
    // by the reshape for the commute to be semantically valid.
    return utils::preservesDim(reshapeUser, -1);
  }

  bool
  isCommuteUpwardsFavorable(DistributedRMSNormOp op,
                            ReshapeOp reshapeUser) const override {
    // Favorable if all users are identical reshapes — moving the reshape
    // above distributed_rms_norm collapses all of them onto the input and may
    // allow further cancellation via other commute patterns.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool
  isCommuteDownwardsViable(DistributedRMSNormOp op,
                           ReshapeOp reshapeOperand) const override {
    // Only commute when the reshape is on the input operand: moving a reshape
    // that feeds the weight (normalized shape) or residual downwards would
    // change the semantics.
    if (op.getInput() != reshapeOperand.getResult()) {
      return false;
    }
    // The last dim (normalization dim) must be preserved across the reshape.
    return utils::preservesDim(reshapeOperand, -1);
  }

  bool
  isCommuteDownwardsFavorable(DistributedRMSNormOp op,
                              ReshapeOp reshapeOperand) const override {
    // distributed_rms_norm typically has a single consumer, so commuting a
    // reshape below it does not duplicate the op — always favorable when
    // viable.
    return true;
  }
};

template <CommuteDirection commuteDirection>
void populateRMSNormCommutePatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns) {
  patterns.add<TTIRCommuteReshapeThroughRMSNorm<commuteDirection>>(ctx);
  patterns.add<TTIRCommuteReshapeThroughDistributedRMSNorm<commuteDirection>>(
      ctx);
}

template void populateRMSNormCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateRMSNormCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
