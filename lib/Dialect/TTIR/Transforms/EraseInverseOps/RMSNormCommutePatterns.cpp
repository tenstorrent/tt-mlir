// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

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
    // RMSNorm normalizes along the trailing dims specified by normalized_shape.
    // The reshape must keep those dims as the actual trailing dims of the
    // output. preservesDim alone is not enough: it checks whether the dim
    // value/stride still exists *somewhere*, but rms_norm requires it to be
    // the last dim. For example, reshape(1x3648x3840 -> 1x3648x3840x1)
    // "preserves" dim -1 (3840) in the sense that 3840 is still present, but
    // the actual trailing dim becomes 1, which breaks normalized_shape=[3840].
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputType = cast<RankedTensorType>(reshapeUser.getType());
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    // The trailing dims that rms_norm normalizes over must be identical.
    ArrayRef<int64_t> normalizedShape = op.getNormalizedShape();
    int64_t numNormDims = normalizedShape.size();
    if (numNormDims > static_cast<int64_t>(outputShape.size())) {
      return false;
    }
    auto outputTrailing = outputShape.take_back(numNormDims);
    auto inputTrailing = inputShape.take_back(numNormDims);
    return outputTrailing == inputTrailing;
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
void populateRMSNormCommutePatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns) {
  patterns.add<TTIRCommuteReshapeThroughRMSNorm<commuteDirection>>(ctx);
}

template void populateRMSNormCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateRMSNormCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
