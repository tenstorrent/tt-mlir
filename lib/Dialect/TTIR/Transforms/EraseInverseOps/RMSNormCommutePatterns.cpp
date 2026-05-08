// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

namespace {
// Shared upwards-commute pattern for RMSNorm-like ops. Both `RMSNormOp` and
// `DistributedRMSNormOp` normalize along the last dimension, so a reshape can
// be commuted above them as long as the last dim is preserved. The
// `DistributedRMSNormOp` variant additionally reshapes the optional residual
// (which must match the input shape).
//
// Consider the following IR pseudocode:
//   %0 = rms_norm(%arg0) {normalized_shape = [2048]}
//   %1 = reshape(%0) <{shape = [32, 2048]}>
//
// This pattern transforms it into:
//   %0 = reshape(%arg0) <{shape = [32, 2048]}>
//   %1 = rms_norm(%0) {normalized_shape = [2048]}
//
// The reshape is moved above rms_norm. If there was already an input reshape,
// consecutive reshapes may be folded. The key constraint is that the last
// dimension (normalization dimension) must remain unchanged by the reshape.
template <typename RMSNormOpT, CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughRMSNormBase
    : public TTIRCommuteOpRewritePattern<ReshapeOp, RMSNormOpT,
                                         commuteDirection> {
  static_assert(std::is_same_v<RMSNormOpT, RMSNormOp> ||
                    std::is_same_v<RMSNormOpT, DistributedRMSNormOp>,
                "Pattern only supports RMSNormOp and DistributedRMSNormOp");

public:
  using TTIRCommuteOpRewritePattern<
      ReshapeOp, RMSNormOpT, commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(RMSNormOpT op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputReshapeType = cast<RankedTensorType>(reshapeUser.getType());

    SmallVector<int32_t> newInputShape(outputReshapeType.getShape());
    auto newInputType = RankedTensorType::get(outputReshapeType.getShape(),
                                              inputType.getElementType(),
                                              inputType.getEncoding());

    auto newInputReshape =
        rewriter.create<ReshapeOp>(op.getLoc(), newInputType, op.getInput(),
                                   rewriter.getI32ArrayAttr(newInputShape));

    RMSNormOpT newRmsNorm;
    if constexpr (std::is_same_v<RMSNormOpT, RMSNormOp>) {
      newRmsNorm = rewriter.create<RMSNormOp>(
          op.getLoc(), outputReshapeType, newInputReshape, op.getWeight(),
          op.getBias(), op.getNormalizedShapeAttr(), op.getEpsilonAttr());
    } else {
      Value newResidual;
      if (op.getResidual()) {
        auto residualType = cast<RankedTensorType>(op.getResidual().getType());
        auto newResidualType = RankedTensorType::get(
            outputReshapeType.getShape(), residualType.getElementType(),
            residualType.getEncoding());
        newResidual = rewriter.create<ReshapeOp>(
            op.getLoc(), newResidualType, op.getResidual(),
            rewriter.getI32ArrayAttr(newInputShape));
      }
      newRmsNorm = rewriter.create<DistributedRMSNormOp>(
          op.getLoc(), outputReshapeType, newInputReshape, op.getWeight(),
          newResidual, op.getClusterAxisAttr(), op.getEpsilonAttr());
    }

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
  bool isCommuteUpwardsViable(RMSNormOpT op,
                              ReshapeOp reshapeUser) const override {
    // RMSNorm normalizes along the last dim, so it must be preserved by the
    // reshape.
    return utils::preservesDim(reshapeUser, -1);
  }

  bool isCommuteUpwardsFavorable(RMSNormOpT op,
                                 ReshapeOp reshapeUser) const override {
    // We should commute if all users are identical reshapes. This will move
    // the reshape above rms_norm, potentially allowing it to cancel with other
    // reshapes via other commute patterns.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(RMSNormOpT, ReshapeOp) const override {
    return false;
  }

  bool isCommuteDownwardsFavorable(RMSNormOpT, ReshapeOp) const override {
    return false;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateRMSNormCommutePatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns) {
  patterns
      .add<TTIRCommuteReshapeThroughRMSNormBase<RMSNormOp, commuteDirection>,
           TTIRCommuteReshapeThroughRMSNormBase<DistributedRMSNormOp,
                                                commuteDirection>>(ctx);
}

template void populateRMSNormCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateRMSNormCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
