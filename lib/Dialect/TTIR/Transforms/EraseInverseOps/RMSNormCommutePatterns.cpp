// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "llvm/ADT/SmallVector.h"

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
  // %0 = reshape(%arg0) ...
  // %1 = rms_norm(%0) ...
  // %2 = reshape(%1) ...
  //
  // This method will transform this into:
  // %0 = rms_norm(%arg0) ...
  //
  // The reshape pair is eliminated because they are inverses of each other.
  // The key constraint is that the last dimension (normalization dimension)
  // must remain unchanged by the reshapes.

  void performCommuteUpwardsRewrite(RMSNormOp op, ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {
    auto inputReshape = op.getInput().getDefiningOp<ReshapeOp>();

    auto originalInputType =
        cast<RankedTensorType>(inputReshape.getInput().getType());
    auto outputReshapeType = cast<RankedTensorType>(reshapeUser.getType());

    SmallVector<int64_t> normalizedShape{originalInputType.getShape().back()};

    auto newRmsNorm = rewriter.create<RMSNormOp>(
        op.getLoc(), outputReshapeType, inputReshape.getInput(), op.getWeight(),
        op.getBias(), rewriter.getDenseI64ArrayAttr(normalizedShape),
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
      rewriter.replaceOp(user, newRmsNorm);
    }
  }

private:
  bool isCommuteUpwardsViable(RMSNormOp op,
                              ReshapeOp reshapeUser) const override {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto outputReshapeType = cast<RankedTensorType>(reshapeUser.getType());

    // RMSNorm normalizes along the last dim, so it must be unchanged by the
    // reshape.
    return inputType.getShape().back() == outputReshapeType.getShape().back();
  }

  bool isCommuteUpwardsFavorable(RMSNormOp op,
                                 ReshapeOp reshapeUser) const override {
    // Check that there's an input reshape that is the inverse of the output
    // reshape.
    auto inputReshape = op.getInput().getDefiningOp<ReshapeOp>();
    if (!inputReshape) {
      return false;
    }

    auto originalInputType =
        cast<RankedTensorType>(inputReshape.getInput().getType());
    auto outputReshapeType = cast<RankedTensorType>(reshapeUser.getType());

    // The shapes before input reshape and after output reshape must match
    // (i.e., the reshapes are inverses).
    if (originalInputType.getShape() != outputReshapeType.getShape()) {
      return false;
    }

    // We should commute if all users are identical reshapes.
    // This will allow us to eliminate the reshape pair.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(RMSNormOp, ReshapeOp) const override {
    // Downwards commute is not implemented for RMSNorm.
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
