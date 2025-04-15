// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace ttmlir::utils {

namespace fusing {

inline mlir::RankedTensorType getRTT(mlir::OpOperand *opOperand) {
  return mlir::cast<mlir::RankedTensorType>(opOperand->get().getType());
}

// Check if we can fuse conv2d followed by add into conv2d with bias.
template <typename Conv2dOpTy, typename AddOpTy>
class Conv2dAddPattern : public mlir::OpRewritePattern<Conv2dOpTy> {
  using mlir::OpRewritePattern<Conv2dOpTy>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOpTy srcOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return mlir::failure();
    }

    AddOpTy addOp = mlir::cast<AddOpTy>(*srcOp.getResult().getUsers().begin());
    mlir::OpOperand *otherOperand = getOtherOperands(srcOp, addOp).front();
    mlir::Value newBias = otherOperand->get();

    mlir::Value newConv = replaceConv2d(rewriter, srcOp, newBias);

    rewriter.replaceAllUsesWith(getAddResult(addOp), newConv);

    return mlir::success();
  }

  virtual mlir::Value replaceConv2d(mlir::PatternRewriter &rewriter,
                                    Conv2dOpTy srcOp,
                                    mlir::Value bias) const = 0;

  // Remove this once https://github.com/tenstorrent/tt-mlir/issues/2829
  virtual mlir::Value getAddResult(AddOpTy addOp) const = 0;

private:
  bool isFusable(Conv2dOpTy srcOp) const {
    // If bias already exists on Conv2d we cannot fuse.
    if (srcOp.getBias().getImpl()) {
      return false;
    }

    // If conv2d has more than one use we cannot fuse.
    // If it's only user is not AddOp we cannot fuse.
    if (!srcOp.getResult().hasOneUse() || !allUsers<AddOpTy>(srcOp)) {
      return false;
    }

    // If we have this IR:
    // %0 = empty()
    // %1 = conv2d()
    // %2 = add(%1, %0)
    // we want to get type of %0 and check if it is compatible with conv2d bias.
    mlir::Operation *addUser = *srcOp.getResult().getUsers().begin();
    mlir::OpOperand *otherOperand = getOtherOperands(srcOp, addUser).front();
    mlir::Operation *otherOperation = otherOperand->get().getDefiningOp();

    // If otherOperand is not defined by an operation, its a block argument
    // and we can fuse if bias is compatible with its shape.
    if (otherOperation == nullptr) {
      mlir::RankedTensorType otherType = getRTT(otherOperand);
      return srcOp.verifyBias(otherType.getShape());
    }

    // If we have this IR:
    // %0 = conv2d()
    // %1 = empty()
    // %2 = add(%0, %1)
    // We cannot fuse since %1 comes after conv2d. In this simple case
    // we can move empty above covn2d, but in general case it would
    // require more complex analysis.
    if (!otherOperation->isBeforeInBlock(srcOp)) {
      return false;
    }

    return srcOp.verifyBias(getRTT(otherOperand).getShape());
  }
};

} // namespace fusing

} // namespace ttmlir::utils
