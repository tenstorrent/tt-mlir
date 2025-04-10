// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_FUSING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if all users of an op are of same type
template <typename UserOp>
static bool allUsers(Operation *srcOp) {
  auto check = [](Operation *op) { return isa<UserOp>(op); };
  return all_of(srcOp->getResult(0).getUsers(), check);
}

// Given srcOp and its user userOp we want to return userOp operands
// minus srcOp and dps operand
static OpOperand *getOtherOperand(Operation *srcOp, Operation *userOp) {
  DestinationStyleOpInterface dpsUserOp =
      cast<DestinationStyleOpInterface>(userOp);

  for (auto &opOperand : userOp->getOpOperands()) {
    if (dpsUserOp.isDpsInit(&opOperand)) {
      continue;
    }

    if (Operation *op = opOperand.get().getDefiningOp()) {
      if (op != srcOp) {
        return &opOperand;
      }
    } else {
      // This is block argument
      return &opOperand;
    }
  }

  return nullptr;
}

static RankedTensorType getRTT(OpOperand *opOperand) {
  return cast<RankedTensorType>(opOperand->get().getType());
}

// Check if we can fuse conv2d followed by add into conv2d with bias
class Conv2DAddPattern : public OpRewritePattern<Conv2dOp> {
  using OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(Conv2dOp srcOp,
                                PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    AddOp addOp = cast<AddOp>(*srcOp.getResult().getUsers().begin());
    OpOperand *otherOperand = getOtherOperand(srcOp, addOp);
    Value newBias = otherOperand->get();

    Value newConv = rewriter.replaceOpWithNewOp<Conv2dOp>(
        srcOp, srcOp.getResult().getType(), srcOp.getInput(), srcOp.getWeight(),
        newBias, srcOp.getOutput(), srcOp.getStride(), srcOp.getPadding(),
        srcOp.getDilation(), srcOp.getGroups());

    rewriter.replaceAllUsesWith(addOp.getResult(0), newConv);

    return success();
  }

private:
  bool isFusable(Conv2dOp srcOp) const {
    // If bias already exists on Conv2d we cannot fuse
    if (srcOp.getBias().getImpl()) {
      return false;
    }

    // If conv2d has more than one use we cannot fuse
    // If it's only user is not AddOp we cannot fuse
    if (!srcOp.getResult().hasOneUse() || !allUsers<AddOp>(srcOp)) {
      return false;
    }

    // If we have this IR:
    // %0 = empty()
    // %1 = conv2d()
    // %2 = add(%1, %0)
    // we want to get type of %0 and check if it is compatible with conv2d bias
    Operation *addUser = *srcOp.getResult().getUsers().begin();
    OpOperand *otherOperand = getOtherOperand(srcOp, addUser);
    Operation *otherOperation = otherOperand->get().getDefiningOp();

    // If otherOperand is not defined by an operation, its a block argument
    // and we can fuse if bias is compatible with its shape.
    if (otherOperation == nullptr) {
      RankedTensorType otherType = getRTT(otherOperand);
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

class FusingPass : public impl::FusingBase<FusingPass> {
public:
  using impl::FusingBase<FusingPass>::FusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<Conv2DAddPattern>(&getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttir
