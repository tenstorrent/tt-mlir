// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOFUSINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class ConcatenateToBroadcastInDimFusionPattern
    : public OpRewritePattern<::mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<::mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(::mlir::stablehlo::ConcatenateOp concatOp,
                                ::mlir::PatternRewriter &rewriter) const final {
    // Check if fusable.
    if (!isFusable(concatOp)) {
      return failure();
    }
    // Fuse concat -> reshape.
    // To find broadcast dims, remove the concat dim from reshape output shape.
    ::mlir::stablehlo::ReshapeOp reshapeOp =
        mlir::dyn_cast<::mlir::stablehlo::ReshapeOp>(
            *concatOp.getResult().getUsers().begin());
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();
    uint64_t dim = concatOp.getDimension();
    SmallVector<int64_t> broadcastDims;
    for (size_t i = 0; i < reshapeOutputShape.size(); i++) {
      if (i != dim) {
        broadcastDims.push_back(i);
      }
    }
    // Create broadcast_in_dim op with concatenate input and broadcast dims.
    // Replace reshape op with broadcast_in_dim op.
    ::mlir::stablehlo::BroadcastInDimOp broadcastInDimOp =
        rewriter.create<::mlir::stablehlo::BroadcastInDimOp>(
            concatOp.getLoc(), reshapeOp.getResult().getType(),
            concatOp.getInputs()[0], broadcastDims);
    rewriter.replaceOp(reshapeOp, broadcastInDimOp.getResult());
    return success();
  }

private:
  bool isFusable(::mlir::stablehlo::ConcatenateOp concatOp) const {
    // Check all arguments of concat are the same.
    Value firstArg = concatOp.getInputs()[0];
    for (auto arg : concatOp.getInputs()) {
      if (arg != firstArg) {
        return false;
      }
    }

    // Check if any inputs or outputs of concat are sharded. If so, don't fuse.
    if (shardy_utils::isOpSharded(concatOp.getOperation())) {
      return false;
    }

    // Case 1: concat -> reshape.
    // Check concat has one user and that user is a reshape op.
    if (concatOp.getResult().hasOneUse()) {
      if (auto reshapeOp = mlir::dyn_cast<::mlir::stablehlo::ReshapeOp>(
              *concatOp.getResult().getUsers().begin())) {
        if (!checkConcatThenReshape(concatOp, reshapeOp)) {
          return false;
        }
        // Check if the reshape output is sharded. If so, don't fuse.
        if (shardy_utils::isOpSharded(reshapeOp.getOperation())) {
          return false;
        }
        return true;
      }
    }

    // todo(@ddilbazTT): add support for reshape -> concat.
    return false;
  }
  bool checkConcatThenReshape(::mlir::stablehlo::ConcatenateOp concatOp,
                              ::mlir::stablehlo::ReshapeOp reshapeOp) const {
    auto concatInput =
        mlir::cast<TypedValue<RankedTensorType>>(concatOp.getInputs()[0]);
    ArrayRef<int64_t> concatInputShape = concatInput.getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();
    // Number of dimensions of concat should be -1 number of dimensions of
    // reshape.
    if (concatInputShape.size() !=
        static_cast<size_t>(reshapeOutputShape.size() - 1)) {
      return false;
    }
    // If concat dim is i, reshape_output[i] = repeat_dim, reshape_output[i+1] =
    // concat_input[i]
    uint64_t dim = concatOp.getDimension();
    // num repeats is the number of concat inputs.
    int64_t numRepeats = static_cast<int64_t>(concatOp.getInputs().size());
    if (reshapeOutputShape[dim] != numRepeats) {
      return false;
    }
    if (reshapeOutputShape[dim + 1] != concatInputShape[dim]) {
      return false;
    }
    return true;
  }
};

struct StablehloFusingPass
    : public impl::StablehloFusingPassBase<StablehloFusingPass> {
public:
  using impl::StablehloFusingPassBase<
      StablehloFusingPass>::StablehloFusingPassBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConcatenateToBroadcastInDimFusionPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace mlir::tt::stablehlo
