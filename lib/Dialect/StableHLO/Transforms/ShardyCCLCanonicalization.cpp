// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/ir/dialect.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_SHARDYCCLCANONICALIZATIONPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// AllReduceAllSliceToReduceScatterPattern - fuses all_reduce + all_slice into
// reduce_scatter when criteria are met:
// - all_reduce has exactly one use (the all_slice)
// - all_reduce has exactly one reduction axis
// - all_slice has slicing on exactly one dimension with one axis
// - The axes match between all_reduce and all_slice
class AllReduceAllSliceToReduceScatterPattern
    : public OpRewritePattern<mlir::sdy::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::AllReduceOp allReduceOp,
                                PatternRewriter &rewriter) const override {
    // Check that all_reduce has exactly one use.
    if (!allReduceOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          allReduceOp, "all_reduce has multiple uses, cannot fuse");
    }

    // Check if the single user is an all_slice.
    mlir::Operation *user = *allReduceOp.getResult().getUsers().begin();
    auto allSliceOp = mlir::dyn_cast<mlir::sdy::AllSliceOp>(user);
    if (!allSliceOp) {
      return rewriter.notifyMatchFailure(
          allReduceOp, "all_reduce user is not an all_slice op");
    }

    // Check that all_reduce has exactly one reduction axis.
    auto reductionAxes = allReduceOp.getReductionAxes();
    if (reductionAxes.size() != 1) {
      return rewriter.notifyMatchFailure(
          allReduceOp,
          "all_reduce has multiple reduction axes, cannot fuse with all_slice");
    }
    mlir::StringRef reduceAxis = reductionAxes.front().getName();

    // Get the all_slice's slicing axes and find the single sliced dimension
    // with a single mesh axis.
    auto axesPerDim = allSliceOp.getSlicingAxes();
    int64_t sliceDim = -1;
    mlir::StringRef sliceAxis;

    for (auto it = axesPerDim.begin(), e = axesPerDim.end(); it != e; ++it) {
      llvm::ArrayRef<mlir::sdy::AxisRefAttr> axisRefs = it->getValue();
      if (axisRefs.empty()) {
        continue;
      }
      if (axisRefs.size() > 1) {
        return rewriter.notifyMatchFailure(
            allReduceOp, "all_slice has multiple axes on a single dimension");
      }
      if (sliceDim != -1) {
        return rewriter.notifyMatchFailure(
            allReduceOp,
            "all_slice has slicing on multiple dimensions, cannot fuse");
      }
      sliceDim = static_cast<int64_t>(std::distance(axesPerDim.begin(), it));
      sliceAxis = axisRefs.front().getName();
    }

    if (sliceDim == -1) {
      return rewriter.notifyMatchFailure(allReduceOp,
                                         "all_slice has no slicing axes");
    }

    // Check that the reduction axis matches the slice axis.
    if (reduceAxis != sliceAxis) {
      return rewriter.notifyMatchFailure(
          allReduceOp,
          "all_reduce and all_slice operate on different mesh axes");
    }

    // All criteria met - create sdy.reduce_scatter.
    auto reduceScatterOp = rewriter.create<mlir::sdy::ReduceScatterOp>(
        allReduceOp.getLoc(), allSliceOp.getType(), allReduceOp.getOperand(),
        allSliceOp.getSlicingAxes(), allSliceOp.getOutSharding());

    // Replace all_slice with reduce_scatter result and erase both ops.
    rewriter.replaceOp(allSliceOp, reduceScatterOp.getResult());
    rewriter.eraseOp(allReduceOp);

    return success();
  }
};

struct ShardyCCLCanonicalizationPass
    : public impl::ShardyCCLCanonicalizationPassBase<
          ShardyCCLCanonicalizationPass> {
public:
  using impl::ShardyCCLCanonicalizationPassBase<
      ShardyCCLCanonicalizationPass>::ShardyCCLCanonicalizationPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<AllReduceAllSliceToReduceScatterPattern>(ctx);

    GreedyRewriteConfig config;
    config.enableConstantCSE(false);

    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::tt::stablehlo
