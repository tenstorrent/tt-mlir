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
// - each all_reduce use is an all_slice op
// - all_reduce has exactly one reduction axis
// - each all_slice has slicing on exactly one dimension with one axis
// - The axes match between all_reduce and each all_slice
class AllReduceAllSliceToReduceScatterPattern
    : public OpRewritePattern<mlir::sdy::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::AllReduceOp allReduceOp,
                                PatternRewriter &rewriter) const override {
    // Check that all_reduce has exactly one reduction axis.
    auto reductionAxes = allReduceOp.getReductionAxes();
    if (reductionAxes.size() != 1) {
      return rewriter.notifyMatchFailure(
          allReduceOp,
          "all_reduce has multiple reduction axes, cannot fuse with all_slice");
    }
    mlir::StringRef reduceAxis = reductionAxes.front().getName();

    // Collect all users and validate they are all fusable all_slice ops.
    llvm::SmallVector<mlir::sdy::AllSliceOp> allSliceUsers;
    for (mlir::Operation *user : allReduceOp.getResult().getUsers()) {
      auto allSliceOp = mlir::dyn_cast<mlir::sdy::AllSliceOp>(user);
      if (!allSliceOp) {
        return rewriter.notifyMatchFailure(
            allReduceOp, "all_reduce has a user that is not an all_slice op");
      }

      // Validate all_slice has exactly one sliced dimension with matching axis.
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

      if (reduceAxis != sliceAxis) {
        return rewriter.notifyMatchFailure(
            allReduceOp,
            "all_reduce and all_slice operate on different mesh axes");
      }

      allSliceUsers.push_back(allSliceOp);
    }

    // All users validated - create reduce_scatter for each all_slice.
    for (mlir::sdy::AllSliceOp allSliceOp : allSliceUsers) {
      auto reduceScatterOp = rewriter.create<mlir::sdy::ReduceScatterOp>(
          allReduceOp.getLoc(), allSliceOp.getType(), allReduceOp.getOperand(),
          allSliceOp.getSlicingAxes(), allSliceOp.getOutSharding());
      rewriter.replaceOp(allSliceOp, reduceScatterOp.getResult());
    }

    // Erase the all_reduce (now has no uses).
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
