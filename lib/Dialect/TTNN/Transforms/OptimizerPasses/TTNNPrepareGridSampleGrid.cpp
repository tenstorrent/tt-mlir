// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPAREGRIDSAMPLEGRID
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {
class TTNNPrepareGridSampleGrid
    : public impl::TTNNPrepareGridSampleGridBase<TTNNPrepareGridSampleGrid> {
public:
  using impl::TTNNPrepareGridSampleGridBase<
      TTNNPrepareGridSampleGrid>::TTNNPrepareGridSampleGridBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Collect first to avoid modifying the IR during the walk.
    SmallVector<GridSampleOp, 4> gsOps;
    moduleOp.walk([&](GridSampleOp op) { gsOps.push_back(op); });

    for (GridSampleOp gsOp : gsOps) {
      processGridSampleOp(gsOp);
    }
  }

private:
  void processGridSampleOp(GridSampleOp gsOp) {
    std::string mode = gsOp.getMode().str();
    bool alignCorners = gsOp.getAlignCorners();

    // Only needed for nearest mode or when align_corners is true.
    if (mode != "nearest" && !alignCorners) {
      return;
    }

    // Already has a PrepareGridSampleGridOp feeding it.
    if (mlir::isa_and_nonnull<PrepareGridSampleGridOp>(
            gsOp.getGrid().getDefiningOp())) {
      return;
    }

    auto inputType =
        mlir::cast<RankedTensorType>(gsOp.getInput().getType());
    auto inputShape = inputType.getShape();
    int32_t inputN = static_cast<int32_t>(inputShape[0]);
    int32_t inputH = static_cast<int32_t>(inputShape[1]);
    int32_t inputW = static_cast<int32_t>(inputShape[2]);
    int32_t inputC = static_cast<int32_t>(inputShape[3]);

    // For nearest mode, prepare_grid_sample_grid preserves the grid shape and
    // layout, so reuse the grid's existing type as the result type.
    mlir::Type resultType = gsOp.getGrid().getType();

    OpBuilder builder(gsOp);
    auto prepareOp = builder.create<PrepareGridSampleGridOp>(
        ttmlir::utils::appendLocationSuffix(
            gsOp.getLoc(), "_prepare_grid_sample_grid"),
        resultType, gsOp.getGrid(),
        builder.getI32IntegerAttr(inputN), builder.getI32IntegerAttr(inputH),
        builder.getI32IntegerAttr(inputW), builder.getI32IntegerAttr(inputC),
        gsOp.getModeAttr(), gsOp.getPaddingModeAttr(),
        gsOp.getAlignCornersAttr());

    gsOp.getGridMutable().assign(prepareOp.getResult());
    gsOp.setUsePrecomputedGrid(true);
  }
};
} // namespace
} // namespace mlir::tt::ttnn
