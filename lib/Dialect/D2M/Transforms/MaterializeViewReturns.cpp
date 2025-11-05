// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMATERIALIZEVIEWRETURNS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

bool isViewOp(Operation *op) {
  return mlir::isa_and_nonnull<d2m::ViewOpInterface>(op);
}

// Extract the grid attribute from a tensor's metal layout encoding.
ttcore::GridAttr getGridFromType(RankedTensorType type) {
  auto layout =
      mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(type.getEncoding());
  if (!layout) {
    return nullptr;
  }

  auto gridShape = layout.getGridShape(type);
  if (gridShape.empty()) {
    return nullptr;
  }

  MLIRContext *ctx = type.getContext();
  return ttcore::GridAttr::get(ctx, gridShape);
}

// Materialize an unmaterialized view by inserting a datamovement generic op.
// View operations are representational (no actual data movement), so when a
// view is directly returned without being consumed by a generic op, we must
// insert a datamovement generic that forces the actual tensor transformation to
// occur.
Value materializeView(OpBuilder &builder, Location loc, Value viewResult) {
  auto tensorType = mlir::cast<RankedTensorType>(viewResult.getType());

  // Allocate output storage for the materialized view result.
  auto layout =
      mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  auto emptyOp = builder.create<d2m::EmptyOp>(
      loc, tensorType.getShape(), tensorType.getElementType(), layout);

  // Extract the grid from the tensor's layout to determine core distribution.
  ttcore::GridAttr grid = getGridFromType(tensorType);
  if (!grid) {
    // Fall back to single-core execution when grid information is unavailable.
    SmallVector<int64_t> defaultGrid = {1, 1};
    grid = ttcore::GridAttr::get(builder.getContext(), defaultGrid);
  }

  // Build identity affine maps for parallel iteration over all grid dimensions.
  size_t rank = grid.getShape().size();
  ArrayAttr indexingMaps, iteratorTypes;
  std::tie(indexingMaps, iteratorTypes) =
      GenericOp::buildParallelAffineMapsAndIteratorTypes(builder,
                                                         /*arity=*/2, rank);

  // Create a datamovement generic op that materializes the view.
  // The region simply reserves the output circular buffer and yields it,
  // causing the datamovement hardware to fetch data according to the view's
  // affine map transformation.
  auto genericOp = builder.create<GenericOp>(
      loc, viewResult, emptyOp.getResult(),
      [](OpBuilder &builder, Location loc, ValueRange blockArgs) {
        Value output =
            builder.create<d2m::ReserveOp>(loc, blockArgs[1]).getResult();
        builder.create<d2m::YieldOp>(loc, output);
      },
      ThreadType::Datamovement, grid, SmallVector<int64_t>{1, 1});

  return genericOp.getResult(0);
}

class MaterializeViewReturnsPass
    : public impl::D2MMaterializeViewReturnsBase<MaterializeViewReturnsPass> {
public:
  using impl::D2MMaterializeViewReturnsBase<
      MaterializeViewReturnsPass>::D2MMaterializeViewReturnsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    // Process each function in the module to find unmaterialized view returns.
    module.walk([&](func::FuncOp funcOp) {
      funcOp.walk([&](func::ReturnOp returnOp) {
        SmallVector<Value> newOperands;
        bool modified = false;

        // Inspect each return value to determine if it needs materialization.
        for (Value operand : returnOp.getOperands()) {
          Operation *definingOp = operand.getDefiningOp();

          if (isViewOp(definingOp)) {
            // Insert a generic op to materialize the view before returning.
            // This ensures the tensor transformation represented by the view
            // actually occurs, rather than just being a symbolic operation.
            builder.setInsertionPoint(returnOp);
            Value materialized =
                materializeView(builder, returnOp.getLoc(), operand);
            newOperands.push_back(materialized);
            modified = true;
          } else {
            // Non-view values can be returned directly without modification.
            newOperands.push_back(operand);
          }
        }

        // Replace the return op if any views were materialized.
        if (modified) {
          builder.setInsertionPoint(returnOp);
          builder.create<func::ReturnOp>(returnOp.getLoc(), newOperands);
          returnOp.erase();
        }
      });
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
