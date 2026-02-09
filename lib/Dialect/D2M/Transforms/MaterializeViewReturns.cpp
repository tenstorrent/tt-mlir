// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  TT_assert(!gridShape.empty());

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

  // This pass runs pre-bufferization, so view ops have MetalLayoutAttr.
  auto layout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  TT_assertv(layout != nullptr, "Expected MetalLayoutAttr pre-bufferization");

  // Allocate output storage for the materialized view result.
  // Create a new layout for fresh storage (no remapping on the layout attr).
  auto newLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), layout.getLogicalShape(), layout.getDimAlignments(),
      layout.getCollapsedIntervals(), layout.getOobVal(),
      layout.getMemorySpace(), layout.getMemoryLayout());
  auto emptyOp = builder.create<d2m::EmptyOp>(
      loc, tensorType.getShape(), tensorType.getElementType(), newLayout);

  // Extract the grid from the tensor's layout to determine core distribution.
  ttcore::GridAttr grid = getGridFromType(tensorType);
  TT_assert(grid != nullptr);

  // Build identity affine maps for parallel iteration over all grid dimensions.
  size_t rank = grid.getShape().size();
  ArrayAttr indexingMaps, iteratorTypes;
  std::tie(indexingMaps, iteratorTypes) =
      GenericOp::buildParallelAffineMapsAndIteratorTypes(builder,
                                                         /*arity=*/2, rank);

  // Create a datamovement generic op that materializes the view.
  auto indexingMapAttr = mlir::cast<AffineMapAttr>(indexingMaps[0]);
  AffineMap indexingMap = indexingMapAttr.getValue();
  auto genericOp = builder.create<GenericOp>(
      loc, viewResult, emptyOp.getResult(),
      [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
        SmallVector<Value> indices =
            utils::buildGridIndices(builder, innerLoc, indexingMap);
        auto inputCbType = mlir::cast<d2m::CBType>(blockArgs[0].getType());
        auto inputShardType = inputCbType.getUnderlying();

        // Create a buffer for the load result
        auto inputTensorType = mlir::cast<RankedTensorType>(inputShardType);
        auto inputBufferOp = builder.create<tensor::EmptyOp>(
            innerLoc, inputTensorType.getShape(),
            inputTensorType.getElementType());
        Value inputBuffer = inputBufferOp.getResult();

        Value loadedData =
            builder
                .create<RemoteLoadOp>(innerLoc, inputShardType, inputBuffer,
                                      viewResult, indices)
                .getResult();
        Value storeResult =
            builder
                .create<RemoteStoreOp>(innerLoc, emptyOp.getType(),
                                       emptyOp.getResult(), indices, loadedData)
                .getResult();
        builder.create<d2m::YieldOp>(innerLoc, storeResult);
      },
      ThreadType::Unified, grid, SmallVector<int64_t>{1, 1});

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
        builder.setInsertionPoint(returnOp);

        // Inspect each return operand to determine if it needs materialization.
        for (OpOperand &opOperand : returnOp->getOpOperands()) {
          Operation *definingOp = opOperand.get().getDefiningOp();

          // Case 1: Direct view return (should not happen with proper
          // pipelines).
          if (isViewOp(definingOp)) {
            // Insert a generic op to materialize the view before returning.
            // This ensures the tensor transformation represented by the view
            // actually occurs, rather than just being a symbolic operation.
            Value materialized =
                materializeView(builder, returnOp.getLoc(), opOperand.get());
            opOperand.set(materialized);
            continue;
          }

          // Case 2: View consumed by device-to-host ToHostOp before return.
          // Pattern: %view = view_layout ... -> %host = to_host %view ->
          // return %host. We need to materialize the view BEFORE the
          // device-to-host transfer.
          auto toLayoutOp =
              mlir::dyn_cast_if_present<d2m::ToLayoutOp>(definingOp);
          bool isToHostOp = mlir::isa_and_nonnull<d2m::ToHostOp>(definingOp) ||
                            (toLayoutOp && toLayoutOp.isDeviceToHost());
          if (isToHostOp) {
            Value toHostInput = definingOp->getOperand(0);
            Operation *inputDefiningOp = toHostInput.getDefiningOp();

            if (isViewOp(inputDefiningOp)) {
              // Materialize the view before the device-to-host transfer.
              builder.setInsertionPoint(definingOp);
              Value materialized =
                  materializeView(builder, definingOp->getLoc(), toHostInput);

              // Update the ToHostOp to use the materialized value.
              definingOp->setOperand(0, materialized);
            }
          }
        }
      });
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
