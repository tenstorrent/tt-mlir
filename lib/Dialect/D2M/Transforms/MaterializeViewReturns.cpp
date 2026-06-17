// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <optional>
#include <type_traits>
#include <utility>

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

// Compute the full virtual-grid-mapping (VGM) forward/inverse affine maps for a
// materialization generic whose execution grid is `gridShape`. Grids that do
// not map directly onto the 2D physical worker grid (rank != 2, or any
// dimension exceeding the device grid) require core virtualization; without it,
// the d2m.core_index ops emitted per grid dimension cannot be lowered for dims
// >= 2 (D2MToTTKernel's CoreIndexRewriter relies on the grid's
// physical_to_virt_map, propagated by GenericRegionsToFuncs) and the generic
// fails verification in D2MToTTMetal. Returns std::nullopt when the grid maps
// directly onto the physical grid (no virtualization needed). The returned maps
// are the (grid+shard) VGM maps as produced by createCoreVirtMaps, i.e. the
// shape stored on EmptyOp/AllocOp VGM attributes.
std::optional<std::pair<AffineMap, AffineMap>>
computeViewVirtualGridMaps(OpBuilder &builder, Operation *op,
                           ArrayRef<int64_t> gridShape) {
  ttcore::DeviceAttr device = ttcore::lookupDevice(op);
  ArrayRef<int64_t> deviceGridShape = device.getWorkerGrid().getShape();
  if (!ttmlir::d2m::utils::grids::requiresVirtualGrid(gridShape,
                                                      deviceGridShape)) {
    return std::nullopt;
  }
  SmallVector<int64_t> physicalGridShape =
      utils::findLegalPhysicalGridForVolume(
          ttmlir::utils::volume<int64_t>(gridShape), deviceGridShape);
  TT_assertv(!physicalGridShape.empty(),
             "Unable to fit virtual grid {} onto device grid {} for view "
             "materialization",
             ttmlir::utils::formatIterable(gridShape, "x"),
             ttmlir::utils::formatIterable(deviceGridShape, "x"));
  return ttmlir::d2m::utils::grids::createCoreVirtMaps(
      builder.getContext(), gridShape, physicalGridShape);
}

// Materialize an unmaterialized view by inserting a datamovement generic op.
// View operations are representational (no actual data movement), so when a
// view is directly returned without being consumed by a generic op, we must
// insert a datamovement generic that forces the actual tensor transformation to
// occur.
Value materializeTensorView(OpBuilder &builder, Location loc,
                            Value viewResult) {
  auto tensorType = mlir::cast<RankedTensorType>(viewResult.getType());

  // This pass runs pre-bufferization, so view ops have MetalLayoutAttr.
  auto layout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  TT_assertv(layout != nullptr, "Expected MetalLayoutAttr pre-bufferization");

  // Allocate output storage for the materialized view result.
  // Create a new layout for fresh storage (no remapping on the layout attr).
  auto newLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), layout.getLogicalShape(), layout.getDimAlignments(),
      layout.getCollapsedIntervals(), layout.getMemorySpace(),
      layout.getMemoryLayout());
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
      loc, viewResult, emptyOp.getResult(), /*additionalArgs=*/ValueRange(),
      [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
        SmallVector<Value> indices =
            utils::buildGridIndices(builder, innerLoc, indexingMap);
        // operandAllocs are tensor.empty results with shard shapes,
        // one per generic operand.
        Type inputShardType = blockArgs[0].getType();
        Value inputBuffer = blockArgs[0];

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
      ThreadType::Unified, grid, SmallVector<int64_t>(rank, 1));

  return genericOp.getResult(0);
}

Value materializeMemrefView(OpBuilder &builder, Location loc,
                            ViewLayoutOp viewOp) {
  Value viewResult = viewOp.getResult();
  auto memrefType = mlir::cast<MemRefType>(viewResult.getType());
  SmallVector<int64_t> gridShape =
      llvm::to_vector(ttcore::getGridShape(viewResult));
  SmallVector<int64_t> shardShape =
      llvm::to_vector(ttcore::getShardShape(viewResult));
  auto storageLayout =
      ttcore::ShardLayoutAttr::get(shardShape, memrefType.getElementType(), 1);
  SmallVector<int64_t> storageShape = gridShape;
  storageShape.append(shardShape.begin(), shardShape.end());
  auto storageType =
      MemRefType::get(storageShape, memrefType.getElementType(), storageLayout,
                      memrefType.getMemorySpace());

  auto storageAlloc = builder.create<memref::AllocOp>(loc, storageType);

  std::optional<std::pair<AffineMap, AffineMap>> vgm =
      computeViewVirtualGridMaps(builder, viewOp, gridShape);

  ttcore::GridAttr gridAttr;
  if (vgm) {
    // The grid requires core virtualization. Stamp the full VGM maps onto the
    // output storage (the generic verifier requires the output operand to carry
    // a VGM matching the grid) and derive the grid-only GridAttr maps from
    // them.
    auto [forwardMap, inverseMap] = *vgm;
    storageAlloc->setAttr(utils::kVirtualGridForwardMappingAttr,
                          AffineMapAttr::get(forwardMap));
    storageAlloc->setAttr(utils::kVirtualGridInverseMappingAttr,
                          AffineMapAttr::get(inverseMap));
    std::optional<std::pair<AffineMap, AffineMap>> gridMaps =
        utils::getGridMapsFromVirtualGridMapping(storageAlloc.getResult(),
                                                 gridShape);
    TT_assertv(gridMaps.has_value(),
               "Failed to derive grid maps from virtual grid mapping for view "
               "materialization");
    gridAttr = ttcore::GridAttr::get(builder.getContext(), gridShape,
                                     gridMaps->first, gridMaps->second);
  } else {
    gridAttr = builder.getAttr<ttcore::GridAttr>(gridShape);
  }
  Value materialized = storageAlloc.getResult();
  ArrayAttr emptyArray = builder.getArrayAttr({});
  ArrayAttr threads =
      builder.getArrayAttr(builder.getAttr<ThreadAttr>(ThreadType::Unified));

  auto genericOp = builder.create<GenericOp>(
      loc, TypeRange{}, ValueRange{viewResult}, ValueRange{materialized},
      ValueRange{}, gridAttr, emptyArray, emptyArray, emptyArray, threads,
      /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);

  Region &region = genericOp.getRegion(0);
  builder.createBlock(&region);
  builder.setInsertionPointToStart(&region.front());

  auto localType =
      MemRefType::get(shardShape, memrefType.getElementType(),
                      MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
  Value localBuffer =
      builder.create<memref::AllocOp>(loc, localType).getResult();

  SmallVector<Value> remoteIndices;
  remoteIndices.reserve(gridShape.size());
  for (size_t dim = 0; dim < gridShape.size(); ++dim) {
    remoteIndices.push_back(
        builder.create<CoreIndexOp>(loc, static_cast<int64_t>(dim)));
  }

  builder.create<RemoteLoadOp>(loc, localBuffer, viewResult, remoteIndices);
  builder.create<RemoteStoreOp>(
      loc, /*resultTypes=*/TypeRange{}, materialized, remoteIndices,
      localBuffer, /*cb=*/Value{}, /*startDevice=*/ValueRange{},
      /*deviceMcastShape=*/ValueRange{}, /*semaphore=*/Value{},
      /*semaphoreIndices=*/ValueRange{});

  return materialized;
}

std::optional<Value> materializeView(OpBuilder &builder, Location loc,
                                     Value viewResult) {
  Operation *definingOp = viewResult.getDefiningOp();
  if (!isViewOp(definingOp)) {
    return std::nullopt;
  }

  OpBuilder::InsertionGuard guard(builder);
  if (mlir::isa<RankedTensorType>(viewResult.getType())) {
    return materializeTensorView(builder, loc, viewResult);
  }

  if (auto viewOp = mlir::dyn_cast<ViewLayoutOp>(definingOp)) {
    if (mlir::isa<MemRefType>(viewResult.getType())) {
      return materializeMemrefView(builder, loc, viewOp);
    }
  }

  return std::nullopt;
}

class MaterializeReturnViewPattern : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(returnOp);
    SmallVector<std::pair<unsigned, Value>> replacements;

    // Inspect each return operand to determine if it needs materialization.
    for (OpOperand &opOperand : returnOp->getOpOperands()) {
      Operation *definingOp = opOperand.get().getDefiningOp();
      if (!isViewOp(definingOp)) {
        continue;
      }

      // Insert a generic op to materialize the view before returning. This
      // ensures the tensor transformation represented by the view actually
      // occurs, rather than just being a symbolic operation.
      std::optional<Value> materialized =
          materializeView(rewriter, returnOp.getLoc(), opOperand.get());
      if (materialized) {
        replacements.emplace_back(opOperand.getOperandNumber(), *materialized);
      }
    }

    if (replacements.empty()) {
      return failure();
    }

    rewriter.modifyOpInPlace(returnOp, [&]() {
      for (auto [operandNumber, materialized] : replacements) {
        returnOp->setOperand(operandNumber, materialized);
      }
    });
    return success();
  }
};

template <typename OpTy>
class MaterializeToHostViewPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<OpTy, d2m::ToLayoutOp>) {
      if (!op.isDeviceToHost()) {
        return failure();
      }
    }

    Value toHostInput = op->getOperand(0);
    Operation *inputDefiningOp = toHostInput.getDefiningOp();
    if (!isViewOp(inputDefiningOp)) {
      return failure();
    }

    // Materialize the view before the device-to-host transfer.
    rewriter.setInsertionPoint(op);
    std::optional<Value> materialized =
        materializeView(rewriter, op->getLoc(), toHostInput);
    if (!materialized) {
      return failure();
    }
    // Update the ToHostOp/ToLayoutOp to use the materialized value.
    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, *materialized); });
    return success();
  }
};

class MaterializeViewReturnsPass
    : public impl::D2MMaterializeViewReturnsBase<MaterializeViewReturnsPass> {
public:
  using impl::D2MMaterializeViewReturnsBase<
      MaterializeViewReturnsPass>::D2MMaterializeViewReturnsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MaterializeReturnViewPattern,
                 MaterializeToHostViewPattern<d2m::ToHostOp>,
                 MaterializeToHostViewPattern<d2m::ToLayoutOp>>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::tt::d2m
