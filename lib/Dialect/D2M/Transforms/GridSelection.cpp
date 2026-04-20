// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/GridAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::d2m {

// ----------------------------------------------------------------------------
// Transform helpers — these modify the IR to apply grid decisions.
// ----------------------------------------------------------------------------

// Update a ToLayoutOp and its associated EmptyOp to use a specified grid by
// recreating the MetalLayoutAttr with the given grid and proper dimension
// alignments.
static void optimizeToLayoutGrid(d2m::ToLayoutOp toLayoutOp,
                                 ArrayRef<int64_t> targetGrid,
                                 ArrayRef<int64_t> effectiveTargetGrid,
                                 bool ttnnMode, ArrayRef<int64_t> optimalGrid,
                                 OpBuilder &builder) {
  auto emptyOp = toLayoutOp.getOutput().getDefiningOp<d2m::EmptyOp>();
  if (!emptyOp) {
    return;
  }

  // Check if we're already at the target grid.
  auto emptyType = mlir::cast<mlir::RankedTensorType>(emptyOp.getType());
  if (emptyType.getShape().take_front(2) == llvm::ArrayRef(optimalGrid)) {
    return;
  }

  auto outputType = mlir::cast<mlir::RankedTensorType>(toLayoutOp.getType(0));
  auto oldLayout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(outputType.getEncoding());
  if (!oldLayout) {
    return;
  }

  bool needsOptimization = false;
  for (int64_t g : optimalGrid) {
    if (g > 1) {
      needsOptimization = true;
      break;
    }
  }

  if (!needsOptimization) {
    return;
  }

  RankedTensorType newTensorType = utils::tensorWithOptimalGrid(
      outputType, targetGrid, ttnnMode, optimalGrid, builder);
  builder.setInsertionPoint(emptyOp);

  // Determine if the chosen grid is virtual (exceeds 2D device bounds or is
  // ND). Note: VGM is NOT propagated from the to_layout's input here — the
  // output EmptyOp has its own grid/shard strategy. VGM for DMA addresses
  // is traced through the stream's input at DMA lowering time.
  mlir::AffineMapAttr virtualGridInverseMapping;
  mlir::AffineMapAttr virtualGridForwardMapping;
  auto device = ttcore::lookupDevice(toLayoutOp);
  auto workerGridShape = device.getWorkerGrid().getShape();
  bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
      optimalGrid, workerGridShape);
  if (isVirtual) {
    auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
        ttmlir::utils::volume<int64_t>(optimalGrid), effectiveTargetGrid);
    TT_assertv(!physicalGridShape.empty(),
               "Unable to find 2D rect that can fit virtual grid {} within "
               "device grid {}",
               ttmlir::utils::formatIterable(optimalGrid, "x"),
               ttmlir::utils::formatIterable(effectiveTargetGrid, "x"));
    auto [forwardMap, inverseMap] =
        ttmlir::d2m::utils::grids::createCoreVirtMaps(
            builder.getContext(), optimalGrid, physicalGridShape);
    virtualGridInverseMapping = AffineMapAttr::get(inverseMap);
    virtualGridForwardMapping = AffineMapAttr::get(forwardMap);
  }

  auto newEmptyOp = builder.create<d2m::EmptyOp>(
      emptyOp.getLoc(), newTensorType, virtualGridInverseMapping,
      virtualGridForwardMapping);

  builder.setInsertionPoint(toLayoutOp);
  auto newToLayoutOp = builder.create<d2m::ToLayoutOp>(
      toLayoutOp.getLoc(), toLayoutOp.getInput(), newEmptyOp);

  // Reblock it back to original shape to preserve IR correctness.
  // The view chain that applyViews composes through depends on this
  // ViewLayoutOp existing between the optimal-grid ToLayout and downstream
  // ViewLayoutOps / GenericOps.
  auto viewOutputType = mlir::cast<RankedTensorType>(utils::reblockShapedType(
      newTensorType, oldLayout.getGridShape(outputType)));
  auto reblockMap = ttmlir::utils::calculateReblockMap(
      newTensorType.getShape(), viewOutputType.getShape(),
      builder.getContext());
  auto view = builder.create<d2m::ViewLayoutOp>(
      toLayoutOp.getLoc(), viewOutputType, newToLayoutOp.getResult(0),
      reblockMap, /*reinterpretLayout=*/false);

  // We expect the ToLayout to be used in one of two ways:
  // 1. Directly by a single GenericOp (or operations within its region)
  // 2. By a view_layout operation, where the result is then
  //    used by a single GenericOp
  d2m::GenericOp parentGeneric = nullptr;

  for (auto &use : toLayoutOp.getResult(0).getUses()) {
    mlir::Operation *user = use.getOwner();

    // Check if this use is by a view_layout operation (e.g., tensor
    // manipulation ops that express data rearrangement as a view).
    if (auto viewLayoutOp = mlir::dyn_cast<d2m::ViewLayoutOp>(user)) {
      // Walk through view_layout users to find the parent GenericOp.
      for (auto &viewUse : viewLayoutOp.getResult().getUses()) {
        mlir::Operation *viewUser = viewUse.getOwner();
        d2m::GenericOp viewUseGeneric =
            mlir::dyn_cast<d2m::GenericOp>(viewUser);
        if (!viewUseGeneric) {
          viewUseGeneric = viewUser->getParentOfType<d2m::GenericOp>();
        }
        if (viewUseGeneric) {
          if (!parentGeneric) {
            parentGeneric = viewUseGeneric;
          } else if (parentGeneric != viewUseGeneric) {
            TT_assertv(false,
                       "ToLayout should only be used within one GenericOp");
          }
        }
      }
      continue;
    }

    // Skip Spatial: it only lists the tensor on ins/outs. The real use is
    // inside the nested d2m.generic, which we handle in the next block.
    if (mlir::isa<d2m::SpatialOp>(user)) {
      continue;
    }

    // Find the parent GenericOp for this use.
    // The user might be the GenericOp itself (if it's an operand), or
    // it might be an operation nested within the GenericOp's regions.
    d2m::GenericOp useGeneric = mlir::dyn_cast<d2m::GenericOp>(user);
    if (!useGeneric) {
      useGeneric = user->getParentOfType<d2m::GenericOp>();
    }

    TT_assertv(useGeneric,
               "ToLayout result must be used by a single GenericOp or a "
               "single ViewLayout that feeds a single GenericOp");

    if (!parentGeneric) {
      parentGeneric = useGeneric;
    } else if (parentGeneric != useGeneric) {
      // Use is within a different GenericOp
      TT_assertv(false, "ToLayout should only be used within one GenericOp");
    }
  }
  toLayoutOp.getResult(0).replaceAllUsesWith(view.getResult());

  toLayoutOp.erase();
  if (emptyOp.getResult().use_empty()) {
    emptyOp.erase();
  }
}

static void insertViewForTTNNDRAMTensor(Value operand,
                                        ArrayRef<int64_t> optimalGrid,
                                        OpBuilder &builder) {
  while (auto viewOp = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    auto originalOperand = viewOp.getInput();
    viewOp.getResult().replaceAllUsesWith(originalOperand);
    viewOp.erase();
    operand = originalOperand;
  }
  // Do not "restream" metal -> ttnn -> metal sequences. This happens when the
  // output of a generic is the input to another generic. The output is
  // already streamed, but the cast back to ttnn silently erases the index
  // map. Instead, we just forward the already streamed metal tensor to the
  // current generic.
  auto castOp = operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
  auto producerCastOp =
      castOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
  if (producerCastOp) {
    castOp.getResult().replaceAllUsesExcept(producerCastOp.getInput(),
                                            producerCastOp);
    return;
  }

  auto metalTensor = mlir::cast<mlir::RankedTensorType>(operand.getType());
  auto baseMetalLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());

  // TTNN DRAM interleaved tensors are represented as having a 1x1 grid.
  llvm::SmallVector<int64_t> unitGridShape{1, 1};
  llvm::SmallVector<int64_t> unShardedShapeWithGrid =
      baseMetalLayout.getDeviceShape(unitGridShape,
                                     ttcore::TileType::getDefaultShape());

  llvm::SmallVector<int64_t> fakeShardedShape = baseMetalLayout.getDeviceShape(
      optimalGrid, ttcore::TileType::getDefaultShape());

  AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      unShardedShapeWithGrid, fakeShardedShape, builder.getContext());

  auto viewOutputLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), baseMetalLayout.getLogicalShape(),
      baseMetalLayout.getOobVal(), ttcore::MemorySpace::DeviceDRAM,
      ttcore::TensorMemoryLayout::Interleaved,
      baseMetalLayout.getCollapsedIntervals(),
      baseMetalLayout.getDimAlignments());

  auto viewOutputTensor = mlir::RankedTensorType::get(
      fakeShardedShape, metalTensor.getElementType(), viewOutputLayout);

  builder.setInsertionPointAfter(castOp);
  auto viewOp = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), viewOutputTensor, castOp.getResult(),
      AffineMapAttr::get(reblockMap));
  castOp.getResult().replaceAllUsesExcept(viewOp.getResult(), viewOp);
}

static void
optimizeTTNNMetalLayoutCastOpGrid(ttir::TTNNMetalLayoutCastOp castOp,
                                  ArrayRef<int64_t> optimalGrid,
                                  OpBuilder &builder) {
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(castOp.getResult().getType());
  auto outputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(outputType.getEncoding());

  if (optimalGrid == outputLayout.getGridShape(outputType)) {
    // Already at target grid shape.
    return;
  }

  auto newTensorType = mlir::cast<RankedTensorType>(
      utils::reblockShapedType(outputType, optimalGrid));

  mlir::AffineMapAttr gridRemapping =
      AffineMapAttr::get(ttmlir::utils::calculateReblockMap(
          outputType.getShape(), newTensorType.getShape(),
          builder.getContext()));

  builder.setInsertionPointAfter(castOp);

  auto newViewLayoutOp = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), newTensorType, castOp.getResult(), gridRemapping);

  // Reblock it back to original shape to preserve IR correctness.
  auto viewOutputType = mlir::cast<RankedTensorType>(utils::reblockShapedType(
      newTensorType, outputLayout.getGridShape(outputType)));
  auto reblockMap = ttmlir::utils::calculateReblockMap(
      newTensorType.getShape(), viewOutputType.getShape(),
      builder.getContext());
  auto revertingView = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), viewOutputType, newViewLayoutOp.getResult(), reblockMap,
      /*reinterpretLayout=*/false);

  castOp.getResult().replaceAllUsesExcept(revertingView.getResult(),
                                          newViewLayoutOp);
}

// ----------------------------------------------------------------------------
// Transform phases — apply pre-computed grid decisions to the IR.
// ----------------------------------------------------------------------------

static void applyToLayoutUpdate(const OperandGridInfo &info,
                                ArrayRef<int64_t> effectiveTargetGrid,
                                bool ttnnMode, OpBuilder &builder) {
  auto toLayoutOp = info.operand.getDefiningOp<d2m::ToLayoutOp>();
  optimizeToLayoutGrid(toLayoutOp, info.targetGrid, effectiveTargetGrid,
                       ttnnMode, info.selectedGrid, builder);
}

static void applyBehindViewToLayoutUpdate(const OperandGridInfo &info,
                                          ArrayRef<int64_t> effectiveTargetGrid,
                                          bool ttnnMode, OpBuilder &builder) {
  auto view = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
  auto toLayoutOp = view.getInput().getDefiningOp<d2m::ToLayoutOp>();
  optimizeToLayoutGrid(toLayoutOp, info.targetGrid, effectiveTargetGrid,
                       ttnnMode, info.behindViewToLayoutGrid, builder);
}

static void applyTTNNTensorUpdate(const OperandGridInfo &info,
                                  OpBuilder &builder) {
  Value ttnnOperand = info.operand;
  auto metalTensor = mlir::cast<mlir::RankedTensorType>(ttnnOperand.getType());
  auto metalLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());
  if (metalLayout.getMemorySpace() == ttcore::MemorySpace::DeviceDRAM) {
    insertViewForTTNNDRAMTensor(ttnnOperand, info.selectedGrid, builder);
  } else if (auto castOp =
                 ttnnOperand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
    optimizeTTNNMetalLayoutCastOpGrid(castOp, info.selectedGrid, builder);
  } else if (auto viewOp = ttnnOperand.getDefiningOp<d2m::ViewLayoutOp>()) {
    auto originalOperand = viewOp.getInput();
    auto castOp = originalOperand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
    TT_assertv(castOp,
               "Expected a TTNNMetalLayoutCastOp as the input of the view op.");
    viewOp.getResult().replaceAllUsesWith(originalOperand);
    viewOp.erase();
    optimizeTTNNMetalLayoutCastOpGrid(castOp, info.selectedGrid, builder);
  } else {
    llvm_unreachable("Expected a TTNNMetalLayoutCastOp or a ViewLayoutOp");
  }
}

static void applyCompositeViewUpdate(const OperandGridInfo &info,
                                     ArrayRef<int64_t> effectiveTargetGrid,
                                     bool ttnnMode, OpBuilder &builder) {
  auto compositeView = info.operand.getDefiningOp<d2m::CompositeViewOp>();
  int32_t concatDim = compositeView.getDim();

  // Compute the output type first — its grid-aware dim alignments determine
  // what physical shapes the inputs must match on non-concat dimensions.
  auto outType =
      mlir::cast<RankedTensorType>(compositeView.getResult().getType());
  RankedTensorType newOutType = utils::tensorWithOptimalGrid(
      outType, info.targetGrid, ttnnMode, info.selectedGrid, builder);
  auto outLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(newOutType.getEncoding());

  // Build per-input dim alignments that are coordinated with the output:
  //  - Non-concat dims use the output's grid-aware alignment (so physical
  //    sizes match).
  //  - The concat dim uses tile-only alignment (so each input's contribution
  //    isn't independently inflated, keeping sum <= output).
  auto outDimAlignments = outLayout.getDimAlignments();

  SmallVector<Value> reblockedInputs;
  for (Value input : compositeView.getInputs()) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto inputLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
    if (!inputLayout) {
      reblockedInputs.push_back(input);
      continue;
    }

    auto tileType = mlir::cast<ttcore::TileType>(inputType.getElementType());
    auto tileShape = tileType.getShape();

    // Derive input dim alignments from the output, overriding the concat
    // dim with tile-only alignment so each input's contribution isn't
    // independently inflated. concatDim is already normalized to
    // [0, logicalRank) by TTIRToD2M; the trailing two logical dims map to
    // the tile's height/width (tileIdx 0/1), earlier dims are batch dims
    // where tile alignment isn't required (fall back to 1).
    llvm::SmallVector<int64_t> inputAlignments(outDimAlignments.begin(),
                                               outDimAlignments.end());
    int64_t logicalRank = inputLayout.getLogicalShape().size();
    int64_t tileIdx = concatDim - (logicalRank - 2);
    inputAlignments[concatDim] = (tileIdx >= 0) ? tileShape[tileIdx] : 1;

    auto coordLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), inputLayout.getLogicalShape(),
        inputLayout.getOobVal(), inputLayout.getMemorySpace(),
        inputLayout.getMemoryLayout(), inputLayout.getCollapsedIntervals(),
        inputAlignments);

    auto inputPhysShape = coordLayout.getPhysicalShape(tileShape);
    auto inputOptimalGrid =
        utils::computeOptimalGrid(inputType, inputPhysShape, info.targetGrid);

    llvm::SmallVector<int64_t> deviceShape =
        coordLayout.getDeviceShape(inputOptimalGrid, tileShape);
    auto newInputType = RankedTensorType::get(
        deviceShape, inputType.getElementType(), coordLayout);

    // When the input comes directly from a to_layout op with a single use,
    // update the to_layout's grid so that data is physically distributed
    // across multiple cores, preventing L1 overflow when multiple concat
    // inputs must coexist in L1 on a single core.
    if (auto toLayoutOp = input.getDefiningOp<d2m::ToLayoutOp>();
        toLayoutOp && input.hasOneUse()) {
      auto emptyOp = toLayoutOp.getOutput().getDefiningOp<d2m::EmptyOp>();
      if (emptyOp) {
        builder.setInsertionPoint(emptyOp);
        auto newEmptyOp = builder.create<d2m::EmptyOp>(
            emptyOp.getLoc(), newInputType.getShape(),
            newInputType.getElementType(), newInputType.getEncoding(),
            effectiveTargetGrid);
        builder.setInsertionPoint(toLayoutOp);
        auto newToLayoutOp = builder.create<d2m::ToLayoutOp>(
            toLayoutOp.getLoc(), toLayoutOp.getInput(), newEmptyOp);
        toLayoutOp.getResult(0).replaceAllUsesWith(newToLayoutOp.getResult(0));
        toLayoutOp.erase();
        if (emptyOp.getResult().use_empty()) {
          emptyOp.erase();
        }
        reblockedInputs.push_back(newToLayoutOp.getResult(0));
        continue;
      }
    }

    builder.setInsertionPoint(compositeView);
    auto view = builder.create<d2m::ViewLayoutOp>(compositeView.getLoc(),
                                                  newInputType, input);
    reblockedInputs.push_back(view.getResult());
  }

  builder.setInsertionPoint(compositeView);
  auto newCompositeView = builder.create<d2m::CompositeViewOp>(
      compositeView.getLoc(), newOutType, reblockedInputs,
      compositeView.getDim());

  compositeView.getResult().replaceAllUsesWith(newCompositeView.getResult());
  compositeView.erase();
}

static void applyEmptyOpUpdate(const OperandGridInfo &info,
                               ArrayRef<int64_t> effectiveTargetGrid,
                               bool ttnnMode, OpBuilder &builder) {
  EmptyOp emptyOp = info.operand.getDefiningOp<d2m::EmptyOp>();
  auto emptyType =
      mlir::cast<mlir::RankedTensorType>(emptyOp.getResult().getType());
  RankedTensorType newTensorType = utils::tensorWithOptimalGrid(
      emptyType, info.targetGrid, ttnnMode, info.selectedGrid, builder);
  builder.setInsertionPoint(emptyOp);

  mlir::AffineMapAttr virtualGridInverseMapping =
      emptyOp.getVirtualGridInverseMappingAttr();
  mlir::AffineMapAttr virtualGridForwardMapping =
      emptyOp.getVirtualGridForwardMappingAttr();
  if (!virtualGridInverseMapping) {
    auto device = ttcore::lookupDevice(emptyOp);
    auto workerGridShape = device.getWorkerGrid().getShape();
    bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
        info.selectedGrid, workerGridShape);
    if (isVirtual) {
      auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
          ttmlir::utils::volume<int64_t>(info.selectedGrid),
          effectiveTargetGrid);
      TT_assertv(!physicalGridShape.empty(),
                 "Unable to find 2D rect that can fit virtual grid");
      auto [forwardMap, inverseMap] =
          ttmlir::d2m::utils::grids::createCoreVirtMaps(
              builder.getContext(), info.selectedGrid, physicalGridShape);
      virtualGridInverseMapping = AffineMapAttr::get(inverseMap);
      virtualGridForwardMapping = AffineMapAttr::get(forwardMap);
    }
  }

  auto newEmptyOp = builder.create<d2m::EmptyOp>(
      emptyOp.getLoc(), newTensorType, virtualGridInverseMapping,
      virtualGridForwardMapping);
  emptyOp.getResult().replaceAllUsesWith(newEmptyOp.getResult());
  emptyOp.erase();
}

// Derive grid (including virtual grid mapping) from the
// optimized operand grids selected by GridSelection, mirroring
// GenericOp::build.
static ttcore::GridAttr deriveGridAttrForOutput(Value output,
                                                ArrayRef<int64_t> gridShape,
                                                OpBuilder &builder) {
  auto layout = ttcore::getDeviceLayout(cast<ShapedType>(output.getType()));
  auto metalLayout = mlir::dyn_cast<ttcore::MetalLayoutAttr>(layout);
  if (!metalLayout) {
    return builder.getAttr<ttcore::GridAttr>(gridShape);
  }

  if (auto invMap = utils::getVirtualGridInverseMapping(output)) {
    // Get virtual to physical map from output as well.
    auto fwdMap = *utils::getVirtualGridForwardMapping(output);
    size_t rank = gridShape.size();
    fwdMap = ttmlir::utils::affineMapDropBackResults(fwdMap, rank);
    for (int i = rank - 1; i >= 0; i--) {
      fwdMap = ttmlir::utils::dropDim(fwdMap, rank + i);
    }
    fwdMap =
        fwdMap.insertResult(getAffineConstantExpr(0, builder.getContext()), 0);

    return builder.getAttr<ttcore::GridAttr>(gridShape, fwdMap, *invMap);
  }

  auto existingRemapping = utils::getAssociatedRemapping(output);
  if (!existingRemapping.has_value() || existingRemapping->isEmpty() ||
      existingRemapping->isIdentity()) {
    return builder.getAttr<ttcore::GridAttr>(gridShape);
  }

  auto indexMap = *existingRemapping;
  constexpr size_t kExpectedDimsFor2DDeviceShape = 2 * 2;
  bool is2DPermutation =
      indexMap.isPermutation() &&
      indexMap.getNumResults() == kExpectedDimsFor2DDeviceShape &&
      indexMap.getNumInputs() == kExpectedDimsFor2DDeviceShape;
  if (!is2DPermutation) {
    return builder.getAttr<ttcore::GridAttr>(gridShape);
  }

  auto [forwardMap, inverseMap] =
      ttmlir::utils::createGridForwardAndInverseMapFor2DPermutation(
          indexMap, gridShape.size(), builder.getContext());
  return builder.getAttr<ttcore::GridAttr>(gridShape, forwardMap, inverseMap);
}

static ttcore::GridAttr
deriveGenericGridAttr(d2m::GenericOp genericOp,
                      ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
                      OpBuilder &builder) {
  Value output = genericOp.getOutputs().front();
  unsigned outputOperandIndex = genericOp.getOutputs().getBeginOperandIndex();
  ArrayRef<int64_t> gridShape = optimalOperandGrids[outputOperandIndex];
  return deriveGridAttrForOutput(output, gridShape, builder);
}

// Update a ViewLayoutOp by recreating it with a new output type that matches
// the normalized grid. The remapping is composed with a reblock map that
// accounts for the shape change from the old grid to the new grid.
static void applyViewLayoutUpdate(const OperandGridInfo &info, bool ttnnMode,
                                  OpBuilder &builder) {
  d2m::ViewLayoutOp viewOp = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
  auto oldResultType =
      mlir::cast<RankedTensorType>(viewOp.getResult().getType());
  auto oldLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(oldResultType.getEncoding());
  RankedTensorType newResultType = utils::tensorWithOptimalGrid(
      oldResultType, info.targetGrid, ttnnMode, info.selectedGrid, builder);

  // Compose the original remapping with a reblock map that maps from the
  // old output shape to the new output shape.
  llvm::SmallVector<int64_t> oldShape(oldResultType.getShape());
  llvm::SmallVector<int64_t> newShape(newResultType.getShape());

  // If dim alignments changed, align-up the old shape to match.
  auto newLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(newResultType.getEncoding());
  if (!llvm::equal(oldLayout.getDimAlignments(),
                   newLayout.getDimAlignments())) {
    llvm::SmallVector<int64_t> tileShape;
    if (auto tileType =
            mlir::dyn_cast<ttcore::TileType>(oldResultType.getElementType())) {
      tileShape = llvm::to_vector(tileType.getShape());
    }
    TT_assert(ttmlir::utils::volume(oldLayout.getGridShape(oldResultType)) ==
              1);
    oldShape = newLayout.getDeviceShape(oldLayout.getGridShape(oldResultType),
                                        tileShape);
  }

  mlir::AffineMap newRemapping = viewOp.getRemapping();
  if (!llvm::equal(oldShape, newShape)) {
    TT_assert(ttmlir::utils::volume<int64_t>(oldShape) ==
              ttmlir::utils::volume<int64_t>(newShape));
    mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
        oldShape, newShape, builder.getContext());
    newRemapping = newRemapping.compose(reblockMap);
  }

  builder.setInsertionPoint(viewOp);
  auto newViewOp = builder.create<d2m::ViewLayoutOp>(
      viewOp.getLoc(), newResultType, viewOp.getInput(), newRemapping,
      viewOp.getReinterpretLayout());
  viewOp.getResult().replaceAllUsesWith(newViewOp.getResult());
  viewOp.erase();
}

// Recreate the d2m.generic with updated operands.
// After updating ToLayout and ViewLayout ops, the generic's operands have
// new types with the selected grids. The generic grid is still anchored by the
// output operand's chosen grid, but we must re-materialize the generic attrs
// from the selected operand grids after those rewrites so the rebuilt op stays
// consistent with the new operand types and the derived block factors.
static void
recreateGenericOp(d2m::GenericOp genericOp,
                  ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return;
  }

  OpBuilder builder(genericOp);
  unsigned outputOperandIndex = genericOp.getOutputs().getBeginOperandIndex();
  ArrayRef<int64_t> outputGridShape = optimalOperandGrids[outputOperandIndex];
  ttcore::GridAttr grid =
      deriveGenericGridAttr(genericOp, optimalOperandGrids, builder);
  SmallVector<int64_t> blockFactors = utils::deriveBlockFactorsFromOperandGrids(
      genericOp.getIndexingMapsValue(), optimalOperandGrids, outputGridShape);
  auto ret = genericOp.withParallelization(builder, grid, blockFactors,
                                           /*generateReturnView=*/false);
  if (failed(ret)) {
    genericOp.emitOpError()
        << "failed to recreate generic op with withParallelization";
    return;
  }

  genericOp->replaceAllUsesWith(ret->genericOp);
  genericOp.erase();
}

// ----------------------------------------------------------------------------
// Apply all grid decisions from a GenericGridAnalysisResult to a GenericOp.
// ----------------------------------------------------------------------------

static void applyGridDecisions(d2m::GenericOp genericOp,
                               const GenericGridAnalysisResult &result,
                               bool ttnnMode) {
  // effectiveTargetGrid is the generic's target grid (full device grid, or the
  // range scoped by an enclosing d2m.spatial region), used for virtual grid
  // physical mapping. Per-operand target grids (info.targetGrid) are used
  // for alignment computation.
  ArrayRef<int64_t> effectiveTargetGrid = result.effectiveTargetGrid;
  OpBuilder builder(genericOp->getContext());

  // Classify operands upfront: once apply* mutates IR, defining ops for
  // operand values may be erased, so we can't re-query the kind mid-pass.
  enum class Kind {
    TTNNTensor,
    ViewLayout,
    CompositeView,
    ToLayout,
    Empty,
    Skip
  };
  llvm::SmallVector<Kind, 4> kinds;
  kinds.reserve(result.operandInfos.size());
  for (const auto &info : result.operandInfos) {
    Value operand = info.operand;
    if (GridAnalysis::isTTNNOperand(operand)) {
      kinds.push_back(Kind::TTNNTensor);
    } else if (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
      // Reinterpret views are type casts only; their grid must match the
      // input, so don't rewrite them.
      kinds.push_back(view.getReinterpretLayout() ? Kind::Skip
                                                  : Kind::ViewLayout);
    } else if (operand.getDefiningOp<d2m::CompositeViewOp>()) {
      kinds.push_back(Kind::CompositeView);
    } else if (operand.getDefiningOp<d2m::ToLayoutOp>()) {
      kinds.push_back(Kind::ToLayout);
    } else if (operand.getDefiningOp<d2m::EmptyOp>()) {
      kinds.push_back(Kind::Empty);
    } else {
      kinds.push_back(Kind::Skip);
    }
  }

  // Pass 1: producer-side rewrites. View layouts are handled in pass 2 so
  // they see the final state after upstream producers are rewritten.
  for (auto [info, kind] : llvm::zip_equal(result.operandInfos, kinds)) {
    switch (kind) {
    case Kind::TTNNTensor:
      applyTTNNTensorUpdate(info, builder);
      break;
    case Kind::CompositeView:
      applyCompositeViewUpdate(info, effectiveTargetGrid, ttnnMode, builder);
      break;
    case Kind::ToLayout:
      applyToLayoutUpdate(info, effectiveTargetGrid, ttnnMode, builder);
      break;
    case Kind::Empty:
      applyEmptyOpUpdate(info, effectiveTargetGrid, ttnnMode, builder);
      break;
    case Kind::ViewLayout:
    case Kind::Skip:
      break;
    }
    if (!info.behindViewToLayoutGrid.empty()) {
      applyBehindViewToLayoutUpdate(info, effectiveTargetGrid, ttnnMode,
                                    builder);
    }
  }

  // Pass 2: view layout rewrites.
  for (auto [info, kind] : llvm::zip_equal(result.operandInfos, kinds)) {
    if (kind == Kind::ViewLayout) {
      applyViewLayoutUpdate(info, ttnnMode, builder);
    }
  }

  recreateGenericOp(genericOp, result.normalizedOperandGrids);
}

// Resolve to a value that dominates the spatial op by following view_layout
// chains defined inside the spatial's regions (region-border value).
static Value resolveToRegionBorderValue(Value operand,
                                        d2m::SpatialOp spatialOp) {
  auto inSpatialRegion = [&](Value val) {
    Operation *def = val.getDefiningOp();
    if (!def) {
      return false;
    }
    Region *parent = def->getBlock()->getParent();
    return llvm::any_of(spatialOp->getRegions(),
                        [parent](Region &r) { return &r == parent; });
  };
  Value current = operand;
  while (inSpatialRegion(current)) {
    if (auto viewOp = current.getDefiningOp<d2m::ViewLayoutOp>()) {
      current = viewOp.getInput();
    } else {
      break;
    }
  }
  return current;
}

// Rebuild d2m.spatial's ins and outs from the operands actually used by
// d2m.generic ops in each region, and set result types from the collected outs.
static void reconstructSpatialOperands(d2m::SpatialOp spatialOp) {
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
  for (Region &region : spatialOp->getRegions()) {
    if (region.empty()) {
      continue;
    }
    for (d2m::GenericOp genericOp : region.front().getOps<d2m::GenericOp>()) {
      for (mlir::Value input : genericOp.getInputs()) {
        inputs.push_back(resolveToRegionBorderValue(input, spatialOp));
      }
      for (mlir::Value output : genericOp.getOutputs()) {
        outputs.push_back(resolveToRegionBorderValue(output, spatialOp));
      }
    }
  }
  spatialOp.getInputsMutable().assign(inputs);
  spatialOp.getOutputsMutable().assign(outputs);
  if (spatialOp->getNumResults() == outputs.size()) {
    for (auto [result, outVal] : llvm::zip(spatialOp->getResults(), outputs)) {
      result.setType(outVal.getType());
    }
  }
}

// ----------------------------------------------------------------------------
// Pass implementation
// ----------------------------------------------------------------------------

#define GEN_PASS_DEF_D2MGRIDSELECTION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGridSelectionPass final
    : public impl::D2MGridSelectionBase<D2MGridSelectionPass> {
public:
  using Base = impl::D2MGridSelectionBase<D2MGridSelectionPass>;

  D2MGridSelectionPass() = default;

  D2MGridSelectionPass(const D2MGridSelectionOptions &options) : Base() {
    this->overrideDeviceShape = llvm::to_vector(options.overrideDeviceShape);
    // Setting TTNN mode to true ensures we do not implicitly pad or wrap-around
    // when sharding. Any grid decisions in this mode are representable
    // using a TTNNLayoutAttr and can be created with a single ttnn.empty()
    // call. This can be removed only when we implement support for creating
    // padded tensors in D2MToTTNN pass.
    this->ttnnMode = options.ttnnMode;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Phase 1: Analyze all generics (no IR mutation).
    GridAnalysis gridAnalysis(module, getDeviceGridShape(), this->ttnnMode);

    // Phase 2: Apply transforms using pre-computed analysis.
    // Collect generics first because the walk is invalidated by IR mutation.
    SmallVector<d2m::GenericOp> generics;
    module.walk(
        [&](d2m::GenericOp genericOp) { generics.push_back(genericOp); });

    for (auto genericOp : generics) {
      const auto *result = gridAnalysis.lookup(genericOp);
      if (!result) {
        continue;
      }
      applyGridDecisions(genericOp, *result, this->ttnnMode);
    }

    // Phase 3: Rebuild each SpatialOp's ins/outs from the operands actually
    // used by generics in its regions.
    module.walk([&](d2m::SpatialOp spatialOp) {
      reconstructSpatialOperands(spatialOp);
    });
  }

private:
  // Returns the device-wide grid shape (worker grid or override).
  llvm::SmallVector<int64_t> getDeviceGridShape() {
    if (!overrideDeviceShape.empty()) {
      return llvm::to_vector(overrideDeviceShape);
    }

    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }
};
} // namespace

} // namespace mlir::tt::d2m
