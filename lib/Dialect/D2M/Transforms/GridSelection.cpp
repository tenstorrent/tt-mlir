// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/GridAnalysis.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialOpNormalizeUtil.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::d2m {

// ----------------------------------------------------------------------------
// Transform helpers — these modify the IR to apply grid decisions.
// ----------------------------------------------------------------------------

static std::pair<mlir::AffineMapAttr, mlir::AffineMapAttr>
deriveVirtualGridAttrs(ArrayRef<int64_t> selectedGrid,
                       const EffectiveTargetGridRange &effectiveTargetGridRange,
                       OpBuilder &builder) {
  ArrayRef<int64_t> effectiveTargetGridShape = effectiveTargetGridRange.shape;
  ArrayRef<int64_t> effectiveTargetGridOffset = effectiveTargetGridRange.offset;

  bool hasOffset = llvm::any_of(effectiveTargetGridOffset,
                                [](int64_t coord) { return coord != 0; });
  bool requiresVirtualization = ttmlir::d2m::utils::grids::requiresVirtualGrid(
      selectedGrid, effectiveTargetGridShape);
  if (!requiresVirtualization && !hasOffset) {
    return {mlir::AffineMapAttr(), mlir::AffineMapAttr()};
  }
  AffineMap forwardMap;
  AffineMap inverseMap;
  if (requiresVirtualization) {
    SmallVector<int64_t> physicalGridShape =
        utils::findLegalPhysicalGridForVolume(
            ttmlir::utils::volume<int64_t>(selectedGrid),
            effectiveTargetGridShape);
    TT_assertv(!physicalGridShape.empty(),
               "Unable to find 2D rect that can fit virtual grid {} within "
               "device grid {}",
               ttmlir::utils::formatIterable(selectedGrid, "x"),
               ttmlir::utils::formatIterable(effectiveTargetGridShape, "x"));
    std::tie(forwardMap, inverseMap) =
        ttmlir::d2m::utils::grids::createCoreVirtMaps(
            builder.getContext(), selectedGrid, physicalGridShape);
  } else {
    // Offset-only remap without virtualization: start from identity mappings.
    TT_assertv(selectedGrid.size() == 2u,
               "Expected 2D selected grid for offset-only remap");
    unsigned rank = selectedGrid.size() * 2;
    forwardMap = AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
    inverseMap =
        ttmlir::utils::createIdentityGridInverseMap(builder.getContext());
  }

  if (hasOffset) {
    TT_assertv(effectiveTargetGridOffset.size() == 2u,
               "Expected 2D effective target grid offset");
    forwardMap = ttmlir::utils::applyOffsetsToAffineMapResults(
        forwardMap, effectiveTargetGridOffset, /*startIndex=*/0);
    SmallVector<int64_t> inverseOffsets = {-effectiveTargetGridOffset[0],
                                           -effectiveTargetGridOffset[1]};
    inverseMap = ttmlir::utils::applyOffsetsToAffineMapDims(
        inverseMap, inverseOffsets, /*startIndex=*/0);
  }
  return {AffineMapAttr::get(inverseMap), AffineMapAttr::get(forwardMap)};
}

static void recordGenericConsumer(Operation *user,
                                  d2m::GenericOp &parentGeneric) {
  d2m::GenericOp useGeneric = dyn_cast<d2m::GenericOp>(user);
  if (!useGeneric) {
    useGeneric = user->getParentOfType<d2m::GenericOp>();
  }

  TT_assertv(useGeneric,
             "ToLayout result must be used by a single GenericOp, a single "
             "ViewLayout, or a single MaskOp feeding a single GenericOp");

  if (!parentGeneric) {
    parentGeneric = useGeneric;
    return;
  }

  TT_assertv(parentGeneric == useGeneric,
             "ToLayout should only be used within one GenericOp");
}

static void verifySingleGenericConsumerThroughViewsAndMasks(Value root) {
  d2m::GenericOp parentGeneric = nullptr;
  SmallVector<Value> worklist{root};

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto viewLayoutOp = dyn_cast<d2m::ViewLayoutOp>(user)) {
        worklist.push_back(viewLayoutOp.getResult());
        continue;
      }
      if (auto maskOp = dyn_cast<d2m::MaskOp>(user)) {
        worklist.push_back(maskOp.getResult());
        continue;
      }
      if (isa<d2m::SpatialOp>(user)) {
        continue;
      }
      recordGenericConsumer(user, parentGeneric);
    }
  }
}

static llvm::SmallVector<int64_t>
getScalarBridgePaddingTileShape(d2m::ToLayoutOp toLayoutOp,
                                RankedTensorType outputType) {
  if (mlir::isa<ttcore::TileType>(outputType.getElementType())) {
    return {};
  }

  // A scalar result can still be a layout bridge for a tiled stream, possibly
  // through earlier to_layout/view_layout ops. Keep that bridge tile-compatible
  // so later layout conversion sees the same final grid/padding contract.
  return utils::findUpstreamTiledLayoutBridgeTileShape(toLayoutOp.getInput());
}

// Update a ToLayoutOp and its associated EmptyOp to use a specified grid by
// recreating the MetalLayoutAttr with the given grid and proper dimension
// alignments.
static void
optimizeToLayoutGrid(d2m::ToLayoutOp toLayoutOp, ArrayRef<int64_t> targetGrid,
                     const EffectiveTargetGridRange &effectiveTargetGridRange,
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
    // A selected 1x1 grid does not require producer-side redistribution.
    // In non-origin spatial regions, offset-aware mapping is materialized on
    // the output path: applyEmptyOpUpdate updates the outs EmptyOp VGM, and
    // deriveGridAttrForOutput rebuilds the generic grid mapping from it.
    return;
  }

  llvm::SmallVector<int64_t> paddingTileShape =
      getScalarBridgePaddingTileShape(toLayoutOp, outputType);
  RankedTensorType newTensorType = utils::tensorWithOptimalGrid(
      outputType, ttnnMode, optimalGrid, paddingTileShape);
  builder.setInsertionPoint(emptyOp);

  // VGM is NOT propagated from the to_layout's input here — the output EmptyOp
  // has its own grid/shard strategy. VGM for DMA addresses is traced through
  // the stream's input at DMA lowering time.
  auto [virtualGridInverseMapping, virtualGridForwardMapping] =
      deriveVirtualGridAttrs(optimalGrid, effectiveTargetGridRange, builder);

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
  // A d2m.mask may also sit between the ToLayout and the GenericOp; in that
  // case the mask is the materialized padding contract and should not hide the
  // stream that still needs grid optimization.
  verifySingleGenericConsumerThroughViewsAndMasks(toLayoutOp.getResult(0));
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
      ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Interleaved,
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

static void
applyToLayoutUpdate(const OperandGridInfo &info,
                    const EffectiveTargetGridRange &effectiveTargetGridRange,
                    bool ttnnMode, OpBuilder &builder) {
  auto toLayoutOp = info.operand.getDefiningOp<d2m::ToLayoutOp>();
  optimizeToLayoutGrid(toLayoutOp, info.targetGrid, effectiveTargetGridRange,
                       ttnnMode, info.selectedGrid, builder);
}

static void applyBehindViewToLayoutUpdate(
    const OperandGridInfo &info,
    const EffectiveTargetGridRange &effectiveTargetGridRange, bool ttnnMode,
    OpBuilder &builder) {
  auto toLayoutOp = utils::getToLayoutProducerBehindViews(info.operand);
  TT_assert(toLayoutOp);
  optimizeToLayoutGrid(toLayoutOp, info.targetGrid, effectiveTargetGridRange,
                       ttnnMode, info.viewSourceGrid, builder);
}

static void
applyMaskUpdate(const OperandGridInfo &info,
                const EffectiveTargetGridRange &effectiveTargetGridRange,
                bool ttnnMode, OpBuilder &builder) {
  auto maskOp = info.operand.getDefiningOp<d2m::MaskOp>();
  if (!maskOp) {
    return;
  }

  if (auto toLayoutOp = maskOp.getInput().getDefiningOp<d2m::ToLayoutOp>()) {
    optimizeToLayoutGrid(toLayoutOp, info.targetGrid, effectiveTargetGridRange,
                         ttnnMode, info.selectedGrid, builder);
  } else if (auto view = maskOp.getInput().getDefiningOp<d2m::ViewLayoutOp>()) {
    if (auto toLayoutOp = view.getInput().getDefiningOp<d2m::ToLayoutOp>()) {
      optimizeToLayoutGrid(toLayoutOp, info.targetGrid,
                           effectiveTargetGridRange, ttnnMode,
                           info.selectedGrid, builder);
    }
  }

  auto oldResultType =
      mlir::cast<RankedTensorType>(maskOp.getResult().getType());
  RankedTensorType newResultType = utils::tensorWithOptimalGrid(
      oldResultType, ttnnMode, info.selectedGrid, info.paddingTileShape);
  if (newResultType == oldResultType) {
    return;
  }

  auto oldOutput = maskOp.getOutput().getDefiningOp<d2m::EmptyOp>();
  builder.setInsertionPoint(maskOp);

  Value newInput = maskOp.getInput();
  if (newInput.getType() != newResultType) {
    newInput =
        builder
            .create<d2m::ViewLayoutOp>(maskOp.getLoc(), newResultType, newInput)
            .getResult();
  }

  auto [virtualGridInverseMapping, virtualGridForwardMapping] =
      deriveVirtualGridAttrs(info.selectedGrid, effectiveTargetGridRange,
                             builder);
  auto newOutput = builder.create<d2m::EmptyOp>(maskOp.getLoc(), newResultType,
                                                virtualGridInverseMapping,
                                                virtualGridForwardMapping);
  auto newMask = builder.create<d2m::MaskOp>(
      maskOp.getLoc(), newInput, newOutput, maskOp.getLogicalShape(),
      maskOp.getFillValue());

  maskOp.getResult().replaceAllUsesWith(newMask.getResult());
  maskOp.erase();
  if (oldOutput && oldOutput.getResult().use_empty()) {
    oldOutput.erase();
  }
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

static Value rewriteSingleUseCompositeInputProducer(
    Value input, RankedTensorType newInputType,
    ArrayRef<int64_t> inputSelectedGrid,
    const EffectiveTargetGridRange &effectiveTargetGridRange,
    OpBuilder &builder) {
  auto toLayoutOp = input.getDefiningOp<d2m::ToLayoutOp>();
  // Only rewrite single-use producers. Shared to_layout results may feed users
  // that still require the original layout, so multi-use inputs need a view.
  if (!toLayoutOp || !input.hasOneUse()) {
    return {};
  }

  auto emptyOp = toLayoutOp.getOutput().getDefiningOp<d2m::EmptyOp>();
  if (!emptyOp) {
    return {};
  }

  auto [virtualGridInverseMapping, virtualGridForwardMapping] =
      deriveVirtualGridAttrs(inputSelectedGrid, effectiveTargetGridRange,
                             builder);
  builder.setInsertionPoint(emptyOp);
  auto newEmptyOp = builder.create<d2m::EmptyOp>(emptyOp.getLoc(), newInputType,
                                                 virtualGridInverseMapping,
                                                 virtualGridForwardMapping);

  builder.setInsertionPoint(toLayoutOp);
  auto newToLayoutOp = builder.create<d2m::ToLayoutOp>(
      toLayoutOp.getLoc(), toLayoutOp.getInput(), newEmptyOp);
  toLayoutOp.getResult(0).replaceAllUsesWith(newToLayoutOp.getResult(0));
  toLayoutOp.erase();
  if (emptyOp.getResult().use_empty()) {
    emptyOp.erase();
  }
  return newToLayoutOp.getResult(0);
}

static Value materializeCompositeInput(
    d2m::CompositeViewOp compositeView, Value input,
    const CompositeInputGridInfo &inputInfo,
    const EffectiveTargetGridRange &effectiveTargetGridRange,
    OpBuilder &builder) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto inputLayout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
  if (!inputLayout) {
    return input;
  }

  if (Value rewrittenProducer = rewriteSingleUseCompositeInputProducer(
          input, inputInfo.materializedType, inputInfo.selectedGrid,
          effectiveTargetGridRange, builder)) {
    return rewrittenProducer;
  }

  builder.setInsertionPoint(compositeView);
  return builder
      .create<d2m::ViewLayoutOp>(compositeView.getLoc(),
                                 inputInfo.materializedType, input)
      .getResult();
}

static void applyCompositeViewUpdate(
    const OperandGridInfo &info,
    const EffectiveTargetGridRange &effectiveTargetGridRange, bool ttnnMode,
    OpBuilder &builder) {
  auto compositeView = info.operand.getDefiningOp<d2m::CompositeViewOp>();
  auto outType =
      mlir::cast<RankedTensorType>(compositeView.getResult().getType());

  RankedTensorType newOutType = utils::tensorWithOptimalGrid(
      outType, ttnnMode, info.selectedGrid, info.paddingTileShape);

  TT_assertv(info.compositeInputInfos.size() ==
                 compositeView.getInputs().size(),
             "CompositeViewOp input grid analysis is stale");

  SmallVector<Value> reblockedInputs;
  for (auto &&[input, inputInfo] :
       llvm::zip_equal(compositeView.getInputs(), info.compositeInputInfos)) {
    TT_assertv(
        input == inputInfo.input,
        "CompositeViewOp input grid analysis does not match input order");
    reblockedInputs.push_back(materializeCompositeInput(
        compositeView, input, inputInfo, effectiveTargetGridRange, builder));
  }

  builder.setInsertionPoint(compositeView);
  auto newCompositeView = builder.create<d2m::CompositeViewOp>(
      compositeView.getLoc(), newOutType, reblockedInputs,
      compositeView.getDim(), compositeView.getLogicalSizesAttr());

  compositeView.getResult().replaceAllUsesWith(newCompositeView.getResult());
  compositeView.erase();
}

static void
applyEmptyOpUpdate(const OperandGridInfo &info,
                   const EffectiveTargetGridRange &effectiveTargetGridRange,
                   bool ttnnMode, OpBuilder &builder) {
  EmptyOp emptyOp = info.operand.getDefiningOp<d2m::EmptyOp>();
  auto emptyType =
      mlir::cast<mlir::RankedTensorType>(emptyOp.getResult().getType());
  RankedTensorType newTensorType = utils::tensorWithOptimalGrid(
      emptyType, ttnnMode, info.selectedGrid, info.paddingTileShape);
  builder.setInsertionPoint(emptyOp);

  // The selected grid may differ from the EmptyOp's previous grid.
  // TODO (#8301): Avoid creating placeholder EmptyOp VGMs before grid
  // selection so this rewrite does not need to repair stale mappings.
  auto [virtualGridInverseMapping, virtualGridForwardMapping] =
      deriveVirtualGridAttrs(info.selectedGrid, effectiveTargetGridRange,
                             builder);

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

  if (auto maps = utils::getGridMapsFromVirtualGridMapping(output, gridShape)) {
    return builder.getAttr<ttcore::GridAttr>(gridShape, maps->first,
                                             maps->second);
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
      oldResultType, ttnnMode, info.selectedGrid, info.paddingTileShape);

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
// new types with the selected grids. The generic grid is still anchored by
// the output operand's chosen grid, but we must re-materialize the generic
// attrs from the selected operand grids after those rewrites so the rebuilt
// op stays consistent with the new operand types and the derived block
// factors.
//
// Returns the generic to use for further work. Success is either:
//  - no-op (empty grids): returns original op
//  - recreated op: returns new op and erases original op
// Failure is returned when generic recreation fails.
static FailureOr<d2m::GenericOp>
recreateGenericOp(d2m::GenericOp genericOp,
                  ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return genericOp;
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
    return failure();
  }

  d2m::GenericOp newGeneric = ret->genericOp;
  genericOp->replaceAllUsesWith(newGeneric);
  genericOp.erase();
  return newGeneric;
}

// ----------------------------------------------------------------------------
// Apply all grid decisions from a GenericGridAnalysisResult to a GenericOp.
// ----------------------------------------------------------------------------

static LogicalResult applyGridDecisions(d2m::GenericOp genericOp,
                                        const GenericGridAnalysisResult &result,
                                        bool ttnnMode) {
  // effectiveTargetGridRange is the range of the generic's target grid (full
  // device grid, or the range scoped by an enclosing d2m.spatial region), used
  // for virtual grid physical mapping. Per-operand target grids
  // (info.targetGrid) are used for alignment computation.
  const EffectiveTargetGridRange &effectiveTargetGridRange =
      result.effectiveTargetGridRange;
  TT_assertv(effectiveTargetGridRange.shape.size() == 2u,
             "Expected 2D effective target grid shape");
  TT_assertv(effectiveTargetGridRange.offset.size() == 2u,
             "Expected 2D effective target grid offset");
  OpBuilder builder(genericOp->getContext());

  // Classify operands upfront: once apply* mutates IR, defining ops for
  // operand values may be erased, so we can't re-query the kind mid-pass.
  enum class Kind {
    TTNNTensor,
    ViewLayout,
    CompositeView,
    ToLayout,
    Mask,
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
    } else if (operand.getDefiningOp<d2m::MaskOp>()) {
      kinds.push_back(Kind::Mask);
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
      applyCompositeViewUpdate(info, effectiveTargetGridRange, ttnnMode,
                               builder);
      break;
    case Kind::ToLayout:
      applyToLayoutUpdate(info, effectiveTargetGridRange, ttnnMode, builder);
      break;
    case Kind::Mask:
      applyMaskUpdate(info, effectiveTargetGridRange, ttnnMode, builder);
      break;
    case Kind::Empty:
      applyEmptyOpUpdate(info, effectiveTargetGridRange, ttnnMode, builder);
      break;
    case Kind::ViewLayout:
    case Kind::Skip:
      break;
    }
    if (!info.viewSourceGrid.empty()) {
      applyBehindViewToLayoutUpdate(info, effectiveTargetGridRange, ttnnMode,
                                    builder);
    }
  }

  // Pass 2: view layout rewrites.
  for (auto [info, kind] : llvm::zip_equal(result.operandInfos, kinds)) {
    if (kind == Kind::ViewLayout) {
      applyViewLayoutUpdate(info, ttnnMode, builder);
    }
  }

  FailureOr<d2m::GenericOp> newGenericOp =
      recreateGenericOp(genericOp, result.normalizedOperandGrids);
  if (failed(newGenericOp)) {
    genericOp.emitOpError() << "grid selection failed to recreate generic op";
    return failure();
  }

  normalizeSpatialOpContainingGeneric(*newGenericOp);
  return success();
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
    // Setting TTNN mode to true ensures we do not implicitly pad or
    // wrap-around when sharding. Any grid decisions in this mode are
    // representable using a TTNNLayoutAttr and can be created with a single
    // ttnn.empty() call. This can be removed only when we implement support
    // for creating padded tensors in D2MToTTNN pass.
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
      if (failed(applyGridDecisions(genericOp, *result, this->ttnnMode))) {
        signalPassFailure();
        return;
      }
    }
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
