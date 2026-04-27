// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/LowerToLayout/Plan.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "llvm/ADT/STLExtras.h"

#include <numeric>
#include <optional>

namespace mlir::tt::d2m {

static OutputBufferSpec makeOutputSpec(RankedTensorType type,
                                       AffineMap vgmForward = AffineMap(),
                                       AffineMap vgmInverse = AffineMap()) {
  return {type, vgmForward, vgmInverse};
}

static void updateStateFromOutput(PlanState &state,
                                  const OutputBufferSpec &output,
                                  AffineMap remapping = AffineMap()) {
  state.type = output.type;
  state.remapping = remapping;
  state.vgmForward = output.vgmForward;
  state.vgmInverse = output.vgmInverse;
}

// ============================================================================
// PlanState accessors
// ============================================================================

std::optional<ttcore::MetalLayoutAttr> PlanState::getLayout() const {
  if (!type) {
    return std::nullopt;
  }
  auto layout =
      mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(type.getEncoding());
  return layout ? std::optional(layout) : std::nullopt;
}

bool PlanState::isSystem() const {
  auto layout = getLayout();
  return !layout || layout->getMemorySpace() == ttcore::MemorySpace::System;
}

bool PlanState::isL1() const {
  auto layout = getLayout();
  return layout && layout->getMemorySpace() == ttcore::MemorySpace::DeviceL1;
}

bool PlanState::isDRAM() const {
  auto layout = getLayout();
  return layout && layout->getMemorySpace() == ttcore::MemorySpace::DeviceDRAM;
}

llvm::ArrayRef<int64_t> PlanState::getGridShape() const {
  auto layout = getLayout();
  assert(layout && "Cannot get grid shape without layout");
  return layout->getGridShape(type);
}

// ============================================================================
// Layout / intermediate-type utilities
// ============================================================================

namespace {

bool needsVirtualGridBounce(ttcore::MetalLayoutAttr referenceLayout,
                            RankedTensorType referenceType,
                            ArrayRef<int64_t> targetGridShape) {
  return ttmlir::d2m::utils::grids::requiresVirtualGrid(
      referenceLayout.getGridShape(referenceType), targetGridShape);
}

Type getScalarType(Type type) {
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(type)) {
    return tileType.getElementType();
  }
  return type;
}

// A target layout requires masking iff its OOBVal is non-Undef and the tiled
// shape has padding beyond the logical shape.
bool needsMasking(ttcore::MetalLayoutAttr layout, RankedTensorType tensorType) {
  if (layout.getOobVal() == ttcore::OOBVal::Undef) {
    return false;
  }
  if (!ttcore::isTiled(tensorType)) {
    return false;
  }
  ArrayRef<int64_t> logicalShape = layout.getLogicalShape();
  ArrayRef<int64_t> dimAlignments = layout.getDimAlignments();
  for (size_t i = 0; i < logicalShape.size(); ++i) {
    if (ttmlir::utils::alignUp(logicalShape[i], dimAlignments[i]) !=
        logicalShape[i]) {
      return true;
    }
  }
  return false;
}

// Some layout fields describe how an already-materialized physical buffer
// should be interpreted. If those fields are the only differences, no data
// movement is required.
bool canReinterpretLayout(const PlanState &current, const PlanState &target) {
  if (!current.hasLayout() || !target.hasLayout()) {
    return false;
  }
  return current.type.getShape() == target.type.getShape() &&
         current.type.getElementType() == target.type.getElementType() &&
         current.getGridShape() == target.getGridShape() &&
         current.getLayout()->getLogicalShape() ==
             target.getLayout()->getLogicalShape() &&
         current.getLayout()->getDimAlignments() ==
             target.getLayout()->getDimAlignments() &&
         current.getLayout()->getMemorySpace() ==
             target.getLayout()->getMemorySpace() &&
         current.getLayout()->getMemoryLayout() ==
             target.getLayout()->getMemoryLayout();
}

// Collapse an ND virtual grid into a 2D bounce shape that fits within the
// device grid. Used when staging data through interleaved DRAM on a unit grid.
llvm::SmallVector<int64_t>
computeVirtualGridBounceShape(ArrayRef<int64_t> virtualGridShape,
                              ArrayRef<int64_t> deviceGridShape) {
  assert(virtualGridShape.size() >= 2u);
  llvm::SmallVector<int64_t> collapsed(2);
  collapsed[0] = virtualGridShape[0];
  for (int64_t i = 1; i < static_cast<int64_t>(virtualGridShape.size()) - 1;
       ++i) {
    collapsed[0] *= virtualGridShape[i];
  }
  collapsed[1] = virtualGridShape.back();

  llvm::SmallVector<int64_t> bounceShape;
  for (size_t i = 0; i < collapsed.size(); i++) {
    bounceShape.push_back(collapsed[i] > deviceGridShape[i]
                              ? std::gcd(collapsed[i], deviceGridShape[i])
                              : collapsed[i]);
  }
  return bounceShape;
}

// Compute canonical (grid-aware) collapsed intervals and dim alignments for a
// reference layout, matching what D2MGridSelection produces. Used when a
// virtual-grid tensor needs to be normalized for bouncing through DRAM.
std::pair<DenseIntElementsAttr, llvm::SmallVector<int64_t>>
computeGridAwareCollapsedIntervalsAndDimAlignments(
    MLIRContext *ctx, ttcore::MetalLayoutAttr referenceLayout,
    ArrayRef<int64_t> gridShape) {
  auto logicalShape = referenceLayout.getLogicalShape();
  auto collapsedIntervals = referenceLayout.computeDefaultCollapsedIntervals(
      ctx, logicalShape.size());
  auto dimAlignments = ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
      logicalShape, gridShape,
      ttcore::MetalLayoutAttr::normalizeAndFlattenIntervals(
          collapsedIntervals, logicalShape.size()));
  return {collapsedIntervals, dimAlignments};
}

// Construct the device tensor type for a tensor entering the device from
// system memory. Virtual grids are staged via the unit grid in interleaved
// DRAM; the subsequent DRAM→L1 step handles the scatter via a view map.
RankedTensorType createDeviceType(MLIRContext *ctx, RankedTensorType systemType,
                                  ttcore::MetalLayoutAttr referenceLayout,
                                  RankedTensorType referenceType,
                                  ArrayRef<int64_t> targetGridShape) {
  SmallVector<int64_t> tensorGridShape =
      llvm::to_vector(referenceLayout.getGridShape(referenceType));
  bool virtualBounceNeeded = ttmlir::d2m::utils::grids::requiresVirtualGrid(
      tensorGridShape, targetGridShape);

  ttcore::MetalLayoutAttr layout;
  if (virtualBounceNeeded) {
    auto collapsedIntervals =
        computeGridAwareCollapsedIntervalsAndDimAlignments(ctx, referenceLayout,
                                                           targetGridShape)
            .first;
    tensorGridShape.assign(targetGridShape.size(), 1);
    assert(collapsedIntervals.getType().getDimSize(0) == 2);
    // Preserve the source layout's alignments so host staging keeps the same
    // logical strides while the DRAM bounce is collapsed to 2D.
    layout = ttcore::MetalLayoutAttr::get(
        ctx, referenceLayout.getLogicalShape(),
        referenceLayout.getDimAlignments(), collapsedIntervals,
        referenceLayout.getOobVal(), ttcore::MemorySpace::DeviceDRAM,
        ttcore::TensorMemoryLayout::Interleaved);
  } else {
    layout = ttcore::MetalLayoutAttr::get(
        ctx, referenceLayout.getLogicalShape(),
        referenceLayout.getDimAlignments(),
        referenceLayout.getCollapsedIntervals(), referenceLayout.getOobVal(),
        ttcore::MemorySpace::DeviceL1, referenceLayout.getMemoryLayout());
  }

  ArrayRef<int64_t> tileShape;
  if (ttcore::isTiled(systemType)) {
    tileShape = ttcore::getTensorTileShape(systemType);
  }
  return RankedTensorType::get(
      layout.getDeviceShape(tensorGridShape, tileShape),
      systemType.getElementType(), layout);
}

// Produce a variant of an existing device tensor type with selectively
// modified fields (memory space, grid, element type, tile shape). Re-collapses
// virtual-grid shapes when `reblockVirtualGridShapes` is set; otherwise
// preserves the original dim alignments and collapsed intervals.
RankedTensorType
modifyDeviceType(MLIRContext *ctx, RankedTensorType baseType,
                 ttcore::MetalLayoutAttr baseLayout,
                 ArrayRef<int64_t> targetGridShape,
                 AffineMap existingRemapping = AffineMap(),
                 std::optional<ttcore::MemorySpace> newMemSpace = {},
                 std::optional<ArrayRef<int64_t>> newTensorGrid = {},
                 std::optional<Type> newElementType = {},
                 std::optional<ArrayRef<int64_t>> newTileShape = {},
                 bool reblockVirtualGridShapes = false) {
  assert(baseLayout && "modifyDeviceType requires a layout");
  auto memSpace = newMemSpace.value_or(baseLayout.getMemorySpace());
  auto elementType = newElementType.value_or(baseType.getElementType());

  bool hasVirtualGrid = existingRemapping && !existingRemapping.isEmpty() &&
                        !existingRemapping.isIdentity();
  SmallVector<int64_t> tensorGrid;
  bool needsReblock = hasVirtualGrid;
  if (newTensorGrid.has_value()) {
    tensorGrid.assign(newTensorGrid->begin(), newTensorGrid->end());
  } else {
    tensorGrid = llvm::to_vector(baseLayout.getGridShape(baseType));
    needsReblock =
        needsReblock || ttmlir::d2m::utils::grids::requiresVirtualGrid(
                            tensorGrid, targetGridShape);
    if (needsReblock && reblockVirtualGridShapes) {
      tensorGrid = computeVirtualGridBounceShape(tensorGrid, targetGridShape);
    }
  }

  ttcore::MetalLayoutAttr layout;
  if (needsReblock && reblockVirtualGridShapes) {
    auto [collapsedIntervals, dimAlignments] =
        computeGridAwareCollapsedIntervalsAndDimAlignments(ctx, baseLayout,
                                                           tensorGrid);
    layout = ttcore::MetalLayoutAttr::get(
        ctx, baseLayout.getLogicalShape(), dimAlignments, collapsedIntervals,
        baseLayout.getOobVal(), memSpace, baseLayout.getMemoryLayout());
  } else {
    layout = ttcore::MetalLayoutAttr::get(
        ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
        baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(), memSpace,
        baseLayout.getMemoryLayout());
  }

  ArrayRef<int64_t> tileShape;
  if (mlir::isa<ttcore::TileType>(elementType)) {
    tileShape =
        newTileShape.value_or(ttcore::getTensorTileShapeOrEmpty(baseType));
  }
  return RankedTensorType::get(layout.getDeviceShape(tensorGrid, tileShape),
                               elementType, layout);
}

// Decompose a tilized layout change that cannot be expressed as a direct
// affine-map reblock (e.g., unaligned shards) into scalar-space form:
// Untilize → Reshard_scalar → Tilize_back. Emits three Steps that the
// minimizer may further simplify against surrounding Tilize/Untilize.
void emitTilizedReshardDecomposition(
    Plan &plan, MLIRContext *ctx, RankedTensorType currentType,
    ttcore::MetalLayoutAttr currentLayout, RankedTensorType targetType,
    ttcore::MetalLayoutAttr targetLayout, ArrayRef<int64_t> targetGridShape,
    AffineMap currentRemapping, AffineMap currentVgmForward,
    AffineMap currentVgmInverse, AffineMap targetVgmForward,
    AffineMap targetVgmInverse) {
  Type scalarType = getScalarType(currentType.getElementType());
  auto untilizedType = modifyDeviceType(
      ctx, currentType, currentLayout, targetGridShape, currentRemapping,
      ttcore::MemorySpace::DeviceL1, /*newTensorGrid=*/{}, scalarType,
      /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/false);
  plan.push_back(UntilizeStep{
      makeOutputSpec(untilizedType, currentVgmForward, currentVgmInverse)});

  auto scalarTargetLayout = ttcore::MetalLayoutAttr::get(
      ctx, targetLayout.getLogicalShape(), targetLayout.getDimAlignments(),
      targetLayout.getCollapsedIntervals(), targetLayout.getOobVal(),
      ttcore::MemorySpace::DeviceL1, targetLayout.getMemoryLayout());
  auto scalarTargetGridShape = targetLayout.getGridShape(targetType);
  auto scalarTargetType =
      RankedTensorType::get(scalarTargetLayout.getDeviceShape(
                                scalarTargetGridShape, /*tileShape=*/{}),
                            scalarType, scalarTargetLayout);
  plan.push_back(ReshardStep{
      llvm::to_vector(scalarTargetGridShape),
      llvm::to_vector(targetLayout.getDimAlignments()),
      targetLayout.getCollapsedIntervals(),
      makeOutputSpec(scalarTargetType, targetVgmForward, targetVgmInverse)});

  ArrayRef<int64_t> tileShape = ttcore::getTensorTileShape(targetType);
  auto tiledType = RankedTensorType::get(
      targetLayout.getDeviceShape(targetLayout.getGridShape(targetType),
                                  tileShape),
      targetType.getElementType(), targetLayout);
  plan.push_back(TilizeStep{
      llvm::to_vector(tileShape),
      makeOutputSpec(tiledType, targetVgmForward, targetVgmInverse)});
}

} // namespace

// ============================================================================
// canonicalize
// ============================================================================

Plan canonicalize(const PlanState &src, const PlanState &tgt,
                  ArrayRef<int64_t> targetGridShape, MLIRContext *ctx) {
  Plan plan;
  PlanState current = src;

  // SYSTEM → DEVICE: scalar-element transfer into the device's memory space.
  if (!current.hasLayout() && tgt.hasLayout()) {
    Type scalarElemType = getScalarType(current.type.getElementType());
    auto baseType = createDeviceType(ctx, current.type, *tgt.getLayout(),
                                     tgt.type, targetGridShape);
    auto baseLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(baseType.getEncoding());
    auto scalarType =
        RankedTensorType::get(baseType.getShape(), scalarElemType, baseLayout);
    if (needsVirtualGridBounce(*tgt.getLayout(), tgt.type, targetGridShape)) {
      auto output = makeOutputSpec(scalarType);
      plan.push_back(HostToBounceBufferStep{output});
      updateStateFromOutput(current, output);
    } else {
      auto output = makeOutputSpec(scalarType, tgt.vgmForward, tgt.vgmInverse);
      plan.push_back(HostToDeviceStep{output});
      updateStateFromOutput(current, output);
    }
  }

  // DRAM → L1: DMA the tensor into an L1 buffer with the target's layout.
  if (current.hasLayout() && current.isDRAM() && tgt.hasLayout() &&
      !tgt.isDRAM()) {
    const bool isDRAMInterleaved = current.getLayout()->getMemoryLayout() ==
                                   ttcore::TensorMemoryLayout::Interleaved;
    auto bounceGrid = llvm::to_vector(
        isDRAMInterleaved ? tgt.getGridShape() : current.getGridShape());
    auto l1Type =
        modifyDeviceType(ctx, tgt.type, *tgt.getLayout(), targetGridShape,
                         AffineMap(), ttcore::MemorySpace::DeviceL1, bounceGrid,
                         current.type.getElementType());
    auto output = makeOutputSpec(l1Type, tgt.vgmForward, tgt.vgmInverse);
    plan.push_back(DRAMToL1Step{output, AffineMap()});
    updateStateFromOutput(current, output);
  }

  // TILIZE with the current layout so subsequent mapping changes operate on
  // the target tile shape.
  if (current.hasLayout() && !ttcore::isTiled(current.type) &&
      ttcore::isTiled(tgt.type)) {
    ArrayRef<int64_t> tileShape = ttcore::getTensorTileShape(tgt.type);
    auto tiledType = RankedTensorType::get(
        current.getLayout()->getDeviceShape(current.getGridShape(), tileShape),
        tgt.type.getElementType(), *current.getLayout());
    auto output =
        makeOutputSpec(tiledType, current.vgmForward, current.vgmInverse);
    plan.push_back(TilizeStep{llvm::to_vector(tileShape), output});
    updateStateFromOutput(current, output, current.remapping);
  }

  // UNTILIZE to scalar, preserving virtual-grid shape so downstream steps can
  // handle any remaining layout differences.
  if (current.hasLayout() && ttcore::isTiled(current.type) &&
      !ttcore::isTiled(tgt.type)) {
    Type scalarType = tgt.type.getElementType();
    auto scalarTypeRanked = modifyDeviceType(
        ctx, current.type, *current.getLayout(), targetGridShape,
        current.remapping, /*memSpace=*/{}, /*newTensorGrid=*/{}, scalarType,
        /*newTileShape=*/std::nullopt, /*reblockVirtualGridShapes=*/false);
    auto output = makeOutputSpec(scalarTypeRanked, current.vgmForward,
                                 current.vgmInverse);
    plan.push_back(UntilizeStep{output});
    updateStateFromOutput(current, output, current.remapping);
  }

  // L1 → DRAM: the DMA writes directly into the output DRAM buffer.
  if (current.hasLayout() && !current.isDRAM() && tgt.hasLayout() &&
      tgt.isDRAM()) {
    auto output = makeOutputSpec(tgt.type);
    plan.push_back(L1ToDRAMStep{output, AffineMap()});
    updateStateFromOutput(current, output);
  }

  if (canReinterpretLayout(current, tgt) && current.type != tgt.type) {
    // Metadata-only changes that do not require data mutation can be erased:
    // downstream ops may continue to use the existing physical buffer/layout,
    // which avoids creating type-only views that later lowering does not need.
    bool needsMaskForOobChange =
        current.getLayout()->getOobVal() != tgt.getLayout()->getOobVal() &&
        needsMasking(*tgt.getLayout(), tgt.type);
    if (needsMaskForOobChange) {
      plan.push_back(ReinterpretLayoutStep{tgt.type});
    }
    current.type = tgt.type;
  }

  // MAPPING CHANGE in L1 (grid / alignments / collapse).
  if (current.hasLayout() && tgt.hasLayout() && current.isL1() &&
      (ttcore::isTiled(current.type) == ttcore::isTiled(tgt.type))) {
    bool needsMappingChange = (current.getGridShape() != tgt.getGridShape() ||
                               current.getLayout()->getLogicalShape() !=
                                   tgt.getLayout()->getLogicalShape() ||
                               current.getLayout()->getCollapsedIntervals() !=
                                   tgt.getLayout()->getCollapsedIntervals() ||
                               current.getLayout()->getDimAlignments() !=
                                   tgt.getLayout()->getDimAlignments());

    if (needsMappingChange) {
      bool isSimpleReblocking = (current.getLayout()->getLogicalShape() ==
                                     tgt.getLayout()->getLogicalShape() &&
                                 current.getLayout()->getDimAlignments() ==
                                     tgt.getLayout()->getDimAlignments() &&
                                 current.getLayout()->getCollapsedIntervals() ==
                                     tgt.getLayout()->getCollapsedIntervals());
      bool bothTilized =
          ttcore::isTiled(current.type) && ttcore::isTiled(tgt.type);

      if (bothTilized && !isSimpleReblocking) {
        // Tilized, misaligned shards: mapping must go through scalar space.
        emitTilizedReshardDecomposition(
            plan, ctx, current.type, *current.getLayout(), tgt.type,
            *tgt.getLayout(), targetGridShape, current.remapping,
            current.vgmForward, current.vgmInverse, tgt.vgmForward,
            tgt.vgmInverse);
        updateStateFromOutput(current,
                              std::get<TilizeStep>(plan.back()).output);
      } else {
        auto deviceShape = llvm::to_vector(tgt.type.getShape());
        auto intermediateLayout = ttcore::MetalLayoutAttr::get(
            ctx, tgt.getLayout()->getLogicalShape(),
            tgt.getLayout()->getDimAlignments(),
            tgt.getLayout()->getCollapsedIntervals(),
            tgt.getLayout()->getOobVal(), ttcore::MemorySpace::DeviceL1,
            tgt.getLayout()->getMemoryLayout());
        auto intermediateType = RankedTensorType::get(
            deviceShape, current.type.getElementType(), intermediateLayout);
        auto output =
            makeOutputSpec(intermediateType, tgt.vgmForward, tgt.vgmInverse);
        plan.push_back(
            ReshardStep{llvm::to_vector(tgt.getGridShape()),
                        llvm::to_vector(tgt.getLayout()->getDimAlignments()),
                        tgt.getLayout()->getCollapsedIntervals(), output});
        updateStateFromOutput(current, output);
      }
    }
  }

  // Materialize into a fresh buffer when VGM changes or when we need to clear
  // an existing remapping before applying a different downstream view.
  // Do not clear virtual-grid/remapping metadata on the way back to host here.
  // If the grid is still virtual, the host-transfer path below must collapse it
  // using that mapping; an early identity rebuffer would scramble the data.
  bool canMaterializeMetadataChange = !tgt.isSystem();
  bool needsVgmChange =
      canMaterializeMetadataChange && (current.vgmForward != tgt.vgmForward ||
                                       current.vgmInverse != tgt.vgmInverse);
  bool needsRemapMaterialization =
      canMaterializeMetadataChange && current.remapping &&
      !current.remapping.isEmpty() && current.remapping != tgt.remapping;
  if (needsVgmChange || needsRemapMaterialization) {
    auto output = makeOutputSpec(current.type, tgt.vgmForward, tgt.vgmInverse);
    plan.push_back(RebufferStep{output});
    updateStateFromOutput(current, output);
  }

  // Apply any remaining target remapping explicitly as a view in the final
  // phase rather than re-deriving it from surrounding IR.
  if (current.remapping != tgt.remapping && tgt.remapping &&
      !tgt.remapping.isEmpty()) {
    plan.push_back(RemapStep{current.type, tgt.remapping});
    current.remapping = tgt.remapping;
  }

  // MASKING: after reshards/layout changes have established the final tiled
  // layout, write the target's OOB fill value into any padded region.
  if (current.hasLayout() && ttcore::isTiled(current.type) &&
      needsMasking(*current.getLayout(), current.type)) {
    plan.push_back(MaskStep{
        current.getLayout()->getOobVal(),
        llvm::to_vector(current.getLayout()->getLogicalShape()),
        makeOutputSpec(current.type, current.vgmForward, current.vgmInverse)});
  }

  // VIRTUAL GRID COLLAPSE: when heading to host, reshape any residual virtual
  // grid into the device grid before the transfer.
  if (current.hasLayout() && tgt.isSystem()) {
    auto currentGridShape = current.getGridShape();
    auto targetGridShapeForLayout =
        tgt.hasLayout() ? tgt.getGridShape() : targetGridShape;
    bool needsVirtualGridCollapse =
        ttmlir::d2m::utils::grids::requiresVirtualGrid(
            currentGridShape, targetGridShapeForLayout);
    if (needsVirtualGridCollapse && current.isL1()) {
      auto reblocked = modifyDeviceType(
          ctx, current.type, *current.getLayout(), targetGridShape,
          current.remapping, ttcore::MemorySpace::DeviceL1,
          /*newTensorGrid=*/{}, /*newElementType=*/{},
          /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/true);
      auto reblockedLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(reblocked.getEncoding());
      auto output = makeOutputSpec(reblocked);
      plan.push_back(
          ReshardStep{llvm::to_vector(reblockedLayout.getGridShape(reblocked)),
                      llvm::to_vector(reblockedLayout.getDimAlignments()),
                      reblockedLayout.getCollapsedIntervals(), output});
      updateStateFromOutput(current, output);
    }
  }

  // DEVICE → SYSTEM.
  if (current.hasLayout() && !tgt.hasLayout()) {
    plan.push_back(DeviceToHostStep{tgt.type});
  }

  return plan;
}

// ============================================================================
// minimize
// ============================================================================

namespace {

// Return true iff the adjacent pair (a; b) reduces to nothing.
bool cancels(const Step &a, const Step &b) {
  // Tilize; Untilize → ∅ and Untilize; Tilize → ∅.
  if (std::holds_alternative<TilizeStep>(a) &&
      std::holds_alternative<UntilizeStep>(b)) {
    return true;
  }
  if (std::holds_alternative<UntilizeStep>(a) &&
      std::holds_alternative<TilizeStep>(b)) {
    return true;
  }
  return false;
}

// Return a fused Step iff the adjacent pair (a; b) merges into a single step.
std::optional<Step> tryFuse(const Step &a, const Step &b) {
  // F3: Reshard; Reshard → Reshard(second).
  if (std::holds_alternative<ReshardStep>(a) &&
      std::holds_alternative<ReshardStep>(b)) {
    return b;
  }
  // F6: Mask(v1); Mask(v2) → Mask(v2). Canonicalizer invariant: v2 != Undef.
  if (std::holds_alternative<MaskStep>(a) &&
      std::holds_alternative<MaskStep>(b)) {
    return b;
  }
  return std::nullopt;
}

bool applyCancellationPass(Plan &plan) {
  bool changed = false;
  for (size_t i = 0; i + 1 < plan.size();) {
    if (cancels(plan[i], plan[i + 1])) {
      plan.erase(plan.begin() + i, plan.begin() + i + 2);
      changed = true;
      // Re-check the new adjacency created by the erase.
      if (i > 0) {
        --i;
      }
    } else {
      ++i;
    }
  }
  return changed;
}

bool applyFusionPass(Plan &plan) {
  bool changed = false;
  for (size_t i = 0; i + 1 < plan.size();) {
    if (auto fused = tryFuse(plan[i], plan[i + 1])) {
      plan[i] = std::move(*fused);
      plan.erase(plan.begin() + i + 1);
      changed = true;
      // Re-check in case the fused step enables further fusion with its new
      // right neighbor.
    } else {
      ++i;
    }
  }
  return changed;
}

// Return true iff (a; b) has the same semantics as (b; a). Only kind-based
// cases are encoded here; commutations with payload preconditions (e.g.
// Tilize ⇌ Reshard only when tile-aligned) require more state and are
// deliberately omitted.
bool commutesFreely(const Step &a, const Step &b) {
  // Mask's effect is in logical coordinates, which Reshard preserves.
  if (std::holds_alternative<MaskStep>(a) &&
      std::holds_alternative<ReshardStep>(b)) {
    return true;
  }
  if (std::holds_alternative<ReshardStep>(a) &&
      std::holds_alternative<MaskStep>(b)) {
    return true;
  }
  return false;
}

// Return true iff swapping plan[i] and plan[i+1] creates a new adjacency
// (either with plan[i-1] or with plan[i+2]) that the cancel / fuse passes
// would then simplify. Gates the commutation pass against infinite swap
// loops and unmotivated churn.
bool swapEnablesSimplification(const Plan &plan, size_t i) {
  const Step &a = plan[i];
  const Step &b = plan[i + 1];
  if (i > 0) {
    const Step &leftNeighbor = plan[i - 1];
    if (cancels(leftNeighbor, b) || tryFuse(leftNeighbor, b).has_value()) {
      return true;
    }
  }
  if (i + 2 < plan.size()) {
    const Step &rightNeighbor = plan[i + 2];
    if (cancels(a, rightNeighbor) || tryFuse(a, rightNeighbor).has_value()) {
      return true;
    }
  }
  return false;
}

bool applyCommutationPass(Plan &plan) {
  bool changed = false;
  for (size_t i = 0; i + 1 < plan.size(); ++i) {
    if (commutesFreely(plan[i], plan[i + 1]) &&
        swapEnablesSimplification(plan, i)) {
      std::swap(plan[i], plan[i + 1]);
      changed = true;
    }
  }
  return changed;
}

} // namespace

Plan minimize(Plan plan) {
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= applyCancellationPass(plan);
    changed |= applyFusionPass(plan);
    changed |= applyCommutationPass(plan);
  }
  return plan;
}

} // namespace mlir::tt::d2m
