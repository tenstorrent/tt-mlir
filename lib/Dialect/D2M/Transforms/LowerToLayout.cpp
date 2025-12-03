// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Utils/AffineMapUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include <algorithm>
#include <string>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERTOLAYOUT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Helper struct to encapsulate tensor info; this allows us to package
// MetalLayoutAttr as optional gracefully.
namespace {
struct TensorInfo {
  RankedTensorType type;
  std::optional<ttcore::MetalLayoutAttr> layout;

  static TensorInfo from(Value val) {
    auto type = mlir::cast<RankedTensorType>(val.getType());
    auto layout =
        mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(type.getEncoding());
    return {type, layout ? std::optional(layout) : std::nullopt};
  }

  bool hasLayout() const { return layout.has_value(); }

  ttcore::MemorySpace getMemorySpace() const {
    return layout ? layout->getMemorySpace() : ttcore::MemorySpace::System;
  }

  bool isL1() const {
    return hasLayout() &&
           layout->getMemorySpace() == ttcore::MemorySpace::DeviceL1;
  }

  bool isDRAM() const {
    return hasLayout() &&
           layout->getMemorySpace() == ttcore::MemorySpace::DeviceDRAM;
  }

  bool isSystem() const {
    return !hasLayout() ||
           layout->getMemorySpace() == ttcore::MemorySpace::System;
  }

  ArrayRef<int64_t> getGridShape() const {
    assert(hasLayout() && "Cannot get grid shape without layout");
    return layout->getGridShape(type);
  }
};

// Helper to analyze compound ToLayoutOp transformations.
// Helper to extract scalar type from potentially tiled type.
static Type getScalarType(Type type) {
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(type)) {
    return tileType.getElementType();
  }
  return type;
}

} // namespace

namespace {
class D2MLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
  // Helper struct to build intermediate bounce types.
  class BounceTypeBuilder {
  public:
    explicit BounceTypeBuilder(MLIRContext *ctx) : ctx(ctx) {}

    // Computes a workable bounce shape grid for a virtual grid.
    llvm::SmallVector<int64_t>
    computeVirtualGridBounceShape(ArrayRef<int64_t> virtualGridShape,
                                  ArrayRef<int64_t> deviceGridShape) const {
      // TODO(bgrady-tt): Generalize to N dimensions.
      assert(virtualGridShape.size() == 2);
      assert(virtualGridShape[0] > deviceGridShape[0] ^
             virtualGridShape[1] > deviceGridShape[1]);

      llvm::SmallVector<int64_t> ret;
      if (virtualGridShape[0] > deviceGridShape[0]) {
        int64_t divisor = std::gcd(virtualGridShape[0], deviceGridShape[0]);
        ret = {divisor, virtualGridShape[1]};
      } else {
        int64_t divisor = std::gcd(virtualGridShape[1], deviceGridShape[1]);
        ret = {virtualGridShape[0], divisor};
      }
      assert(ret.size());
      return ret;
    }

    // Create a device tensor type from a system tensor type.
    RankedTensorType createDeviceType(RankedTensorType systemType,
                                      ttcore::MetalLayoutAttr referenceLayout,
                                      RankedTensorType referenceType,
                                      ttcore::MemorySpace memSpace,
                                      ArrayRef<int64_t> targetGridShape) {
      SmallVector<int64_t> tensorGridShape =
          llvm::to_vector(referenceLayout.getGridShape(referenceType));
      if (auto metalLayout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
              referenceType.getEncoding());
          metalLayout && !metalLayout.getIndexAffineMap().isEmpty()) {
        bool exceedsPhysicalBounds =
            (tensorGridShape[0] > targetGridShape[0]) ||
            (tensorGridShape[1] > targetGridShape[1]);
        if (exceedsPhysicalBounds) {
          tensorGridShape =
              computeVirtualGridBounceShape(tensorGridShape, targetGridShape);
        }
      }

      // Preserve the reference layout's index_map so GenericOps can properly
      // map virtual grids to physical cores.
      auto layout = ttcore::MetalLayoutAttr::get(
          ctx, referenceLayout.getLogicalShape(),
          referenceLayout.getDimAlignments(),
          referenceLayout.getCollapsedIntervals(), referenceLayout.getOobVal(),
          memSpace, referenceLayout.getMemoryLayout(),
          referenceLayout.getIndexAffineMap());

      ArrayRef<int64_t> tileShape;
      if (ttcore::isTiled(systemType)) {
        tileShape = ttcore::getTensorTileShape(systemType);
      }
      auto deviceShape = layout.getDeviceShape(tensorGridShape, tileShape);

      return RankedTensorType::get(deviceShape, systemType.getElementType(),
                                   layout);
    }

    // Modify an existing device tensor type.
    RankedTensorType
    modifyDeviceType(RankedTensorType baseType,
                     ttcore::MetalLayoutAttr baseLayout,
                     ArrayRef<int64_t> targetGridShape,
                     std::optional<ttcore::MemorySpace> newMemSpace = {},
                     std::optional<ArrayRef<int64_t>> newTensorGrid = {},
                     std::optional<Type> newElementType = {},
                     std::optional<ArrayRef<int64_t>> newTileShape = {},
                     bool reblockVirtualGridShapes = false) {
      assert(baseLayout && "modifyDeviceType requires a layout");

      auto memSpace = newMemSpace.value_or(baseLayout.getMemorySpace());
      auto elementType = newElementType.value_or(baseType.getElementType());

      SmallVector<int64_t> tensorGrid;
      bool didReblockVirtualGrid = false;
      if (newTensorGrid.has_value()) {
        tensorGrid.assign(newTensorGrid->begin(), newTensorGrid->end());
      } else {
        auto currentGrid = llvm::to_vector(baseLayout.getGridShape(baseType));
        tensorGrid = currentGrid;
        bool hasVirtualGrid = !baseLayout.getIndexAffineMap().isEmpty();
        if (hasVirtualGrid && reblockVirtualGridShapes) {
          tensorGrid =
              computeVirtualGridBounceShape(currentGrid, targetGridShape);
          didReblockVirtualGrid = true;
        }
      }

      AffineMap indexMap = AffineMap::get(ctx);
      if (memSpace == ttcore::MemorySpace::DeviceL1 &&
          baseLayout.getMemorySpace() == ttcore::MemorySpace::DeviceL1 &&
          !didReblockVirtualGrid) {
        indexMap = baseLayout.getIndexAffineMap();
      }

      auto layout = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(), memSpace,
          baseLayout.getMemoryLayout(), indexMap);

      ArrayRef<int64_t> tileShape;
      if (mlir::isa<ttcore::TileType>(elementType)) {
        tileShape =
            newTileShape.value_or(ttcore::getTensorTileShapeOrEmpty(baseType));
      }
      auto deviceShape = layout.getDeviceShape(tensorGrid, tileShape);

      return RankedTensorType::get(deviceShape, elementType, layout);
    }

  private:
    MLIRContext *ctx;
  };

public:
  D2MLowerToLayoutRewriter(MLIRContext *context,
                           ArrayRef<int64_t> targetGridShape)
      : OpRewritePattern(context, PatternBenefit(1)),
        targetGridShape(targetGridShape) {}

  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  // Lower mapping transformations (grid redistribution, padding changes,
  // collapse changes, index map transformations) to ViewLayoutOp + DMA generic.
  // The ViewLayoutOp represents the transformation as an affine map, and the
  // DMA generic materializes the data movement for L1→L1 transformations.
  static Value lowerMappingChange(PatternRewriter &rewriter, Value input,
                                  Value output, Location loc,
                                  ArrayRef<int64_t> targetGridShape) {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    // Precondition: both operands must have layouts, be in the same memory
    // space, and have the same element type. These are guaranteed by the
    // compound splitting logic upstream.
    assert((inputInfo.hasLayout() && outputInfo.hasLayout()) &&
           "Mapping change requires both input and output to have layouts");
    assert(inputInfo.getMemorySpace() == outputInfo.getMemorySpace() &&
           "Mapping change should not change memory space");
    assert(inputInfo.type.getElementType() ==
               outputInfo.type.getElementType() &&
           "Mapping change should not change element type");

    auto inputLayout = *inputInfo.layout;
    auto outputLayout = *outputInfo.layout;

    // Check if output has virtual grid for later use.
    bool outputHasVirtualGrid = !outputLayout.getIndexAffineMap().isEmpty();

    // Classify the type of mapping change to choose the optimal approach.
    // Simple reblocking: only grid shape differs, all other layout properties
    // are identical. For tilized tensors, we can use a direct device-space
    // reblock map (calculateReblockMap) which is more efficient and avoids
    // issues with unaligned tensors where logical shapes don't divide evenly
    // into tiles.
    bool isSimpleReblocking =
        (inputLayout.getLogicalShape() == outputLayout.getLogicalShape() &&
         inputLayout.getDimAlignments() == outputLayout.getDimAlignments() &&
         inputLayout.getCollapsedIntervals() ==
             outputLayout.getCollapsedIntervals());

    bool bothTilized =
        ttcore::isTiled(inputInfo.type) && ttcore::isTiled(outputInfo.type);

    AffineMap viewMap;

    if (isSimpleReblocking && bothTilized) {
      // Fast path: pure grid reblocking on tilized tensors.
      // Use calculateReblockMap which works directly on device shapes without
      // going through logical space (avoids tile alignment issues).
      viewMap = ttmlir::utils::calculateReblockMap(inputInfo.type.getShape(),
                                                   outputInfo.type.getShape(),
                                                   rewriter.getContext());
    } else {
      // Complex mapping: layout properties differ (padding, collapse, etc).
      // Use buildLayoutTransformMap which goes through logical space.
      // For tilized tensors, this should only be called from the untilized
      // decomposition path in step 5.

      // Build an affine map that transforms input device coordinates to output
      // device coordinates via the shared logical space. This map handles grid
      // redistribution, collapse changes, padding changes, and virtual grid
      // index_maps.
      viewMap = ttcore::utils::buildLayoutTransformMap(
          inputLayout, inputInfo.type, outputLayout, outputInfo.type);
    }

    // Embed the transformation map in the output layout.
    auto newLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), outputLayout.getLogicalShape(),
        outputLayout.getDimAlignments(), outputLayout.getCollapsedIntervals(),
        outputLayout.getOobVal(), outputLayout.getMemorySpace(),
        outputLayout.getMemoryLayout(), viewMap);

    auto viewType =
        RankedTensorType::get(outputInfo.type.getShape(),
                              outputInfo.type.getElementType(), newLayout);

    Value viewOp = rewriter.create<ViewLayoutOp>(loc, viewType, input,
                                                 /*reinterpretLayout=*/false);

    // Materialize L1→L1 transformations with a DMA generic that performs the
    // actual data movement according to the view's affine map.
    if (!inputInfo.isDRAM() && !outputInfo.isDRAM()) {
      auto gridShape = outputInfo.getGridShape();

      // If the output has a virtual grid (and by our early check, input does
      // too with the same grid shape), we need to include the virtual→physical
      // coordinate translation in the grid attribute.
      ttcore::GridAttr grid;
      if (outputHasVirtualGrid) {
        // Check if this is actually a virtual grid (grid shape exceeds physical
        // bounds)
        bool isVirtualGrid = gridShape[0] > targetGridShape[0] ||
                             gridShape[1] > targetGridShape[1];

        if (isVirtualGrid) {
          // Create the virtual grid coordinate maps and use the inverse map
          // for the grid's coordinate translation.
          auto [fwdMap, invMap] = ttmlir::d2m::utils::grids::createCoreVirtMaps(
              rewriter.getContext(), gridShape, targetGridShape);
          grid =
              ttcore::GridAttr::get(rewriter.getContext(), gridShape, invMap);
        } else {
          // If the operand has index_map but doesn't exceed physical grid
          // (e.g., reblocking, transpose), derive the grid inverse map from
          // the output's index_map to ensure roundtrip consistency.
          auto indexMap = outputLayout.getIndexAffineMap();
          auto invMap = ttmlir::utils::createGridInverseMapFromIndexMap(
              indexMap, gridShape.size(), rewriter.getContext());
          grid =
              ttcore::GridAttr::get(rewriter.getContext(), gridShape, invMap);
        }
      } else {
        grid = ttcore::GridAttr::get(rewriter.getContext(), gridShape);
      }

      const size_t gridRank = gridShape.size();

      // Build identity indexing maps for the generic operation. The view's
      // affine map handles all address transformations.
      ArrayAttr indexingMaps, iteratorTypes;
      std::tie(indexingMaps, iteratorTypes) =
          GenericOp::buildParallelAffineMapsAndIteratorTypes(
              rewriter, /*arity=*/2, gridRank);
      auto indexingMap = mlir::cast<AffineMapAttr>(indexingMaps[0]);

      return rewriter
          .create<GenericOp>(
              loc, viewOp, output,
              [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
                Value outputCB =
                    builder.create<ReserveOp>(innerLoc, blockArgs[1])
                        .getResult();
                auto dma = builder.create<d2m::DMAOp>(innerLoc, viewOp,
                                                      indexingMap, outputCB);
                builder.create<d2m::DMAWaitOp>(innerLoc, dma);
                builder.create<YieldOp>(innerLoc, outputCB);
              },
              ThreadType::Datamovement, grid)
          .getResult(0);
    }
    // DRAM operations use the view directly without immediate
    // materialization.
    return viewOp;
  }

  static Value lowerSystemLayoutChange(PatternRewriter &rewriter, Value input,
                                       Value output, Location loc) {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    assert(inputInfo.isSystem() != outputInfo.isSystem() &&
           "one of input or output must be system for now");

    // Use the layout of whichever side has a layout (input or output).
    auto deviceLayout =
        inputInfo.isSystem() ? outputInfo.layout : inputInfo.layout;
    assert(deviceLayout.has_value() && "Device side must have a layout");

    // TODO (vwells): If the device side has a virtual grid (non-empty index
    // map), ideally we should materialize the view before system transfer
    // (similar to MaterializeViewReturns pass). For now, we allow it and let
    // downstream passes handle it.

    // Create the final ToLayoutOp with layout attribute set
    return rewriter.create<ToLayoutOp>(loc, input, output, *deviceLayout)
        .getResult(0);
  }

  // Return true if the input operand to a ToLayoutOp is itself a result of a
  // device->device memspace ToLayoutOp.
  static bool producerMustBeLoweredFirst(ToLayoutOp op) {
    if (auto producer = op.getInput().getDefiningOp<ToLayoutOp>()) {
      auto producerInputInfo = TensorInfo::from(producer.getInput());
      auto producerOutputInfo = TensorInfo::from(producer.getOutput());

      // Check if both producer's input and output are on device
      // (i.e., both have layouts and neither is system memory).
      if (producerInputInfo.hasLayout() && producerOutputInfo.hasLayout() &&
          !producerInputInfo.isSystem() && !producerOutputInfo.isSystem()) {
        return true;
      }
    }
    return false;
  }

  Value lowerDatamovementGeneric(PatternRewriter &rewriter, Value input,
                                 Value output, Location loc) const {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    if (inputInfo.isSystem() || outputInfo.isSystem()) {
      return lowerSystemLayoutChange(rewriter, input, output, loc);
    }

    // Both input and output should have layouts at this point.
    assert(inputInfo.hasLayout() && outputInfo.hasLayout());

    Value viewInput = input;

    bool isSrcDramOrReblock =
        inputInfo.isDRAM() ||
        (!outputInfo.isDRAM() &&
         (inputInfo.getGridShape() != outputInfo.getGridShape()));

    assert(!(isSrcDramOrReblock && outputInfo.isDRAM()) &&
           "input and output cannot both be remote");

    auto buildConcreteView = [&](Value fromVal, RankedTensorType fromTy,
                                 RankedTensorType toTy) -> Value {
      auto *ctx = rewriter.getContext();
      AffineMap map = ttmlir::utils::calculateReblockMap(fromTy.getShape(),
                                                         toTy.getShape(), ctx);
      auto baseLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(fromTy.getEncoding());

      auto enc = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(),
          baseLayout.getMemorySpace(), baseLayout.getMemoryLayout(), map);
      auto resultTy =
          RankedTensorType::get(toTy.getShape(), toTy.getElementType(), enc);
      return rewriter
          .create<ViewLayoutOp>(loc, resultTy, fromVal,
                                /*reinterpretLayout=*/false)
          .getResult();
    };

    if (isSrcDramOrReblock) {
      viewInput = buildConcreteView(input, inputInfo.type, outputInfo.type);
    }

    Value viewOutput = output;
    if (outputInfo.isDRAM()) {
      viewOutput = buildConcreteView(output, outputInfo.type, inputInfo.type);
    }

    const size_t gridRank = outputInfo.getGridShape().size();

    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, gridRank);
    auto indexingMap = mlir::cast<AffineMapAttr>(indexingMaps[0]);

    return rewriter
        .create<GenericOp>(
            loc, viewInput, viewOutput,
            [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
              DMAOp dma;
              Value yield;
              if (isSrcDramOrReblock) {
                Value outputCB =
                    builder.create<ReserveOp>(innerLoc, blockArgs[1])
                        .getResult();
                dma = builder.create<d2m::DMAOp>(innerLoc, viewInput,
                                                 indexingMap, outputCB);
                yield = outputCB;
              } else {
                // Note: Naturally you'd think to use a WaitOp since this is in
                // input cb, but in the layout lowering there is no producer
                // thread. The ReserveOp here effectively unwraps the CB so the
                // DMA can access it.
                Value inputCB =
                    builder.create<ReserveOp>(innerLoc, blockArgs[0])
                        .getResult();
                dma = builder.create<d2m::DMAOp>(innerLoc, inputCB, viewOutput,
                                                 indexingMap);
                yield = inputCB;
              }
              builder.create<d2m::DMAWaitOp>(innerLoc, dma);
              builder.create<YieldOp>(innerLoc, yield);
            },
            ThreadType::Datamovement)
        .getResult(0);
  }

  Value lowerFormatConversionGeneric(PatternRewriter &rewriter, Value input,
                                     Value output, Location loc) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(output.getType());
    bool inputTiled = ttcore::isTiled(inputType);
    bool outputTiled = ttcore::isTiled(outputType);
    assert(inputTiled != outputTiled &&
           "one of input or output must be tiled for now");

    return rewriter
        .create<GenericOp>(
            loc, input, output,
            [=](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
              Value src =
                  builder.create<WaitOp>(innerLoc, blockArgs[0]).getResult();
              Value dst =
                  builder.create<ReserveOp>(innerLoc, blockArgs[1]).getResult();
              if (inputTiled) {
                builder.create<TileUntilizeBlockOp>(innerLoc, src, dst);
              } else {
                builder.create<TileTilizeBlockOp>(innerLoc, src, dst);
              }
              builder.create<YieldOp>(innerLoc, dst);
            },
            ThreadType::Compute)
        .getResult(0);
  }

  ToLayoutOp createToLayoutOp(PatternRewriter &rewriter, Location loc,
                              Value input, RankedTensorType desiredType) const {
    auto layout =
        mlir::cast<ttcore::MetalLayoutAttr>(desiredType.getEncoding());
    auto output = rewriter.create<d2m::EmptyOp>(
        loc, desiredType.getShape(), desiredType.getElementType(), layout);
    return rewriter.create<d2m::ToLayoutOp>(loc, input, output);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               RankedTensorType bounceType) const {
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceType);
    return rewriter
        .replaceOpWithNewOp<d2m::ToLayoutOp>(op, bounced->getResult(0),
                                             op.getOutput())
        ->getResult(0);
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    // Skip ToLayoutOps that are already lowered (have layout attribute set)
    if (op.getLayout()) {
      return failure();
    }

    // Use producer-first ordering to ensure dependencies are lowered first.
    if (producerMustBeLoweredFirst(op)) {
      return failure();
    }

    auto targetInfo = TensorInfo::from(op.getOutput());
    auto currentInfo = TensorInfo::from(op.getInput());
    Value currentValue = op.getInput();

    BounceTypeBuilder typeBuilder(rewriter.getContext());

    // === TRANSFORMATION PIPELINE ===
    // Apply transformations in priority order
    // Each step emits lowered ops and updates currentValue/currentInfo

    // Helper to create empty ops for intermediate types. If the type matches
    // the final target, reuse the original output.
    auto createEmpty = [&](RankedTensorType type) -> Value {
      // If this type matches the final target, reuse the original output
      if (type == op.getOutput().getType()) {
        return op.getOutput();
      }

      auto layout = mlir::dyn_cast<ttcore::MetalLayoutAttr>(type.getEncoding());
      return rewriter
          .create<d2m::EmptyOp>(op.getLoc(), type.getShape(),
                                type.getElementType(), layout)
          .getResult();
    };

    // 1. SYSTEM→DEVICE: Transfer to L1 with same element type as input
    if (!currentInfo.hasLayout() && targetInfo.hasLayout()) {
      // System transfer can ONLY change memory space, not element type
      // Create L1 intermediate with scalar element type (same as system input)
      Type scalarElemType = getScalarType(currentInfo.type.getElementType());
      auto l1Type = typeBuilder.createDeviceType(
          currentInfo.type, *targetInfo.layout, targetInfo.type,
          ttcore::MemorySpace::DeviceL1, targetGridShape);

      // Force scalar element type for the L1 intermediate.
      auto l1Layout = mlir::cast<ttcore::MetalLayoutAttr>(l1Type.getEncoding());
      auto scalarL1Type =
          RankedTensorType::get(l1Type.getShape(), scalarElemType, l1Layout);

      auto l1Empty = createEmpty(scalarL1Type);
      currentValue =
          lowerSystemLayoutChange(rewriter, currentValue, l1Empty, op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 2. DRAM→L1: Must happen before other device ops
    // Use target's layout characteristics.
    if (currentInfo.hasLayout() && currentInfo.isDRAM() &&
        targetInfo.hasLayout() && !targetInfo.isDRAM()) {
      // Use target's layout but force L1 and preserve current's grid shape.
      auto l1Type = typeBuilder.modifyDeviceType(
          targetInfo.type, *targetInfo.layout, targetGridShape,
          ttcore::MemorySpace::DeviceL1, currentInfo.getGridShape(),
          currentInfo.type.getElementType());
      auto l1Empty = createEmpty(l1Type);
      currentValue = lowerDatamovementGeneric(rewriter, currentValue, l1Empty,
                                              op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 3. TILIZE: Before mapping (so mapping operates on final format).
    bool needsTilize =
        !ttcore::isTiled(currentInfo.type) && ttcore::isTiled(targetInfo.type);
    if (needsTilize && currentInfo.hasLayout()) {
      // Tilize with current layout, then mapping change will adjust layout if
      // needed.
      ArrayRef<int64_t> tileShape = ttcore::getTensorTileShape(targetInfo.type);
      auto deviceShape = currentInfo.layout->getDeviceShape(
          currentInfo.getGridShape(), tileShape);
      auto tiledType = RankedTensorType::get(
          deviceShape, targetInfo.type.getElementType(), *currentInfo.layout);
      auto tiledEmpty = createEmpty(tiledType);
      currentValue = lowerFormatConversionGeneric(rewriter, currentValue,
                                                  tiledEmpty, op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 4. MAPPING CHANGE: Grid/index_map/logical_shape/dim_alignments (after
    // tilize). Includes all reblocking (both virtual and normal grids). Must
    // happen in L1 (can't reblock in DRAM). Only when element type formats
    // match (tilize/untilize should happen first).
    if (currentInfo.hasLayout() && targetInfo.hasLayout() &&
        currentInfo.isL1() &&
        (ttcore::isTiled(currentInfo.type) ==
         ttcore::isTiled(targetInfo.type))) {
      // Check if layout properties differ (excluding memSpace and memLayout).
      bool needsMappingChange =
          (currentInfo.getGridShape() != targetInfo.getGridShape() ||
           currentInfo.layout->getIndexAffineMap() !=
               targetInfo.layout->getIndexAffineMap() ||
           currentInfo.layout->getLogicalShape() !=
               targetInfo.layout->getLogicalShape() ||
           currentInfo.layout->getDimAlignments() !=
               targetInfo.layout->getDimAlignments());

      if (needsMappingChange) {
        // Classify the transformation type.
        bool isSimpleReblocking =
            (currentInfo.layout->getLogicalShape() ==
                 targetInfo.layout->getLogicalShape() &&
             currentInfo.layout->getDimAlignments() ==
                 targetInfo.layout->getDimAlignments() &&
             currentInfo.layout->getCollapsedIntervals() ==
                 targetInfo.layout->getCollapsedIntervals());

        bool bothTilized = ttcore::isTiled(currentInfo.type) &&
                           ttcore::isTiled(targetInfo.type);

        if (bothTilized && !isSimpleReblocking) {
          // Complex mapping change on tilized tensors: the affine map approach
          // via logical space doesn't work for unaligned tensors where logical
          // shapes don't divide evenly into tiles. Decompose via scalar space:
          // untilize → map in scalar space → tilize back.

          // 4a. Untilize to scalar space (preserve current layout properties).
          Type scalarType = getScalarType(currentInfo.type.getElementType());
          auto untilizedType = typeBuilder.modifyDeviceType(
              currentInfo.type, *currentInfo.layout, targetGridShape,
              ttcore::MemorySpace::DeviceL1,
              /*newTensorGrid=*/{}, scalarType);
          auto untilizedEmpty = createEmpty(untilizedType);
          currentValue = lowerFormatConversionGeneric(
              rewriter, currentValue, untilizedEmpty, op.getLoc());
          currentInfo = TensorInfo::from(currentValue);

          // 4b. Apply complex mapping change in scalar space.
          // Build scalar target with ALL target's layout properties.
          auto scalarTargetLayout = ttcore::MetalLayoutAttr::get(
              rewriter.getContext(), targetInfo.layout->getLogicalShape(),
              targetInfo.layout->getDimAlignments(),
              targetInfo.layout->getCollapsedIntervals(),
              targetInfo.layout->getOobVal(),
              ttcore::MemorySpace::DeviceL1, // Stay in L1
              targetInfo.layout->getMemoryLayout(),
              targetInfo.layout->getIndexAffineMap());

          auto scalarTargetGridShape = targetInfo.getGridShape();
          auto scalarTargetDeviceShape =
              scalarTargetLayout.getDeviceShape(scalarTargetGridShape, {});

          auto scalarTargetType = RankedTensorType::get(
              scalarTargetDeviceShape, scalarType, scalarTargetLayout);
          auto scalarTargetEmpty = createEmpty(scalarTargetType);
          currentValue =
              lowerMappingChange(rewriter, currentValue, scalarTargetEmpty,
                                 op.getLoc(), targetGridShape);
          currentInfo = TensorInfo::from(currentValue);

          // 4c. Tilize back to match target format.
          ArrayRef<int64_t> tileShape =
              ttcore::getTensorTileShape(targetInfo.type);
          auto tiledDeviceShape = targetInfo.layout->getDeviceShape(
              targetInfo.getGridShape(), tileShape);
          auto tiledType = RankedTensorType::get(
              tiledDeviceShape, targetInfo.type.getElementType(),
              *targetInfo.layout);
          auto tiledEmpty = createEmpty(tiledType);
          currentValue = lowerFormatConversionGeneric(rewriter, currentValue,
                                                      tiledEmpty, op.getLoc());
          currentInfo = TensorInfo::from(currentValue);

        } else {
          // Simple reblocking or untilized complex: use direct approach.
          auto deviceShape = llvm::to_vector(targetInfo.type.getShape());

          // Use target's layout properties (including index_map) but stay in
          // L1.
          auto intermediateLayout = ttcore::MetalLayoutAttr::get(
              rewriter.getContext(), targetInfo.layout->getLogicalShape(),
              targetInfo.layout->getDimAlignments(),
              targetInfo.layout->getCollapsedIntervals(),
              targetInfo.layout->getOobVal(),
              ttcore::MemorySpace::DeviceL1, // Force L1 for reblocking.
              targetInfo.layout->getMemoryLayout(),
              targetInfo.layout->getIndexAffineMap());

          auto intermediateType = RankedTensorType::get(
              deviceShape, currentInfo.type.getElementType(),
              intermediateLayout);

          auto intermediateEmpty = createEmpty(intermediateType);

          currentValue =
              lowerMappingChange(rewriter, currentValue, intermediateEmpty,
                                 op.getLoc(), targetGridShape);
          currentInfo = TensorInfo::from(currentValue);
        }
      }
    }

    // 5. UNTILIZE: Before L1→DRAM or Device→System.
    bool needsUntilize =
        ttcore::isTiled(currentInfo.type) && !ttcore::isTiled(targetInfo.type);
    if (needsUntilize) {
      Type scalarType = getScalarType(currentInfo.type.getElementType());
      auto scalarType_ranked = typeBuilder.modifyDeviceType(
          currentInfo.type, *currentInfo.layout, targetGridShape,
          /*memSpace=*/{}, /*newTensorGrid=*/{}, scalarType,
          /*newTileShape=*/std::nullopt);
      auto scalarEmpty = createEmpty(scalarType_ranked);
      currentValue = lowerFormatConversionGeneric(rewriter, currentValue,
                                                  scalarEmpty, op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 6. L1→DRAM (lowerDatamovementGeneric handles grid mismatch via views).
    if (currentInfo.hasLayout() && !currentInfo.isDRAM() &&
        targetInfo.hasLayout() && targetInfo.isDRAM()) {
      currentValue = lowerDatamovementGeneric(rewriter, currentValue,
                                              op.getOutput(), op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 7. VIRTUAL GRID COLLAPSE: If current has virtual grid but target doesn't
    // need it. This should happen BEFORE any system transfer or whenever grid
    // needs to shrink.
    if (currentInfo.hasLayout()) {
      auto currentGridShape = currentInfo.getGridShape();
      auto targetGridShape_layout =
          targetInfo.hasLayout() ? targetInfo.getGridShape() : targetGridShape;

      // Check if we need to collapse a virtual grid.
      bool needsVirtualGridCollapse =
          (currentGridShape[0] > targetGridShape_layout[0]) ||
          (currentGridShape[1] > targetGridShape_layout[1]);

      if (needsVirtualGridCollapse && currentInfo.isL1()) {
        auto reblocked = typeBuilder.modifyDeviceType(
            currentInfo.type, *currentInfo.layout, targetGridShape,
            ttcore::MemorySpace::DeviceL1,
            /*newTensorGrid=*/{}, /*newElementType=*/{},
            /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/true);
        auto reblockedEmpty = createEmpty(reblocked);
        currentValue =
            lowerMappingChange(rewriter, currentValue, reblockedEmpty,
                               op.getLoc(), targetGridShape);
        currentInfo = TensorInfo::from(currentValue);
      }
    }

    // 8. DEVICE→SYSTEM: Creates final ToLayoutOp with layout attribute.
    if (currentInfo.hasLayout() && !targetInfo.hasLayout()) {
      // Device→system creates a ToLayoutOp with layout attribute set.
      currentValue = lowerSystemLayoutChange(rewriter, currentValue,
                                             op.getOutput(), op.getLoc());
      rewriter.replaceOp(op, currentValue);
      return success();
    }

    // Replace the original ToLayoutOp with the final value.
    rewriter.replaceOp(op, currentValue);
    return success();
  }

  ArrayRef<int64_t> getTargetGridShape() const { return targetGridShape; }

private:
  llvm::SmallVector<int64_t> targetGridShape;
};
} // namespace

namespace {
class D2MLowerToLayout : public impl::D2MLowerToLayoutBase<D2MLowerToLayout> {
public:
  using impl::D2MLowerToLayoutBase<D2MLowerToLayout>::D2MLowerToLayoutBase;

  llvm::SmallVector<int64_t> getTargetGridShape() {
    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    // Use square grid to simplify virtual grid bounce calculations
    llvm::SmallVector<int64_t> targetGridShape =
        d2m::utils::getSquareTargetGrid(getTargetGridShape());

    patterns.add<D2MLowerToLayoutRewriter>(&getContext(), targetGridShape);
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
