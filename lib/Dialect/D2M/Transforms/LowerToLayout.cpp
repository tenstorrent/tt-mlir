// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Utils/AffineMapUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// Check if a layout requires masking due to non-trivial OOBVal and padding.
static bool needsMasking(ttcore::MetalLayoutAttr layout,
                         RankedTensorType tensorType) {
  // Only mask if OOBVal is not Undef
  if (layout.getOobVal() == ttcore::OOBVal::Undef) {
    return false;
  }

  // Check if tensor is tiled - masking only applies to tiled tensors
  if (!ttcore::isTiled(tensorType)) {
    return false;
  }

  // Check if padding exists by comparing logical shape to aligned shape.
  // If any logical dimension doesn't match its aligned size, there's padding.
  ArrayRef<int64_t> logicalShape = layout.getLogicalShape();
  ArrayRef<int64_t> dimAlignments = layout.getDimAlignments();

  for (size_t i = 0; i < logicalShape.size(); ++i) {
    int64_t aligned = ttmlir::utils::alignUp(logicalShape[i], dimAlignments[i]);
    if (aligned != logicalShape[i]) {
      // Padding found, masking is needed.
      return true;
    }
  }

  return false;
}

} // namespace

namespace {

// ============================================================================
// Helper functions for building GenericOp regions with RemoteLoad/RemoteStore
// ============================================================================

// Extract the underlying shard type from a circular buffer block argument
static Type getShardTypeFromCB(Value cbBlockArg) {
  auto cbType = mlir::cast<CBType>(cbBlockArg.getType());
  return cbType.getUnderlying();
}

// Build identity grid indices for a given grid rank
static SmallVector<Value>
buildIdentityGridIndices(OpBuilder &builder, Location loc, size_t gridRank) {
  AffineMap indexingMap = builder.getMultiDimIdentityMap(gridRank);
  return d2m::utils::buildGridIndices(builder, loc, indexingMap);
}

// Create a RemoteLoadOp in implicit form (returns loaded memref directly)
static Value createRemoteLoad(OpBuilder &builder, Location loc, Type shardType,
                              Value source, ArrayRef<Value> indices) {
  // Create a buffer for the load result
  auto tensorType = mlir::cast<RankedTensorType>(shardType);
  auto bufferOp = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
  Value buffer = bufferOp.getResult();
  return builder.create<RemoteLoadOp>(loc, shardType, buffer, source, indices)
      .getResult();
}

// Create a tensor.empty with identical result type
static Value createTensorEmpty(OpBuilder &builder, Location loc,
                               Type shardType) {
  auto tensorType = mlir::cast<RankedTensorType>(shardType);
  return builder
      .create<tensor::EmptyOp>(loc, tensorType.getShape(),
                               tensorType.getElementType())
      .getResult();
}

// Create a RemoteStoreOp in implicit form and return the result
static Value createRemoteStore(OpBuilder &builder, Location loc,
                               Value destination, ArrayRef<Value> indices,
                               Value localBuffer) {
  return builder
      .create<RemoteStoreOp>(loc, destination.getType(), destination, indices,
                             localBuffer)
      .getResult();
}

// Complete identity load-store pattern: load from input, acquire output buffer,
// and return both along with the indices. This is useful for operations that
// need to perform transformations between load and store (e.g., tilize, mask).
struct IdentityLoadStoreResult {
  Value src;
  Value dst;
  SmallVector<Value> indices;
};

static IdentityLoadStoreResult
buildIdentityLoadStore(OpBuilder &builder, Location loc, Value inputCBBlockArg,
                       Value outputCBBlockArg, Value input, Value output,
                       int64_t outputOperandIndex) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto inputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
  size_t gridRank = inputLayout.getGridShape(inputType).size();

  Type inputShardType = getShardTypeFromCB(inputCBBlockArg);
  Type outputShardType = getShardTypeFromCB(outputCBBlockArg);
  SmallVector<Value> indices = buildIdentityGridIndices(builder, loc, gridRank);

  Value src = createRemoteLoad(builder, loc, inputShardType, input, indices);
  Value dst = createTensorEmpty(builder, loc, outputShardType);

  return {src, dst, indices};
}

class D2MLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
  // Helper struct to build intermediate bounce types.
  class BounceTypeBuilder {
  public:
    explicit BounceTypeBuilder(MLIRContext *ctx) : ctx(ctx) {}

    // Computes a workable bounce shape grid for a virtual grid.
    llvm::SmallVector<int64_t>
    computeVirtualGridBounceShape(ArrayRef<int64_t> virtualGridShape,
                                  ArrayRef<int64_t> deviceGridShape) const {

      TT_assert(virtualGridShape.size() >= 2u);
      // Collapse all leading dimensions into the first dimension of a 2D shape.
      llvm::SmallVector<int64_t> collapsedVirtualGridShape(2);
      collapsedVirtualGridShape[0] = virtualGridShape[0];
      for (int64_t i = 1; i < static_cast<int64_t>(virtualGridShape.size()) - 1;
           ++i) {
        collapsedVirtualGridShape[0] *= virtualGridShape[i];
      }
      collapsedVirtualGridShape[1] = virtualGridShape.back();

      llvm::SmallVector<int64_t> bounceShape;
      for (size_t i = 0; i < collapsedVirtualGridShape.size(); i++) {
        auto dim =
            (collapsedVirtualGridShape[i] > deviceGridShape[i])
                ? std::gcd(collapsedVirtualGridShape[i], deviceGridShape[i])
                : collapsedVirtualGridShape[i];
        bounceShape.push_back(dim);
      }
      TT_assert(bounceShape.size() == 2u);

      return bounceShape;
    }

    // Creates conventional ND -> 2D collapsed intervals and dim alignments for
    // a given reference layout and target grid shape. This pads out the dim
    // alignments to work well with the physical device grid and be consistent
    // with alignments used by the D2MGridSelection pass.
    std::pair<DenseIntElementsAttr, llvm::SmallVector<int64_t>>
    computeGridAwareCollapsedIntervalsAndDimAlignments(
        ttcore::MetalLayoutAttr referenceLayout,
        ArrayRef<int64_t> targetGridShape) {
      auto logicalShape = referenceLayout.getLogicalShape();
      auto collapsedIntervals =
          referenceLayout.computeDefaultCollapsedIntervals(ctx,
                                                           logicalShape.size());
      auto dimAlignments =
          ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
              logicalShape, targetGridShape,
              ttcore::MetalLayoutAttr::normalizeAndFlattenIntervals(
                  collapsedIntervals, logicalShape.size()));
      return {collapsedIntervals, dimAlignments};
    }

    // Create a device tensor type from a system tensor type.
    // For virtual grids (ND or exceeding device bounds), the data must bounce
    // through DRAM interleaved because the host cannot directly scatter data
    // across an ND/virtual L1 grid.  The subsequent DRAM→L1 step
    // (lowerDatamovementGeneric) handles the actual scatter via a view/reblock
    // map.
    RankedTensorType createDeviceType(RankedTensorType systemType,
                                      ttcore::MetalLayoutAttr referenceLayout,
                                      RankedTensorType referenceType,
                                      ArrayRef<int64_t> targetGridShape) {
      SmallVector<int64_t> tensorGridShape =
          llvm::to_vector(referenceLayout.getGridShape(referenceType));

      bool virtualBounceNeeded = ttmlir::d2m::utils::grids::requiresVirtualGrid(
          tensorGridShape, targetGridShape);

      ttcore::MetalLayoutAttr layout;
      if (virtualBounceNeeded) {
        // Virtual grids need to bounce through DRAM — the bounce shape
        // should be collapsed to 2D for the unit-grid DRAM buffer.
        tensorGridShape =
            computeVirtualGridBounceShape(tensorGridShape, targetGridShape);

        auto [collapsedIntervals, dimAlignments] =
            computeGridAwareCollapsedIntervalsAndDimAlignments(referenceLayout,
                                                               targetGridShape);
        // Bounce virtual grids through interleaved DRAM on the unit grid.
        tensorGridShape.assign(targetGridShape.size(), 1);

        // Keep old dimAlignments but use new collapsedIntervals to collapse the
        // DRAM tensor to 2D.
        TT_assert(collapsedIntervals.getType().getDimSize(0) == 2);
        layout = ttcore::MetalLayoutAttr::get(
            ctx, referenceLayout.getLogicalShape(),
            referenceLayout.getDimAlignments(), collapsedIntervals,
            referenceLayout.getOobVal(), ttcore::MemorySpace::DeviceDRAM,
            ttcore::TensorMemoryLayout::Interleaved);
      } else {
        layout = ttcore::MetalLayoutAttr::get(
            ctx, referenceLayout.getLogicalShape(),
            referenceLayout.getDimAlignments(),
            referenceLayout.getCollapsedIntervals(),
            referenceLayout.getOobVal(), ttcore::MemorySpace::DeviceL1,
            referenceLayout.getMemoryLayout());
      }

      ArrayRef<int64_t> tileShape;
      if (ttcore::isTiled(systemType)) {
        tileShape = ttcore::getTensorTileShape(systemType);
      }
      auto deviceShape = layout.getDeviceShape(tensorGridShape, tileShape);

      return RankedTensorType::get(deviceShape, systemType.getElementType(),
                                   layout);
    }

    // Modify an existing device tensor type.
    // Note: Index maps are now stored on view_layout ops, not on the layout
    // attribute. The existingRemapping parameter captures any remapping that
    // was associated with the base tensor.
    RankedTensorType
    modifyDeviceType(RankedTensorType baseType,
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

      // Do not consider a tensor with an identity remapping as having a virtual
      // grid.
      bool hasVirtualGrid = existingRemapping && !existingRemapping.isEmpty() &&
                            !existingRemapping.isIdentity();
      SmallVector<int64_t> tensorGrid;
      // An ND grid (>2D) also needs reblocking to a valid 2D physical grid,
      // regardless of whether there is an explicit remapping. This handles
      // the case where a stream/view remapping was already consumed by a
      // prior generic op, leaving a materialized ND tensor that still needs
      // to be collapsed before host transfer.
      bool needsReblock = hasVirtualGrid;
      if (newTensorGrid.has_value()) {
        tensorGrid.assign(newTensorGrid->begin(), newTensorGrid->end());
      } else {
        auto currentGrid = llvm::to_vector(baseLayout.getGridShape(baseType));
        tensorGrid = currentGrid;
        needsReblock =
            needsReblock || ttmlir::d2m::utils::grids::requiresVirtualGrid(
                                tensorGrid, targetGridShape);
        if (needsReblock && reblockVirtualGridShapes) {
          tensorGrid =
              computeVirtualGridBounceShape(tensorGrid, targetGridShape);
        }
      }

      ttcore::MetalLayoutAttr layout;
      if (needsReblock && reblockVirtualGridShapes) {
        // Recompute default collapsed intervals and dim alignments if virtual
        // grid shape is being reblocked.
        auto [collapsedIntervals, dimAlignments] =
            computeGridAwareCollapsedIntervalsAndDimAlignments(baseLayout,
                                                               targetGridShape);
        layout = ttcore::MetalLayoutAttr::get(ctx, baseLayout.getLogicalShape(),
                                              dimAlignments, collapsedIntervals,
                                              baseLayout.getOobVal(), memSpace,
                                              baseLayout.getMemoryLayout());
      } else {
        // Otherwise, preserve dim alignments and collapsed intervals.
        layout = ttcore::MetalLayoutAttr::get(
            ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
            baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(),
            memSpace, baseLayout.getMemoryLayout());
      }

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

    auto newLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), outputLayout.getLogicalShape(),
        outputLayout.getDimAlignments(), outputLayout.getCollapsedIntervals(),
        outputLayout.getOobVal(), outputLayout.getMemorySpace(),
        outputLayout.getMemoryLayout());

    auto viewType =
        RankedTensorType::get(outputInfo.type.getShape(),
                              outputInfo.type.getElementType(), newLayout);

    // Pass the transformation map via the remapping attribute.
    Value viewOp = rewriter.create<ViewLayoutOp>(loc, viewType, input, viewMap,
                                                 /*reinterpretLayout=*/false);

    // Materialize L1→L1 transformations with a DMA generic that performs the
    // actual data movement according to the view's affine map.
    if (!inputInfo.isDRAM() && !outputInfo.isDRAM()) {
      auto gridShape = outputInfo.getGridShape();
      const size_t gridRank = gridShape.size();

      // Build identity indexing maps for the generic operation. The view's
      // affine map handles all address transformations.
      ArrayAttr indexingMaps, iteratorTypes;
      std::tie(indexingMaps, iteratorTypes) =
          GenericOp::buildParallelAffineMapsAndIteratorTypes(
              rewriter, /*arity=*/2, gridRank);
      auto indexingMapAttr = mlir::cast<AffineMapAttr>(indexingMaps[0]);
      AffineMap indexingMap = indexingMapAttr.getValue();

      return rewriter
          .create<GenericOp>(
              loc, viewOp, output,
              [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
                // Load from input, store to output (load+store pair for proper
                // CB association)
                Type inputShardType = getShardTypeFromCB(blockArgs[0]);
                SmallVector<Value> indices = d2m::utils::buildGridIndices(
                    builder, innerLoc, indexingMap);

                // Load-store idiom
                Value loadedData = createRemoteLoad(
                    builder, innerLoc, inputShardType, viewOp, indices);
                Value storeResult = createRemoteStore(builder, innerLoc, output,
                                                      indices, loadedData);
                builder.create<YieldOp>(innerLoc, storeResult);
              },
              ThreadType::Unified)
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

    // Emit dedicated host transfer ops based on direction.
    if (inputInfo.isSystem()) {
      // Host → Device: use ToDeviceOp.
      return rewriter.create<ToDeviceOp>(loc, input, output, *deviceLayout)
          .getResult(0);
    }
    // Device → Host: use ToHostOp.
    return rewriter.create<ToHostOp>(loc, input, output, *deviceLayout)
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
          baseLayout.getMemorySpace(), baseLayout.getMemoryLayout());
      auto resultTy =
          RankedTensorType::get(toTy.getShape(), toTy.getElementType(), enc);
      return rewriter
          .create<ViewLayoutOp>(loc, resultTy, fromVal, map,
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
    auto indexingMapAttr = mlir::cast<AffineMapAttr>(indexingMaps[0]);
    AffineMap indexingMap = indexingMapAttr.getValue();

    auto result =
        rewriter
            .create<GenericOp>(
                loc, viewInput, viewOutput,
                [&](OpBuilder &builder, Location innerLoc,
                    ValueRange blockArgs) {
                  Type inputShardType = getShardTypeFromCB(blockArgs[0]);
                  SmallVector<Value> indices = d2m::utils::buildGridIndices(
                      builder, innerLoc, indexingMap);

                  // Use load+store idiom for proper CB association
                  Value loadedData = createRemoteLoad(
                      builder, innerLoc, inputShardType, viewInput, indices);
                  Value storeResult = createRemoteStore(
                      builder, innerLoc, viewOutput, indices, loadedData);
                  builder.create<YieldOp>(innerLoc, storeResult);
                },
                ThreadType::Unified)
            .getResult(0);
    return result;
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
              auto [src, dst, indices] =
                  buildIdentityLoadStore(builder, innerLoc, blockArgs[0],
                                         blockArgs[1], input, output, 1);

              Value result;
              if (inputTiled) {
                result = builder
                             .create<TileUntilizeBlockOp>(
                                 innerLoc, dst.getType(), src, dst)
                             .getResult();
              } else {
                result = builder
                             .create<TileTilizeBlockOp>(innerLoc, dst.getType(),
                                                        src, dst)
                             .getResult();
              }

              Value storeResult =
                  createRemoteStore(builder, innerLoc, output, indices, result);
              builder.create<YieldOp>(innerLoc, storeResult);
            },
            ThreadType::Unified)
        .getResult(0);
  }

  // Lower masking operation using a d2m.generic with BlockMaskOp.
  // The BlockMaskOp operates at block level and gets decomposed later.
  //
  // Strategy: Use CB-based mask generation (L1 writes + copy_tile).
  // This is more reliable than SFPU-based mask generation which has
  // complex face iteration pattern, at cost of extra memory usage.
  Value lowerMaskingGeneric(PatternRewriter &rewriter, Value input,
                            Value output, Location loc,
                            ArrayRef<int64_t> logicalShape,
                            ttcore::OOBVal fillValue) const {
    // Extract the last two dimensions as the logical rows/cols for masking.
    int64_t logicalRows = logicalShape[logicalShape.size() - 2];
    int64_t logicalCols = logicalShape[logicalShape.size() - 1];

    // Check if partial masking is needed (non-tile-aligned shape).
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto inputLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
    // shardRank is the shard shape rank (used for indexing maps).
    const size_t shardRank = inputLayout.getShardShape(inputType).size();

    // Create scratch mask tensors (single tile each).
    // These are used as scratch CBs to write masks via L1, then copy to DST.
    // The mask tensor must have same rank as input tensor for GenericOp to
    // work. Use the input's logical shape but with 1s except last two dims
    // (32x32).
    auto inputLogicalShape = inputLayout.getLogicalShape();
    SmallVector<int64_t> maskLogicalShape(inputLogicalShape.begin(),
                                          inputLogicalShape.end());
    // Set all dims to 1 except last two which are 32x32 (single tile).
    for (size_t i = 0; i < maskLogicalShape.size(); ++i) {
      if (i < maskLogicalShape.size() - 2) {
        maskLogicalShape[i] = 1;
      } else {
        maskLogicalShape[i] = 32;
      }
    }
    // Create mask layout using the input's collapsed intervals so the mask
    // tensor has the same grid rank as the input tensor, regardless of
    // collapse.
    auto inputNormalizedIntervals = inputLayout.getNormalizedIntervals();
    auto maskDimAlignments = ttcore::MetalLayoutAttr::computeTileAlignments(
        maskLogicalShape, inputNormalizedIntervals);
    auto maskLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), maskLogicalShape, maskDimAlignments,
        inputLayout.getCollapsedIntervals(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceL1, ttcore::TensorMemoryLayout::Sharded);

    auto elemType = inputType.getElementType();
    // Mask is a single tile (broadcast via constant indexing maps).
    // Use a unit grid with the same rank as the input grid.
    auto gridShape = inputLayout.getGridShape(inputType);
    SmallVector<int64_t> unitGrid(gridShape.size(), 1);
    auto tileShape = ttcore::getTensorTileShape(inputType);
    auto maskShape = maskLayout.getDeviceShape(unitGrid, tileShape);

    Value rowMaskTensor =
        rewriter.create<d2m::EmptyOp>(loc, maskShape, elemType, maskLayout)
            .getResult();
    Value colMaskTensor =
        rewriter.create<d2m::EmptyOp>(loc, maskShape, elemType, maskLayout)
            .getResult();

    // Input list includes scratch mask CBs.
    SmallVector<Value> allInputs = {input, rowMaskTensor, colMaskTensor};
    SmallVector<Value> allOutputs = {output};

    // Build indexing maps based on shard rank (iteration space).
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(shardRank);
    // For mask operands: broadcast (constant 0 for each grid/shard dim).
    SmallVector<AffineExpr> zeroExprs(shardRank,
                                      rewriter.getAffineConstantExpr(0));
    AffineMap constantMap =
        AffineMap::get(shardRank, 0, zeroExprs, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {
        identityMap, // input: iterate over all tiles.
        constantMap, // rowMask: single tile, constant.
        constantMap, // colMask: single tile, constant.
        identityMap  // output: iterate over all tiles.
    };
    Attribute parallel = rewriter.getAttr<ttcore::IteratorTypeAttr>(
        ttcore::IteratorType::Parallel);
    ArrayAttr indexingMapsAttr = rewriter.getAffineMapArrayAttr(indexingMaps);
    ArrayAttr iteratorTypesAttr =
        rewriter.getArrayAttr(SmallVector<Attribute>(shardRank, parallel));

    auto genericOp = rewriter.create<GenericOp>(
        loc, ValueRange(allInputs), ValueRange(allOutputs), indexingMapsAttr,
        iteratorTypesAttr,
        [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
          // blockArgs: [inputCB, rowMaskCB, colMaskCB, outputCB].
          Type inputShardType = getShardTypeFromCB(blockArgs[0]);
          Type outputShardType = getShardTypeFromCB(blockArgs[3]);

          size_t gridRank = gridShape.size();
          SmallVector<Value> indices =
              buildIdentityGridIndices(builder, innerLoc, gridRank);

          // Load input data.
          Value src = createRemoteLoad(builder, innerLoc, inputShardType, input,
                                       indices);

          // Load mask data from scratch CBs using RemoteLoad. This establishes
          // the connection between local buffers and the CBs. The masks use
          // constant zero indices (broadcast - single tile shared across grid).
          Type rowMaskType = getShardTypeFromCB(blockArgs[1]);
          Type colMaskType = getShardTypeFromCB(blockArgs[2]);
          SmallVector<Value> zeroIndices(
              gridRank, builder.create<arith::ConstantIndexOp>(innerLoc, 0));
          Value rowMaskLocal = createRemoteLoad(builder, innerLoc, rowMaskType,
                                                rowMaskTensor, zeroIndices);
          Value colMaskLocal = createRemoteLoad(builder, innerLoc, colMaskType,
                                                colMaskTensor, zeroIndices);

          // Create output buffer.
          Value dst = createTensorEmpty(builder, innerLoc, outputShardType);

          Value logicalRowsVal =
              builder.create<arith::ConstantIndexOp>(innerLoc, logicalRows);
          Value logicalColsVal =
              builder.create<arith::ConstantIndexOp>(innerLoc, logicalCols);

          // BlockMaskOp with mask tensors - the mask writes will be handled
          // in DecomposeMasking, which runs after bufferization.
          Value masked = builder
                             .create<BlockMaskOp>(innerLoc, dst.getType(), src,
                                                  dst, rowMaskLocal,
                                                  colMaskLocal, logicalRowsVal,
                                                  logicalColsVal, fillValue)
                             .getResult();

          // Store the masked result to output.
          Value storeResult =
              createRemoteStore(builder, innerLoc, output, indices, masked);
          builder.create<YieldOp>(innerLoc, storeResult);
        },
        ThreadType::Unified);

    // Mark mask inputs (indices 1, 2) as scratch - they don't need streaming.
    genericOp.setScratchInputsAttr(rewriter.getDenseI64ArrayAttr({1, 2}));

    return genericOp.getResult(0);
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
    // Use producer-first ordering to ensure dependencies are lowered first.
    if (producerMustBeLoweredFirst(op)) {
      return failure();
    }

    auto targetInfo = TensorInfo::from(op.getOutput());
    auto currentInfo = TensorInfo::from(op.getInput());
    Value currentValue = op.getInput();

    BounceTypeBuilder typeBuilder(rewriter.getContext());

    // === TRANSFORMATION PIPELINE ===
    // Apply transformations in priority order.
    // Each step emits lowered ops and updates currentValue/currentInfo.

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

    // 1. SYSTEM→DEVICE: Transfer to L1/DRAM with same element type as input.
    if (!currentInfo.hasLayout() && targetInfo.hasLayout()) {
      // System transfer can ONLY change memory space, not element type.
      // Create intermediate with scalar element type (same as system input).
      Type scalarElemType = getScalarType(currentInfo.type.getElementType());
      auto newType =
          typeBuilder.createDeviceType(currentInfo.type, *targetInfo.layout,
                                       targetInfo.type, targetGridShape);

      // Force scalar element type for the L1/DRAM intermediate.
      auto newLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(newType.getEncoding());
      auto scalarNewType =
          RankedTensorType::get(newType.getShape(), scalarElemType, newLayout);

      auto newEmpty = createEmpty(scalarNewType);
      currentValue = lowerSystemLayoutChange(rewriter, currentValue, newEmpty,
                                             op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 2. DRAM→L1: Must happen before other device ops.
    // Use target's layout characteristics.
    if (currentInfo.hasLayout() && currentInfo.isDRAM() &&
        targetInfo.hasLayout() && !targetInfo.isDRAM()) {
      // Use target's layout but force L1 and preserve current's grid shape
      // unless we are copying from an interleaved DRAM tensor on a unit grid.
      const bool isDRAMInterleaved = currentInfo.layout->getMemoryLayout() ==
                                     ttcore::TensorMemoryLayout::Interleaved;
      auto bounceGrid =
          llvm::to_vector(isDRAMInterleaved ? targetInfo.getGridShape()
                                            : currentInfo.getGridShape());
      // No existing remapping for DRAM→L1 transfer.
      auto l1Type = typeBuilder.modifyDeviceType(
          targetInfo.type, *targetInfo.layout, targetGridShape, AffineMap(),
          ttcore::MemorySpace::DeviceL1, bounceGrid,
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

    // 4. MASKING: Apply boundary masking after tilization if needed.
    // Insert TileMaskBoundaryOp when the target layout has non-Undef OOBVal
    // and padding exists.
    if (currentInfo.hasLayout() && ttcore::isTiled(currentInfo.type) &&
        needsMasking(*currentInfo.layout, currentInfo.type)) {
      // Create a NEW output buffer for masking - must NOT be aliased with input
      // during bufferization, otherwise the CB synchronization will fail.
      // Always create a fresh EmptyOp rather than potentially reusing existing
      // buffers via createEmpty().
      auto layout = mlir::dyn_cast<ttcore::MetalLayoutAttr>(
          currentInfo.type.getEncoding());
      auto maskedEmpty =
          rewriter
              .create<d2m::EmptyOp>(op.getLoc(), currentInfo.type.getShape(),
                                    currentInfo.type.getElementType(), layout)
              .getResult();
      currentValue =
          lowerMaskingGeneric(rewriter, currentValue, maskedEmpty, op.getLoc(),
                              currentInfo.layout->getLogicalShape(),
                              currentInfo.layout->getOobVal());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 5. MAPPING CHANGE: Grid/index_map/logical_shape/dim_alignments (after
    // tilize). Includes all reblocking (both virtual and normal grids). Must
    // happen in L1 (can't reblock in DRAM). Only when element type formats
    // match (tilize/untilize should happen first).
    if (currentInfo.hasLayout() && targetInfo.hasLayout() &&
        currentInfo.isL1() &&
        (ttcore::isTiled(currentInfo.type) ==
         ttcore::isTiled(targetInfo.type))) {
      // Check if layout properties differ (excluding memSpace and memLayout).
      // Compare remappings via their defining view/stream ops.
      auto currentRemapping = utils::getAssociatedRemapping(currentValue);
      auto targetRemapping = utils::getAssociatedRemapping(op.getOutput());
      bool remappingsDiffer = currentRemapping != targetRemapping;

      // Compare virtualGridMappings: different TTNN shard strategies (e.g.
      // height_sharded vs block_sharded) can produce identical MetalLayoutAttr
      // types but still require a mapping change to preserve the shard strategy
      // through the pipeline.
      auto currentVGM = utils::getVirtualGridMapping(currentValue);
      auto targetVGM = utils::getVirtualGridMapping(op.getOutput());
      bool vgmsDiffer = currentVGM != targetVGM;

      bool needsMappingChange =
          (currentInfo.getGridShape() != targetInfo.getGridShape() ||
           remappingsDiffer || vgmsDiffer ||
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

          // 5a. Untilize to scalar space (preserve current layout properties).
          // Reblock virtual grid shape here to align with earlier splitting
          // phases that use reblocked intermediates to bounce virtual grid
          // shapes from host to device.
          Type scalarType = getScalarType(currentInfo.type.getElementType());
          auto untilizedType = typeBuilder.modifyDeviceType(
              currentInfo.type, *currentInfo.layout, targetGridShape,
              currentRemapping.value_or(AffineMap()),
              ttcore::MemorySpace::DeviceL1,
              /*newTensorGrid=*/{}, scalarType,
              /*newTileShape=*/{}, /* reblockVirtualGridShapes */ true);
          auto untilizedEmpty = createEmpty(untilizedType);
          currentValue = lowerFormatConversionGeneric(
              rewriter, currentValue, untilizedEmpty, op.getLoc());
          currentInfo = TensorInfo::from(currentValue);

          // 5b. Apply complex mapping change in scalar space.
          // Build scalar target with ALL target's layout properties.
          auto scalarTargetLayout = ttcore::MetalLayoutAttr::get(
              rewriter.getContext(), targetInfo.layout->getLogicalShape(),
              targetInfo.layout->getDimAlignments(),
              targetInfo.layout->getCollapsedIntervals(),
              targetInfo.layout->getOobVal(),
              ttcore::MemorySpace::DeviceL1, // Stay in L1
              targetInfo.layout->getMemoryLayout());

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

          // 5c. Tilize back to match target format.
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

          // Use target's layout properties but stay in L1.
          auto intermediateLayout = ttcore::MetalLayoutAttr::get(
              rewriter.getContext(), targetInfo.layout->getLogicalShape(),
              targetInfo.layout->getDimAlignments(),
              targetInfo.layout->getCollapsedIntervals(),
              targetInfo.layout->getOobVal(),
              ttcore::MemorySpace::DeviceL1, // Force L1 for reblocking.
              targetInfo.layout->getMemoryLayout());

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

    // 6. UNTILIZE: Before L1→DRAM or Device→System.
    bool needsUntilize =
        ttcore::isTiled(currentInfo.type) && !ttcore::isTiled(targetInfo.type);
    if (needsUntilize) {
      Type scalarType = getScalarType(currentInfo.type.getElementType());
      // Avoid reblocking virtual grid shapes here. Output type here retains
      // input's virtual grid shape; only transformation is to scalar dtype.
      auto existingRemapping =
          utils::getAssociatedRemapping(currentValue).value_or(AffineMap());
      auto scalarType_ranked = typeBuilder.modifyDeviceType(
          currentInfo.type, *currentInfo.layout, targetGridShape,
          existingRemapping, /*memSpace=*/{}, /*newTensorGrid=*/{}, scalarType,
          /*newTileShape=*/std::nullopt, /*reblockVirtualGridShapes=*/false);
      auto scalarEmpty = createEmpty(scalarType_ranked);
      currentValue = lowerFormatConversionGeneric(rewriter, currentValue,
                                                  scalarEmpty, op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 7. L1→DRAM (lowerDatamovementGeneric handles grid mismatch via views).
    if (currentInfo.hasLayout() && !currentInfo.isDRAM() &&
        targetInfo.hasLayout() && targetInfo.isDRAM()) {
      currentValue = lowerDatamovementGeneric(rewriter, currentValue,
                                              op.getOutput(), op.getLoc());
      currentInfo = TensorInfo::from(currentValue);
    }

    // 8. VIRTUAL GRID COLLAPSE: If current has virtual grid but target doesn't
    // need it. This should happen BEFORE any system transfer or whenever grid
    // needs to shrink.
    if (currentInfo.hasLayout() && targetInfo.isSystem()) {
      auto currentGridShape = currentInfo.getGridShape();
      auto targetGridShape_layout =
          targetInfo.hasLayout() ? targetInfo.getGridShape() : targetGridShape;

      // Check if we need to collapse a virtual grid.
      bool needsVirtualGridCollapse =
          ttmlir::d2m::utils::grids::requiresVirtualGrid(
              currentGridShape, targetGridShape_layout);

      if (needsVirtualGridCollapse && currentInfo.isL1()) {
        auto existingRemapping =
            utils::getAssociatedRemapping(currentValue).value_or(AffineMap());
        auto reblocked = typeBuilder.modifyDeviceType(
            currentInfo.type, *currentInfo.layout, targetGridShape,
            existingRemapping, ttcore::MemorySpace::DeviceL1,
            /*newTensorGrid=*/{}, /*newElementType=*/{},
            /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/true);
        auto reblockedEmpty = createEmpty(reblocked);
        currentValue =
            lowerMappingChange(rewriter, currentValue, reblockedEmpty,
                               op.getLoc(), targetGridShape);
        currentInfo = TensorInfo::from(currentValue);
      }
    }

    // 9. DEVICE→SYSTEM: Creates final ToLayoutOp with layout attribute.
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
