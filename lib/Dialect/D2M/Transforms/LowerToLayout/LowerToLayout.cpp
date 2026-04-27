// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/LowerToLayout/Plan.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Utils/AffineMapUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

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
    return fromType(mlir::cast<RankedTensorType>(val.getType()));
  }

  static TensorInfo fromType(RankedTensorType type) {
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

} // namespace

namespace {

// ============================================================================
// Helper functions for building GenericOp regions with RemoteLoad/RemoteStore
// ============================================================================

// Extract the shard type from an operand allocation value (tensor.empty or
// memref.alloc, or CB block arg in old form).
static Type getShardTypeFromCB(Value operandAlloc) {
  return operandAlloc.getType();
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
                                  Value output, Location loc) {
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
              loc, viewOp, output, /*additionalArgs=*/ValueRange(),
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
      auto baseLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(fromTy.getEncoding());
      auto targetLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(toTy.getEncoding());

      AffineMap map;
      if (ttmlir::utils::volume<int64_t>(fromTy.getShape()) ==
          ttmlir::utils::volume<int64_t>(toTy.getShape())) {
        map = ttmlir::utils::calculateReblockMap(fromTy.getShape(),
                                                 toTy.getShape(), ctx);
      } else {
        map = ttcore::utils::buildLayoutTransformMap(baseLayout, fromTy,
                                                     targetLayout, toTy);
      }

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
    ttcore::GridAttr grid;
    if (outputInfo.isDRAM()) {
      viewOutput = buildConcreteView(output, outputInfo.type, inputInfo.type);
      if (auto invMap = utils::getVirtualGridInverseMapping(input)) {
        auto gridShape = llvm::to_vector(inputInfo.getGridShape());

        auto fwdMap = *utils::getVirtualGridForwardMapping(input);
        size_t rank = gridShape.size();
        fwdMap = ttmlir::utils::affineMapDropBackResults(fwdMap, rank);
        for (int i = rank - 1; i >= 0; i--) {
          fwdMap = ttmlir::utils::dropDim(fwdMap, rank + i);
        }
        fwdMap = fwdMap.insertResult(
            getAffineConstantExpr(0, rewriter.getContext()), 0);

        grid = rewriter.getAttr<ttcore::GridAttr>(gridShape, fwdMap, *invMap);
      }
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
                loc, viewInput, viewOutput, /*additionalArgs=*/ValueRange(),
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
                ThreadType::Unified, grid)
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
    assert(TensorInfo::from(input).getGridShape() ==
               TensorInfo::from(output).getGridShape() &&
           "format conversion generic requires matching input/output grids");

    return rewriter
        .create<GenericOp>(
            loc, input, output, /*additionalArgs=*/ValueRange(),
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
        loc, ValueRange(allInputs), ValueRange(allOutputs),
        /*additionalArgs=*/ValueRange(), indexingMapsAttr, iteratorTypesAttr,
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

  // Pull the planning-relevant metadata off a Value: its type plus any
  // remapping / virtual-grid-mapping attached via producing view/empty ops.
  static PlanState extractPlanState(Value v) {
    PlanState state;
    state.type = mlir::cast<RankedTensorType>(v.getType());
    state.remapping = utils::getAssociatedRemapping(v).value_or(AffineMap());
    state.vgmForward =
        utils::getVirtualGridForwardMapping(v).value_or(AffineMap());
    state.vgmInverse =
        utils::getVirtualGridInverseMapping(v).value_or(AffineMap());
    return state;
  }

  // Materialize `plan` as IR, threading `op.getInput()` through each Step's
  // lowering helper. The output buffer of the final Step is `op.getOutput()`
  // whenever its type matches — that's how the ToLayoutOp's destination gets
  // wired without any special-case branching here.
  Value emit(PatternRewriter &rewriter, ToLayoutOp op, const Plan &plan) const {
    Value currentValue = op.getInput();
    Location loc = op.getLoc();

    auto matchesOutputSpec = [&](Value value,
                                 const OutputBufferSpec &spec) -> bool {
      if (value.getType() != spec.type) {
        return false;
      }
      AffineMap currentForward =
          utils::getVirtualGridForwardMapping(value).value_or(AffineMap());
      AffineMap currentInverse =
          utils::getVirtualGridInverseMapping(value).value_or(AffineMap());
      return currentForward == spec.vgmForward &&
             currentInverse == spec.vgmInverse;
    };

    auto createEmpty = [&](const OutputBufferSpec &spec,
                           bool allowReuse = true) -> Value {
      if (allowReuse && matchesOutputSpec(op.getOutput(), spec)) {
        return op.getOutput();
      }
      AffineMapAttr invAttr =
          spec.vgmInverse ? AffineMapAttr::get(spec.vgmInverse) : nullptr;
      AffineMapAttr fwdAttr =
          spec.vgmForward ? AffineMapAttr::get(spec.vgmForward) : nullptr;
      return rewriter.create<d2m::EmptyOp>(loc, spec.type, invAttr, fwdAttr)
          .getResult();
    };

    for (const Step &step : plan) {
      if (const auto *s = std::get_if<HostToDeviceStep>(&step)) {
        currentValue = lowerSystemLayoutChange(rewriter, currentValue,
                                               createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<HostToBounceBufferStep>(&step)) {
        currentValue = lowerSystemLayoutChange(rewriter, currentValue,
                                               createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<DeviceToHostStep>(&step)) {
        currentValue = lowerSystemLayoutChange(
            rewriter, currentValue,
            createEmpty(OutputBufferSpec{s->outputType}), loc);
      } else if (const auto *s = std::get_if<L1ToDRAMStep>(&step)) {
        currentValue = lowerDatamovementGeneric(rewriter, currentValue,
                                                createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<DRAMToL1Step>(&step)) {
        currentValue = lowerDatamovementGeneric(rewriter, currentValue,
                                                createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<TilizeStep>(&step)) {
        currentValue = lowerFormatConversionGeneric(
            rewriter, currentValue, createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<UntilizeStep>(&step)) {
        currentValue = lowerFormatConversionGeneric(
            rewriter, currentValue, createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<RebufferStep>(&step)) {
        currentValue = lowerDatamovementGeneric(rewriter, currentValue,
                                                createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<ReshardStep>(&step)) {
        currentValue = lowerMappingChange(rewriter, currentValue,
                                          createEmpty(s->output), loc);
      } else if (const auto *s = std::get_if<RemapStep>(&step)) {
        currentValue = rewriter
                           .create<ViewLayoutOp>(loc, s->outputType,
                                                 currentValue, s->remapping,
                                                 /*reinterpretLayout=*/false)
                           .getResult();
      } else if (const auto *s = std::get_if<ReinterpretLayoutStep>(&step)) {
        currentValue =
            rewriter
                .create<ViewLayoutOp>(
                    loc, s->outputType, currentValue,
                    rewriter.getMultiDimIdentityMap(
                        mlir::cast<ShapedType>(currentValue.getType())
                            .getRank()),
                    /*reinterpretLayout=*/true)
                .getResult();
      } else if (const auto *s = std::get_if<MaskStep>(&step)) {
        // Mask requires a fresh, non-aliased output buffer: during
        // bufferization, sharing a buffer with the input breaks CB
        // synchronization.
        auto maskedEmpty = createEmpty(s->output, /*allowReuse=*/false);
        currentValue = lowerMaskingGeneric(rewriter, currentValue, maskedEmpty,
                                           loc, s->logicalShape, s->oobVal);
      }
    }
    return currentValue;
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    if (producerMustBeLoweredFirst(op)) {
      return failure();
    }
    PlanState src = extractPlanState(op.getInput());
    PlanState tgt = extractPlanState(op.getOutput());
    Plan plan = minimize(
        canonicalize(src, tgt, targetGridShape, rewriter.getContext()));
    if (plan.empty()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    Value result = emit(rewriter, op, plan);
    rewriter.replaceOp(op, result);
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

    // Use the full device grid; squaring here would mismatch the target grid
    // that d2m-grid-selection picked from and produce unplaceable virtual
    // grids on non-square devices (e.g. Blackhole 10x13).
    llvm::SmallVector<int64_t> targetGridShape = getTargetGridShape();

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
