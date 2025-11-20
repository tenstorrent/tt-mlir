// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
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
struct CompoundComponents {
  bool isMemorySpaceChange = false;
  bool isFormatChange = false;
  bool isMappingChange = false;
  bool isVirtualGridReblock = false;

  bool isCompound() const {
    int count = 0;
    if (isMemorySpaceChange) {
      ++count;
    }
    if (isFormatChange) {
      ++count;
    }
    if (isMappingChange) {
      ++count;
    }
    // Virtual grid reblocks always need special handling via bounce
    if (isVirtualGridReblock) {
      ++count;
    }
    return count > 1;
  }

  static CompoundComponents analyze(Value input, Value output) {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    CompoundComponents result;

    // Check for memory space changes (L1 <-> DRAM).
    result.isMemorySpaceChange =
        (inputInfo.getMemorySpace() != outputInfo.getMemorySpace());

    // Check for format (element type) changes.
    result.isFormatChange =
        (inputInfo.type.getElementType() != outputInfo.type.getElementType());

    // Check for mapping changes, which are transformations that can be
    // expressed as affine maps: tensor shape changes (from collapsed_intervals
    // or dim_alignments causing different padding) and index map changes
    // (logical view transformations). Grid reblocking is excluded here as it
    // was historically classified separately, though it is now also handled as
    // a mapping change.
    if (inputInfo.hasLayout() && outputInfo.hasLayout() &&
        !result.isMemorySpaceChange) {

      auto inputLayout = *inputInfo.layout;
      auto outputLayout = *outputInfo.layout;

      // Shape changes due to format conversions (tile<->scalar) don't count as
      // mapping changes, only shape changes from padding/collapse/alignment.
      bool shapeChanged =
          !result.isFormatChange &&
          (inputInfo.type.getShape() != outputInfo.type.getShape());

      bool indexMapChanged =
          (inputLayout.getIndexAffineMap() != outputLayout.getIndexAffineMap());

      bool gridChanged =
          (inputInfo.getGridShape() != outputInfo.getGridShape());

      result.isMappingChange = (shapeChanged || indexMapChanged || gridChanged);
    }

    // Check for virtual grid reblocking: when input has a virtual grid
    // (non-empty index_map) and either output doesn't have one or has a
    // different grid shape. This always requires an explicit bounce operation.
    if (inputInfo.hasLayout() && outputInfo.hasLayout()) {
      bool inputHasVirtualGrid =
          !inputInfo.layout->getIndexAffineMap().isEmpty();
      bool outputHasVirtualGrid =
          !outputInfo.layout->getIndexAffineMap().isEmpty();
      auto inputGridShape = inputInfo.getGridShape();
      auto outputGridShape = outputInfo.getGridShape();

      result.isVirtualGridReblock =
          inputHasVirtualGrid &&
          (!outputHasVirtualGrid || inputGridShape != outputGridShape);
    }

    return result;
  }
};

} // namespace

namespace {
class D2MLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  D2MLowerToLayoutRewriter(MLIRContext *context,
                           ArrayRef<int64_t> targetGridShape)
      : OpRewritePattern(context, PatternBenefit(2)),
        targetGridShape(targetGridShape) {}

  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  // Lower mapping transformations (grid redistribution, padding changes,
  // collapse changes, index map transformations) to ViewLayoutOp + DMA generic.
  // The ViewLayoutOp represents the transformation as an affine map, and the
  // DMA generic materializes the data movement for L1→L1 transformations.
  static LogicalResult lowerMappingChange(PatternRewriter &rewriter,
                                          ToLayoutOp op) {
    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

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

    // If this is a virtual grid reblock, fail so D2MSplitCompoundLayoutRewriter
    // can create an explicit bounce operation.
    auto components =
        CompoundComponents::analyze(op.getInput(), op.getOutput());
    if (components.isVirtualGridReblock) {
      return failure();
    }

    auto inputLayout = *inputInfo.layout;
    auto outputLayout = *outputInfo.layout;

    // Check if output has virtual grid for later use
    bool outputHasVirtualGrid = !outputLayout.getIndexAffineMap().isEmpty();

    // Build an affine map that transforms input device coordinates to output
    // device coordinates via the shared logical space. This map handles grid
    // redistribution, collapse changes, padding changes, and index map
    // transformations, including rank-changing operations.
    AffineMap viewMap = utils::buildLayoutTransformMap(
        inputLayout, inputInfo.type, outputLayout, outputInfo.type);

    // Embed the transformation map in the output layout.
    auto newLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), outputLayout.getLogicalShape(),
        outputLayout.getDimAlignments(), outputLayout.getCollapsedIntervals(),
        outputLayout.getOobVal(), outputLayout.getMemorySpace(),
        outputLayout.getMemoryLayout(), viewMap);

    auto viewType =
        RankedTensorType::get(outputInfo.type.getShape(),
                              outputInfo.type.getElementType(), newLayout);

    Value viewOp =
        rewriter.create<ViewLayoutOp>(op.getLoc(), viewType, op.getInput(),
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
        ::mlir::ModuleOp moduleOp = op->getParentOfType<::mlir::ModuleOp>();
        mlir::tt::ttcore::DeviceAttr device =
            mlir::tt::ttcore::lookupDevice(moduleOp);
        assert(device && "Device not found");
        auto physicalGridShape = device.getWorkerGrid().getShape();

        bool isVirtualGrid = gridShape[0] > physicalGridShape[0] ||
                             gridShape[1] > physicalGridShape[1];

        if (isVirtualGrid) {
          // Create the virtual grid coordinate maps and use the inverse map
          // for the grid's coordinate translation.
          auto [fwdMap, invMap] = ttmlir::d2m::utils::grids::createCoreVirtMaps(
              rewriter.getContext(), gridShape, physicalGridShape);
          grid =
              ttcore::GridAttr::get(rewriter.getContext(), gridShape, invMap);
        } else {
          // Grid fits within physical bounds, no coordinate translation needed
          grid = ttcore::GridAttr::get(rewriter.getContext(), gridShape);
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

      rewriter.replaceOpWithNewOp<GenericOp>(
          op, viewOp, op.getOutput(),
          [&](OpBuilder &builder, Location loc, ValueRange blockArgs) {
            Value outputCB =
                builder.create<ReserveOp>(loc, blockArgs[1]).getResult();
            auto dma =
                builder.create<d2m::DMAOp>(loc, viewOp, indexingMap, outputCB);
            builder.create<d2m::DMAWaitOp>(loc, dma);
            builder.create<YieldOp>(loc, outputCB);
          },
          ThreadType::Datamovement, grid);
    } else {
      // DRAM operations use the view directly without immediate
      // materialization.
      rewriter.replaceOp(op, viewOp);
    }

    return success();
  }

  static LogicalResult lowerSystemLayoutChange(PatternRewriter &rewriter,
                                               ToLayoutOp op) {
    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

    assert(inputInfo.isSystem() != outputInfo.isSystem() &&
           "one of input or output must be system for now");

    if (op.getLayout()) {
      // Already lowered.
      return failure();
    }

    // Use the layout of whichever side has a layout (input or output).
    auto deviceLayout =
        inputInfo.isSystem() ? outputInfo.layout : inputInfo.layout;
    assert(deviceLayout.has_value() && "Device side must have a layout");

    // If the device side has a virtual grid (non-empty index map), we cannot
    // directly transfer to/from system memory. Fail to trigger a bounce that
    // reblocks the virtual grid to physical first.
    if (!deviceLayout->getIndexAffineMap().isEmpty()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<ToLayoutOp>(op, op.getInput(), op.getOutput(),
                                            *deviceLayout);
    return success();
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

  LogicalResult lowerDatamovementGeneric(PatternRewriter &rewriter,
                                         ToLayoutOp op) const {
    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

    if (inputInfo.isSystem() || outputInfo.isSystem()) {
      return lowerSystemLayoutChange(rewriter, op);
    }

    // Both input and output should have layouts at this point.
    assert(inputInfo.hasLayout() && outputInfo.hasLayout());

    Value viewInput = op.getInput();

    bool isSrcDramOrReblock =
        inputInfo.isDRAM() ||
        (!outputInfo.isDRAM() &&
         (inputInfo.getGridShape() != outputInfo.getGridShape()));

    assert(!(isSrcDramOrReblock && outputInfo.isDRAM()) &&
           "input and output cannot both be remote");

    auto buildConcreteView = [&](Value fromVal, RankedTensorType fromTy,
                                 RankedTensorType toTy) -> Value {
      auto *ctx = rewriter.getContext();
      AffineMap map = mlir::tt::d2m::utils::calculateReblockMap(
          fromTy.getShape(), toTy.getShape(), ctx);
      auto baseLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(fromTy.getEncoding());

      auto enc = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(),
          baseLayout.getMemorySpace(), baseLayout.getMemoryLayout(), map);
      auto resultTy =
          RankedTensorType::get(toTy.getShape(), toTy.getElementType(), enc);
      return rewriter
          .create<ViewLayoutOp>(op.getLoc(), resultTy, fromVal,
                                /*reinterpretLayout=*/false)
          .getResult();
    };

    if (isSrcDramOrReblock) {
      viewInput =
          buildConcreteView(op.getInput(), inputInfo.type, outputInfo.type);
    }

    Value viewOutput = op.getOutput();
    if (outputInfo.isDRAM()) {
      viewOutput =
          buildConcreteView(op.getOutput(), outputInfo.type, inputInfo.type);
    }

    const size_t gridRank = outputInfo.getGridShape().size();

    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, gridRank);
    auto indexingMap = mlir::cast<AffineMapAttr>(indexingMaps[0]);

    rewriter.replaceOpWithNewOp<GenericOp>(
        op, viewInput, viewOutput,
        [&](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          DMAOp dma;
          Value yield;
          if (isSrcDramOrReblock) {
            Value outputCB =
                builder.create<ReserveOp>(loc, blockArgs[1]).getResult();
            dma = builder.create<d2m::DMAOp>(loc, viewInput, indexingMap,
                                             outputCB);
            yield = outputCB;
          } else {
            // Note: Naturally you'd think to use a WaitOp since this is in
            // input cb, but in the layout lowering there is no producer thread.
            // The ReserveOp here effectively unwraps the CB so the DMA can
            // access it.
            Value inputCB =
                builder.create<ReserveOp>(loc, blockArgs[0]).getResult();
            dma = builder.create<d2m::DMAOp>(loc, inputCB, viewOutput,
                                             indexingMap);
            yield = inputCB;
          }
          builder.create<d2m::DMAWaitOp>(loc, dma);
          builder.create<YieldOp>(loc, yield);
        },
        ThreadType::Datamovement);

    return success();
  }

  LogicalResult lowerFormatConversionGeneric(PatternRewriter &rewriter,
                                             ToLayoutOp op) const {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    bool inputTiled = ttcore::isTiled(inputType);
    bool outputTiled = ttcore::isTiled(outputType);
    assert(inputTiled != outputTiled &&
           "one of input or output must be tiled for now");

    rewriter.replaceOpWithNewOp<GenericOp>(
        op, op.getInput(), op.getOutput(),
        [=](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          Value src = builder.create<WaitOp>(loc, blockArgs[0]).getResult();
          Value dst = builder.create<ReserveOp>(loc, blockArgs[1]).getResult();
          if (inputTiled) {
            builder.create<TileUntilizeBlockOp>(loc, src, dst);
          } else {
            builder.create<TileTilizeBlockOp>(loc, src, dst);
          }
          builder.create<YieldOp>(loc, dst);
        },
        ThreadType::Compute);

    return success();
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components =
        CompoundComponents::analyze(op.getInput(), op.getOutput());

    if (components.isCompound()) {
      return failure();
    }

    // By convention, consecutive device->device ToLayout ops must be converted
    // in **producer to consumer order**, such that the consumer ops DO NOT
    // apply a view to an output that will itself be wrapped in a view by the
    // producer op.
    //
    // The GreedyPatternRewriteDriver will handle iterating until all ToLayout
    // ops have been rewritten in producer to consumer order.
    if (producerMustBeLoweredFirst(op)) {
      return failure();
    }

    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

    // Route format changes (tilize/untilize) to lowerFormatConversionGeneric.
    if (components.isFormatChange) {
      return lowerFormatConversionGeneric(rewriter, op);
    }

    // Route all L1→L1 mapping changes to lowerMappingChange, which uses
    // buildLayoutTransformMap to support rank-changing transformations.
    bool isL1toL1MappingChange = false;
    if (inputInfo.hasLayout() && outputInfo.hasLayout() &&
        !inputInfo.isDRAM() && !outputInfo.isDRAM()) {
      auto inputLayout = *inputInfo.layout;
      auto outputLayout = *outputInfo.layout;

      bool shapeChanged =
          (inputInfo.type.getShape() != outputInfo.type.getShape());
      bool indexMapChanged =
          (inputLayout.getIndexAffineMap() != outputLayout.getIndexAffineMap());
      bool gridChanged =
          (inputInfo.getGridShape() != outputInfo.getGridShape());

      isL1toL1MappingChange = (shapeChanged || indexMapChanged || gridChanged);
    }

    if (isL1toL1MappingChange) {
      return lowerMappingChange(rewriter, op);
    }

    // Route memory space changes and grid changes (involving DRAM) to
    // lowerDatamovementGeneric.
    if (components.isMemorySpaceChange || components.isMappingChange) {
      return lowerDatamovementGeneric(rewriter, op);
    }

    llvm_unreachable("ToLayoutOp with no detectable changes");
  }

  ArrayRef<int64_t> getTargetGridShape() const { return targetGridShape; }

private:
  llvm::SmallVector<int64_t> targetGridShape;
};
} // namespace

namespace {
class D2MSplitCompoundLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
  // Helper struct to build intermediate bounce types.
  // This builder will always create a MetalLayoutAttr directly through the
  // primary constructor--it should never make any decisions w.r.t. to grid
  // shape, etc. (those decisions were already made in D2MToD2MGeneric, here
  // we simply decompose ToLayoutOps).
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

    // Create a device tensor type from a system tensor type, using reference
    // layout's characteristics to populate the MetalLayoutAttr appropriately.
    RankedTensorType createDeviceType(RankedTensorType systemType,
                                      ttcore::MetalLayoutAttr referenceLayout,
                                      RankedTensorType referenceType,
                                      ttcore::MemorySpace memSpace,
                                      ArrayRef<int64_t> targetGridShape) {

      // Extract the tensor grid from the reference device tensor.
      SmallVector<int64_t> tensorGridShape =
          llvm::to_vector(referenceLayout.getGridShape(referenceType));
      if (auto metalLayout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
              referenceType.getEncoding());
          metalLayout && !metalLayout.getIndexAffineMap().isEmpty()) {
        tensorGridShape =
            computeVirtualGridBounceShape(tensorGridShape, targetGridShape);
      }

      // Preserve all layout decisions from the referenceType tensor.
      auto layout = ttcore::MetalLayoutAttr::get(
          ctx, referenceLayout.getLogicalShape(),
          referenceLayout.getDimAlignments(),
          referenceLayout.getCollapsedIntervals(), referenceLayout.getOobVal(),
          memSpace, referenceLayout.getMemoryLayout(), AffineMap::get(ctx));

      // Compute the device shape using the referenceType's grid shape.
      ArrayRef<int64_t> tileShape;
      if (ttcore::isTiled(systemType)) {
        tileShape = ttcore::getTensorTileShape(systemType);
      }
      auto deviceShape = layout.getDeviceShape(tensorGridShape, tileShape);

      return RankedTensorType::get(deviceShape, systemType.getElementType(),
                                   layout);
    }

    // Modify an existing device tensor type while preserving layout
    // characteristics.
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

      // For L1→L1 operations (format conversions), preserve virtual grid
      // index_maps since only shard dimensions change, not grid dimensions.
      // The virtual grid's affine map transforms virtual coordinates to
      // physical coordinates and must be preserved through format conversions.
      // For transitions involving DRAM/System, or when reblocking virtual grids
      // to physical grids, clear index_maps (the reblock creates a physical
      // grid).
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
  D2MSplitCompoundLayoutRewriter(MLIRContext *context,
                                 ArrayRef<int64_t> targetGridShape)
      : OpRewritePattern(context, PatternBenefit(2)) {
    this->targetGridShape.assign(targetGridShape.begin(),
                                 targetGridShape.end());
  }

  d2m::ToLayoutOp createToLayoutOp(PatternRewriter &rewriter, Location loc,
                                   Value input,
                                   RankedTensorType desiredType) const {
    // Create empty tensor with desired type and layout.
    auto layout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
        desiredType.getEncoding());
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
    auto components =
        CompoundComponents::analyze(op.getInput(), op.getOutput());

    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

    // Virtual grid reblocks are handled as a special type of compound operation
    // even if they don't involve other changes (format, memory space, etc).
    if (!components.isCompound() && !components.isVirtualGridReblock) {
      return failure();
    }

    BounceTypeBuilder typeBuilder(rewriter.getContext());

    // Handle system <-> device transitions specially.
    if (inputInfo.hasLayout() != outputInfo.hasLayout()) {
      if (!inputInfo.hasLayout()) {
        // System -> Device: move to L1 using output's layout characteristics.
        assert(outputInfo.layout &&
               "Output must have layout for system->device");
        auto bounceType = typeBuilder.createDeviceType(
            inputInfo.type, *outputInfo.layout, outputInfo.type,
            ttcore::MemorySpace::DeviceL1, getTargetGridShape());
        bounce(rewriter, op, bounceType);
      } else {
        // Device -> System: if there's also a format change, do that first.
        // Otherwise handle memory space transition.
        if (components.isFormatChange && ttcore::isTiled(inputInfo.type)) {
          // Bounce through scalar first, preserving L1 and virtual grid.
          Type scalarType = inputInfo.type.getElementType();
          if (auto tileType = mlir::dyn_cast<ttcore::TileType>(scalarType)) {
            scalarType = tileType.getElementType();
          }
          auto bounceType = typeBuilder.modifyDeviceType(
              inputInfo.type, *inputInfo.layout, getTargetGridShape(),
              ttcore::MemorySpace::DeviceL1,
              /*newTensorGrid=*/{}, scalarType,
              /*newTileShape=*/std::nullopt);
          bounce(rewriter, op, bounceType);
        } else {
          // No format change (both scalar or both tiled).
          // Reblock virtual grid to physical before system transfer.
          assert(inputInfo.layout &&
                 "Input must have layout for device->system");
          bool reblockVirtualGridShapes = true;
          auto bounceType = typeBuilder.modifyDeviceType(
              inputInfo.type, *inputInfo.layout, getTargetGridShape(),
              ttcore::MemorySpace::DeviceL1,
              /*newTensorGrid=*/{}, /*newElementType=*/{},
              /*newTileShape=*/{}, reblockVirtualGridShapes);
          bounce(rewriter, op, bounceType);
        }
      }
      return success();
    }

    // If neither has a layout, both are in system memory.
    if (!inputInfo.hasLayout() && !outputInfo.hasLayout()) {
      // Pure host-side operation - should have been handled by
      // compoundComponents
      assert(false && "Host-side only operations should not be compound");
      return failure();
    }

    // Otherwise, if both have layouts, we need to handle device-side
    // transformations.
    assert(inputInfo.layout && outputInfo.layout);

    // Prioritize L1 operations.
    if (!inputInfo.isL1()) {
      // Move input to L1, preserving its grid and layout characteristics.
      auto bounceType = typeBuilder.modifyDeviceType(
          inputInfo.type, *inputInfo.layout, getTargetGridShape(),
          ttcore::MemorySpace::DeviceL1);
      bounce(rewriter, op, bounceType);
    } else if (!outputInfo.isL1()) {
      // Move output to L1, preserving its grid and layout characteristics.
      auto bounceType = typeBuilder.modifyDeviceType(
          outputInfo.type, *outputInfo.layout, getTargetGridShape(),
          ttcore::MemorySpace::DeviceL1);
      bounce(rewriter, op, bounceType);
    } else if (ttcore::isTiled(inputInfo.type) !=
               ttcore::isTiled(outputInfo.type)) {
      // Format conversion
      if (ttcore::isTiled(inputInfo.type)) {
        // Tilized -> scalar: use input's layout/grid, change to scalar element
        // type.
        Type scalarType = inputInfo.type.getElementType();
        if (auto tileType = mlir::dyn_cast<ttcore::TileType>(scalarType)) {
          scalarType = tileType.getElementType();
        }
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, getTargetGridShape(),
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, scalarType,
            /*newTileShape=*/std::nullopt);
        bounce(rewriter, op, bounceType);
      } else {
        // Scalar -> tilized: use input's layout/grid, change element type.
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, getTargetGridShape(),
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, outputInfo.type.getElementType(),
            ttcore::getTensorTileShape(outputInfo.type));
        bounce(rewriter, op, bounceType);
      }
    } else if (components.isMappingChange && ttcore::isTiled(inputInfo.type)) {
      // Mapping change with tiled data - bounce through scalar.
      Type scalarType = inputInfo.type.getElementType();
      if (auto tileType = mlir::dyn_cast<ttcore::TileType>(scalarType)) {
        scalarType = tileType.getElementType();
      }

      auto bounceType = typeBuilder.modifyDeviceType(
          inputInfo.type, *inputInfo.layout, getTargetGridShape(),
          /*memSpace=*/{},
          /*newTensorGrid=*/{}, scalarType,
          /*tileShape=*/std::nullopt);
      bounce(rewriter, op, bounceType);
    } else if (components.isVirtualGridReblock) {
      // Virtual grid reblock: transform virtual grid to physical grid or
      // change virtual grid shape. This clears the index_map.
      auto bounceType = typeBuilder.modifyDeviceType(
          inputInfo.type, *inputInfo.layout, getTargetGridShape(),
          /*memSpace=*/{},
          /*newTensorGrid=*/{}, /*newElementType=*/{},
          /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/true);
      bounce(rewriter, op, bounceType);
    } else if (components.isMappingChange) {
      // Other mapping changes should be handled by lowerMappingChange.
      assert(false && "Unsupported mapping change - should be handled by "
                      "lowerMappingChange");
      return failure();
    } else {
      // Note we should eventually support DRAM <-> DRAM, or System <-> System
      // w/ format conversion via streaming supported.
      assert(false && "Unsupported compound layout change");
      return failure();
    }

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

    patterns.add<D2MSplitCompoundLayoutRewriter, D2MLowerToLayoutRewriter>(
        &getContext(), targetGridShape);
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
