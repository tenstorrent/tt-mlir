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
// Helper to extract scalar type from potentially tiled type
static Type getScalarType(Type type) {
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(type)) {
    return tileType.getElementType();
  }
  return type;
}

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

    auto inputLayout = *inputInfo.layout;
    auto outputLayout = *outputInfo.layout;

    // Check if output has virtual grid for later use
    bool outputHasVirtualGrid = !outputLayout.getIndexAffineMap().isEmpty();

    // When building the data transformation map, we need to exclude virtual
    // grid index_maps because those are for grid coordinate translation, not
    // data layout transformation. Create temporary layouts without index_maps.
    auto inputLayoutForTransform = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), inputLayout.getLogicalShape(),
        inputLayout.getDimAlignments(), inputLayout.getCollapsedIntervals(),
        inputLayout.getOobVal(), inputLayout.getMemorySpace(),
        inputLayout.getMemoryLayout(), AffineMap::get(rewriter.getContext()));

    auto outputLayoutForTransform = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), outputLayout.getLogicalShape(),
        outputLayout.getDimAlignments(), outputLayout.getCollapsedIntervals(),
        outputLayout.getOobVal(), outputLayout.getMemorySpace(),
        outputLayout.getMemoryLayout(), AffineMap::get(rewriter.getContext()));

    // Build an affine map that transforms input device coordinates to output
    // device coordinates via the shared logical space. This map handles grid
    // redistribution, collapse changes, padding changes, but NOT virtual grid
    // coordinate translation (which goes in the GridAttr instead).
    AffineMap viewMap = utils::buildLayoutTransformMap(
        inputLayoutForTransform, inputInfo.type, outputLayoutForTransform,
        outputInfo.type);

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
    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());
    BounceTypeBuilder typeBuilder(rewriter.getContext());

    // Compute transformation characteristics
    bool memorySpaceChange =
        inputInfo.getMemorySpace() != outputInfo.getMemorySpace();
    bool systemDeviceTransition =
        inputInfo.hasLayout() != outputInfo.hasLayout();
    bool formatChange =
        inputInfo.type.getElementType() != outputInfo.type.getElementType();
    bool gridShapeChange =
        inputInfo.getGridShape() != outputInfo.getGridShape();

    bool inputHasVirtualGrid = inputInfo.hasLayout() &&
                               !inputInfo.layout->getIndexAffineMap().isEmpty();
    bool outputHasVirtualGrid =
        outputInfo.hasLayout() &&
        !outputInfo.layout->getIndexAffineMap().isEmpty();
    bool virtualGridChange = (inputHasVirtualGrid != outputHasVirtualGrid) ||
                             (inputHasVirtualGrid && gridShapeChange);

    // Decision tree: handle transformations in priority order

    // 1. SYSTEM ↔ DEVICE transitions (highest priority)
    if (systemDeviceTransition) {
      if (!inputInfo.hasLayout()) {
        // System → Device: move to L1
        auto bounceType = typeBuilder.createDeviceType(
            inputInfo.type, *outputInfo.layout, outputInfo.type,
            ttcore::MemorySpace::DeviceL1, getTargetGridShape());
        bounce(rewriter, op, bounceType);
        return success();
      } else {
        // Device → System: need to prepare for transfer
        if (formatChange && ttcore::isTiled(inputInfo.type)) {
          // Untilize first
          Type scalarType = getScalarType(inputInfo.type.getElementType());
          auto bounceType = typeBuilder.modifyDeviceType(
              inputInfo.type, *inputInfo.layout, getTargetGridShape(),
              ttcore::MemorySpace::DeviceL1,
              /*newTensorGrid=*/{}, scalarType,
              /*newTileShape=*/std::nullopt);
          bounce(rewriter, op, bounceType);
          return success();
        } else if (inputHasVirtualGrid) {
          // Reblock virtual grid to physical
          auto bounceType = typeBuilder.modifyDeviceType(
              inputInfo.type, *inputInfo.layout, getTargetGridShape(),
              ttcore::MemorySpace::DeviceL1,
              /*newTensorGrid=*/{}, /*newElementType=*/{},
              /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/true);
          bounce(rewriter, op, bounceType);
          return success();
        } else {
          // Ready for system transfer (handled by D2MLowerToLayoutRewriter)
          return failure();
        }
      }
    }

    // 2. MEMORY SPACE changes within device (L1 ↔ DRAM)
    if (memorySpaceChange) {
      if (!inputInfo.isL1()) {
        // DRAM → L1
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, getTargetGridShape(),
            ttcore::MemorySpace::DeviceL1);
        bounce(rewriter, op, bounceType);
        return success();
      } else {
        // L1 → DRAM (handle other transformations first)
        if (formatChange || virtualGridChange) {
          // These need to be handled before DRAM transfer
          // Fall through to handle them
        } else {
          // Ready for DRAM transfer
          return failure();
        }
      }
    }

    // 3. VIRTUAL GRID changes (must be done before format changes)
    if (virtualGridChange && inputHasVirtualGrid) {
      // Virtual → physical or grid shape change
      auto bounceType = typeBuilder.modifyDeviceType(
          inputInfo.type, *inputInfo.layout, getTargetGridShape(),
          /*memSpace=*/{},
          /*newTensorGrid=*/{}, /*newElementType=*/{},
          /*newTileShape=*/{}, /*reblockVirtualGridShapes=*/true);
      bounce(rewriter, op, bounceType);
      return success();
    }

    // 4. FORMAT changes (tilize/untilize)
    if (formatChange) {
      if (ttcore::isTiled(inputInfo.type)) {
        // Tilize → scalar
        Type scalarType = getScalarType(inputInfo.type.getElementType());
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, getTargetGridShape(),
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, scalarType,
            /*newTileShape=*/std::nullopt);
        bounce(rewriter, op, bounceType);
        return success();
      } else {
        // Scalar → tilize
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, getTargetGridShape(),
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, outputInfo.type.getElementType(),
            ttcore::getTensorTileShape(outputInfo.type));
        bounce(rewriter, op, bounceType);
        return success();
      }
    }

    // 5. MAPPING changes with tiled data (need scalar intermediate)
    if (ttcore::isTiled(inputInfo.type) &&
        (gridShapeChange ||
         inputInfo.type.getShape() != outputInfo.type.getShape())) {
      // Untilize for mapping transformations
      Type scalarType = getScalarType(inputInfo.type.getElementType());
      auto bounceType = typeBuilder.modifyDeviceType(
          inputInfo.type, *inputInfo.layout, getTargetGridShape(),
          /*memSpace=*/{},
          /*newTensorGrid=*/{}, scalarType,
          /*tileShape=*/std::nullopt);
      bounce(rewriter, op, bounceType);
      return success();
    }

    // 6. Not compound - let D2MLowerToLayoutRewriter handle it
    return failure();
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
