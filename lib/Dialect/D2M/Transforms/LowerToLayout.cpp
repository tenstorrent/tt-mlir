// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
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

// Helper to analyze compound ToLayoutOp transformations
struct CompoundComponents {
  bool isMemorySpaceChange = false;
  bool isFormatChange = false;
  bool isMappingChange = false;

  bool isCompound() const {
    int count = 0;
    if (isMemorySpaceChange) {
      count++;
    }
    if (isFormatChange) {
      count++;
    }
    if (isMappingChange) {
      count++;
    }
    return count > 1;
  }

  static CompoundComponents analyze(Value input, Value output) {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    CompoundComponents result;

    // Check memory space change
    result.isMemorySpaceChange =
        (inputInfo.getMemorySpace() != outputInfo.getMemorySpace());

    // Check format change (element type)
    result.isFormatChange =
        (inputInfo.type.getElementType() != outputInfo.type.getElementType());

    // Check mapping change (everything else):
    // - Grid shape changes
    // - Tensor shape changes (collapsed_intervals, alignments)
    // - Memory layout type changes (row-major, blocked, etc.)
    // - Index map changes
    if (inputInfo.hasLayout() && outputInfo.hasLayout() &&
        !result.isMemorySpaceChange && !result.isFormatChange) {

      auto inputLayout = *inputInfo.layout;
      auto outputLayout = *outputInfo.layout;

      // Different tensor shapes
      bool shapeChanged =
          (inputInfo.type.getShape() != outputInfo.type.getShape());

      // Different memory layout types
      bool memLayoutChanged =
          (inputLayout.getMemoryLayout() != outputLayout.getMemoryLayout());

      // Different index maps
      bool indexMapChanged =
          (inputLayout.getIndexAffineMap() != outputLayout.getIndexAffineMap());

      result.isMappingChange =
          shapeChanged || memLayoutChanged || indexMapChanged;
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

  // Lower mapping changes via View
  // All mapping transformations (grid redistribution, alignment changes, etc.)
  // can be expressed as affine maps and are therefore zero-cost views
  static LogicalResult lowerMappingChange(PatternRewriter &rewriter,
                                          ToLayoutOp op) {
    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

    // Preconditions - these should be guaranteed by the split/compound logic
    assert(inputInfo.hasLayout() && outputInfo.hasLayout() &&
           "Mapping change requires both input and output to have layouts");
    assert(inputInfo.getMemorySpace() == outputInfo.getMemorySpace() &&
           "Mapping change should not change memory space");
    assert(inputInfo.type.getElementType() ==
               outputInfo.type.getElementType() &&
           "Mapping change should not change element type");

    return lowerAsView(rewriter, op, inputInfo, outputInfo);
  }

  static LogicalResult lowerAsView(PatternRewriter &rewriter, ToLayoutOp op,
                                   const TensorInfo &inputInfo,
                                   const TensorInfo &outputInfo) {
    auto inputLayout = *inputInfo.layout;
    auto outputLayout = *outputInfo.layout;

    // Check if grid shapes match
    auto inputGrid = inputLayout.getGridShape(inputInfo.type);
    auto outputGrid = outputLayout.getGridShape(outputInfo.type);
    bool sameGrid = (inputGrid == outputGrid);

    AffineMap viewMap;

    if (sameGrid) {
      // When grid shapes match, we can use a simple index map composition
      // at the logical level without going through device space.
      //
      // For example, if both tensors are [2,4,32,32] and we want to transpose
      // the logical view, we compose their 2D logical index maps.

      auto inputIndexMap = inputLayout.getIndexAffineMap();
      auto outputIndexMap = outputLayout.getIndexAffineMap();
      size_t logicalRank = inputLayout.getLogicalShape().size();

      // Compose: output_logical -> input_logical
      // This gives us the view relationship at the logical level
      viewMap = composeLogicalIndexMaps(rewriter.getContext(), inputIndexMap,
                                        outputIndexMap, logicalRank);
    } else {
      // Different grid shapes (redistribution, alignment changes, etc.)
      // Use the full device-to-device affine transformation via logical space
      viewMap = utils::buildLayoutTransformMap(inputLayout, inputInfo.type,
                                               outputLayout, outputInfo.type);
    }

    // Apply the composed map as the index map on the output layout
    auto newLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), outputLayout.getLogicalShape(),
        outputLayout.getDimAlignments(), outputLayout.getCollapsedIntervals(),
        outputLayout.getOobVal(), outputLayout.getMemorySpace(),
        outputLayout.getMemoryLayout(), viewMap);

    auto resultType =
        RankedTensorType::get(outputInfo.type.getShape(),
                              outputInfo.type.getElementType(), newLayout);

    rewriter.replaceOpWithNewOp<ViewLayoutOp>(op, resultType, op.getInput(),
                                              /*reinterpretLayout=*/false);

    return success();
  }

  // Compose two logical-level index maps for view operations
  // When device shapes match, we operate on logical coordinates
  //
  // The index map transforms from logical coordinates to physical coordinates.
  // When creating a view from input to output, we want to express:
  //   "where output logical coords map in terms of input logical coords"
  //
  // Since both layouts share the same device/physical memory, and:
  //   input_physical = inputMap(input_logical)
  //   output_physical = outputMap(output_logical)
  //   input_physical = output_physical (same device shape)
  //
  // Therefore: inputMap(input_logical) = outputMap(output_logical)
  // Solving: input_logical = inputMap^(-1)(outputMap(output_logical))
  //
  // For the view, we just need to store the output's index map directly,
  // since the ViewLayoutOp will reinterpret the input tensor's memory
  // using the output's index map.
  static AffineMap composeLogicalIndexMaps(MLIRContext *ctx,
                                           AffineMap inputIndexMap,
                                           AffineMap outputIndexMap,
                                           size_t logicalRank) {
    // When device shapes match, the output's index map becomes the view's
    // index map. The ViewLayoutOp will reinterpret the input memory using
    // this map.
    if (outputIndexMap.isEmpty()) {
      return AffineMap::getMultiDimIdentityMap(logicalRank, ctx);
    }
    return outputIndexMap;
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
      // (i.e., both have layouts and neither is system memory)
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

    // Route to appropriate lowering based on what's changing
    if (components.isMemorySpaceChange) {
      return lowerDatamovementGeneric(rewriter, op);
    }

    if (components.isFormatChange) {
      return lowerFormatConversionGeneric(rewriter, op);
    }

    if (components.isMappingChange) {
      return lowerMappingChange(rewriter, op);
    }

    // No changes? This shouldn't happen but handle gracefully
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
      if (newTensorGrid.has_value()) {
        tensorGrid.assign(newTensorGrid->begin(), newTensorGrid->end());
      } else {
        auto currentGrid = llvm::to_vector(baseLayout.getGridShape(baseType));
        tensorGrid = currentGrid;
        bool hasVirtualGrid = !baseLayout.getIndexAffineMap().isEmpty();
        tensorGrid =
            (hasVirtualGrid && reblockVirtualGridShapes)
                ? computeVirtualGridBounceShape(currentGrid, targetGridShape)
                : currentGrid;
      }

      auto layout = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(), memSpace,
          baseLayout.getMemoryLayout(), AffineMap::get(ctx));

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

    if (!components.isCompound()) {
      return failure();
    }

    auto inputInfo = TensorInfo::from(op.getInput());
    auto outputInfo = TensorInfo::from(op.getOutput());

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
        // Device -> System: need intermediate in L1 with input's
        // characteristics.
        assert(inputInfo.layout && "Input must have layout for device->system");
        bool reblockVirtualGridShapes = true;
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, getTargetGridShape(),
            ttcore::MemorySpace::DeviceL1,
            /*newTensorGrid=*/{}, outputInfo.type.getElementType(),
            /*newTileShape=*/{}, reblockVirtualGridShapes);
        bounce(rewriter, op, bounceType);
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
        // Tilized -> scalar: use output's layout/grid, change element type.
        auto bounceType = typeBuilder.modifyDeviceType(
            outputInfo.type, *outputInfo.layout, getTargetGridShape(),
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, inputInfo.type.getElementType(),
            ttcore::getTensorTileShape(inputInfo.type));
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
