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
} // namespace

namespace {
class D2MLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  static LogicalResult lowerLayoutChange(PatternRewriter &rewriter,
                                         ToLayoutOp op) {
    assert(false &&
           "TODO issue https://github.com/tenstorrent/tt-mlir/issues/3037");
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

  static LogicalResult lowerDatamovementGeneric(PatternRewriter &rewriter,
                                                ToLayoutOp op) {
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
      AffineMap reblockMap = mlir::tt::d2m::utils::calculateReblockMap(
          fromTy.getShape(), toTy.getShape(), ctx);
      auto toLayout = mlir::cast<ttcore::MetalLayoutAttr>(toTy.getEncoding());

      // compose reblocking map with existing index map on toTy
      auto toIndexMap =
          toLayout.getIndexAffineMapOrIdentity(toTy.getShape().size());
      AffineMap mergedMap = toIndexMap.compose(reblockMap);

      auto baseLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(fromTy.getEncoding());

      auto enc = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(),
          baseLayout.getMemorySpace(), baseLayout.getMemoryLayout(), mergedMap);
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

    // TODO: cleanup
    ttcore::GridAttr grid =
        ttcore::GridAttr::get(rewriter.getContext(), outputInfo.getGridShape());
    if (outputInfo.getGridShape()[0] > 8 || outputInfo.getGridShape()[1] > 8) {
      grid = ttcore::GridAttr::get(rewriter.getContext(), {8, 8});
    }

    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, gridRank);
    auto indexingMap = mlir::cast<AffineMapAttr>(indexingMaps[0]);

    rewriter.replaceOpWithNewOp<GenericOp>(
        op, viewInput, viewOutput,
        [&](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          DMAOp dma = isSrcDramOrReblock
                          ? builder.create<d2m::DMAOp>(
                                loc, viewInput, indexingMap, blockArgs[1])
                          : builder.create<d2m::DMAOp>(loc, blockArgs[0],
                                                       viewOutput, indexingMap);
          builder.create<d2m::DMAWaitOp>(loc, dma);
          builder.create<YieldOp>(loc, blockArgs[1]);
        },
        ThreadType::Datamovement, grid);

    return success();
  }

  static LogicalResult lowerFormatConversionGeneric(PatternRewriter &rewriter,
                                                    ToLayoutOp op) {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    bool inputTiled = ttcore::isTiled(inputType);
    bool outputTiled = ttcore::isTiled(outputType);
    assert(inputTiled != outputTiled &&
           "one of input or output must be tiled for now");

    // TODO: cleanup
    ArrayRef<int64_t> outputShape = outputType.getShape();
    assert(outputShape.size() % 2 == 0);
    auto gridShape = outputShape.drop_back(outputShape.size() / 2);

    ttcore::GridAttr grid =
        ttcore::GridAttr::get(rewriter.getContext(), gridShape);
    if (outputShape[0] > 8 || outputShape[1] > 8) {
      grid = ttcore::GridAttr::get(rewriter.getContext(), {8, 8});
    }

    rewriter.replaceOpWithNewOp<GenericOp>(
        op, op.getInput(), op.getOutput(),
        [=](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          if (inputTiled) {
            builder.create<TileUntilizeBlockOp>(loc, blockArgs[0],
                                                blockArgs[1]);
          } else {
            builder.create<TileTilizeBlockOp>(loc, blockArgs[0], blockArgs[1]);
          }
          builder.create<YieldOp>(loc, blockArgs[1]);
        },
        ThreadType::Compute, grid);

    return success();
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();

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

    if (components.isLayoutChange) {
      return lowerLayoutChange(rewriter, op);
    }

    if (components.isGridChange || components.isMemorySpaceChange) {
      return lowerDatamovementGeneric(rewriter, op);
    }

    if (components.isFormatChange) {
      return lowerFormatConversionGeneric(rewriter, op);
    }

    llvm_unreachable("Unknown compound component");
  }
};
} // namespace

// namespace {
// class D2MVirtualGridRewriter : public OpRewritePattern<GenericOp> {
// public:
//   using OpRewritePattern<GenericOp>::OpRewritePattern;
//
//   static bool hasVirtualGrid(ArrayRef<int64_t> tensorGrid,
//                             ArrayRef<int64_t> deviceGrid2D) {
//     if (tensorGrid.size() > 2) {
//       return true;
//     }
//     int64_t gy = tensorGrid.size() >= 1 ? tensorGrid[0] : 1;
//     int64_t gx = tensorGrid.size() >= 2 ? tensorGrid[1] : 1;
//     return gy > deviceGrid2D[0] || gx > deviceGrid2D[1];
//   }
//
//   static bool opHasRawVirtualGridOutput(GenericOp op, ArrayRef<int64_t>
//   deviceGrid2D) {
//     for (Value res : op.getResults()) {
//       auto t = dyn_cast<RankedTensorType>(res.getType());
//       if (!t) {
//         return false;
//       }
//       auto layout =
//           dyn_cast_if_present<tt::ttcore::MetalLayoutAttr>(t.getEncoding());
//       if (!layout) {
//         return false;
//       }
//       ArrayRef<int64_t> grid = layout.getGridShape(t);
//       return hasVirtualGrid(grid, deviceGrid2D);
//     }
//     return false;
//   }
//
//   LogicalResult matchAndRewrite(GenericOp op,
//                                 PatternRewriter &rewriter) const final {
//     SmallVector<int64_t, 2> deviceGrid2D = {8,8};//getDeviceGrid2D(root);
//
//     size_t changesMade = 0;
//     for (Value res : op.getResults()) {
//       auto t = dyn_cast<RankedTensorType>(res.getType());
//       if (!t) continue;
//       auto layout =
//       dyn_cast_if_present<tt::ttcore::MetalLayoutAttr>(t.getEncoding()); if
//       (!layout) continue; ArrayRef<int64_t> grid = layout.getGridShape(t);
//       return hasVirtualGrid(grid, deviceGrid2D);
//     }
//
//     if (changesMade) {
//       return success();
//     } else {
//       return failure();
//     }
//   }
// };
// } // namespace

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

    // Create a device tensor type from a system tensor type, using reference
    // layout's characteristics to populate the MetalLayoutAttr appropriately.
    RankedTensorType createDeviceType(RankedTensorType systemType,
                                      ttcore::MetalLayoutAttr referenceLayout,
                                      RankedTensorType referenceType,
                                      ttcore::MemorySpace memSpace) {

      // Extract the tensor grid from the reference device tensor.
      auto tensorGridShape = referenceLayout.getGridShape(referenceType);

      // Preserve all layout decisions from the referenceType tensor.
      auto layout = ttcore::MetalLayoutAttr::get(
          ctx, referenceLayout.getLogicalShape(),
          referenceLayout.getDimAlignments(),
          referenceLayout.getCollapsedIntervals(), referenceLayout.getOobVal(),
          memSpace, referenceLayout.getMemoryLayout(),
          referenceLayout.getIndexAffineMap());

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
                     std::optional<ttcore::MemorySpace> newMemSpace = {},
                     std::optional<ArrayRef<int64_t>> newTensorGrid = {},
                     std::optional<Type> newElementType = {},
                     std::optional<ArrayRef<int64_t>> newTileShape = {}) {

      assert(baseLayout && "modifyDeviceType requires a layout");

      auto memSpace = newMemSpace.value_or(baseLayout.getMemorySpace());
      auto elementType = newElementType.value_or(baseType.getElementType());

      SmallVector<int64_t> tensorGrid;
      if (newTensorGrid.has_value()) {
        tensorGrid.assign(newTensorGrid->begin(), newTensorGrid->end());
      } else {
        auto currentGrid = baseLayout.getGridShape(baseType);
        tensorGrid.assign(currentGrid.begin(), currentGrid.end());
      }

      auto layout = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(), memSpace,
          baseLayout.getMemoryLayout(), baseLayout.getIndexAffineMap());

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

  // TODO: share one impl between LowerToLayout and TTIRToD2M passes
  static std::pair<AffineMap, AffineMap>
  createCoreVirtMaps(MLIRContext *context, ArrayRef<int64_t> virtualGrid,
                     ArrayRef<int64_t> targetGrid) {

    assert(targetGrid.size() == 2);
    int64_t rank = virtualGrid.size();
    int64_t totalDims = 2 * rank; // grid + shard dims

    bool is2DWidthSharded = (rank == 2) && virtualGrid[1] == 1;
    bool is2DHeightSharded = (rank == 2) && virtualGrid[0] == 1;

    if (is2DWidthSharded || is2DHeightSharded) {

      SmallVector<AffineExpr> forwardMapExprs;
      SmallVector<AffineExpr> inverseMapExprs;

      AffineExpr d0 = getAffineDimExpr(0, context);
      AffineExpr d1 = getAffineDimExpr(1, context);
      AffineExpr d2 = getAffineDimExpr(2, context);
      AffineExpr d3 = getAffineDimExpr(3, context);
      AffineExpr zero = getAffineConstantExpr(0, context);
      AffineExpr gridRowStride = getAffineConstantExpr(targetGrid[0], context);
      AffineExpr gridColStride = getAffineConstantExpr(targetGrid[1], context);

      if (is2DWidthSharded) {
        forwardMapExprs = {d0 % gridRowStride, d0.floorDiv(gridRowStride), d2,
                           d3};
        inverseMapExprs = {d0 * gridRowStride + d1, zero, d2, d3};
      } else if (is2DHeightSharded) {
        forwardMapExprs = {d1.floorDiv(gridColStride), d1 % gridColStride, d2,
                           d3};
        inverseMapExprs = {zero, d1 * gridColStride + d0, d2, d3};
      }
      auto forward =
          mlir::AffineMap::get(totalDims, 0, forwardMapExprs, context);
      auto inverse =
          mlir::AffineMap::get(totalDims, 0, inverseMapExprs, context);
      return {forward, inverse};
    } else {
      llvm_unreachable("Expected 2D width or height sharded");
    }
  }

  // computes a workable bounce shape grid for a virtual grid
  llvm::SmallVector<int64_t>
  computeVirtualGridBounceShape(ArrayRef<int64_t> virtualGridShape,
                                ArrayRef<int64_t> deviceGridShape) const {
    // TODO: generalize to N dimensions
    assert(virtualGridShape.size() == 2);
    assert(virtualGridShape[0] > deviceGridShape[0] ^
           virtualGridShape[1] > deviceGridShape[1]);

    // Helper lambda for finding the greatest divisor <= maxDivisor for n
    auto findMaxDivisor = [](int64_t n, int64_t maxDivisor) -> int64_t {
      for (int64_t i = maxDivisor; i > 0; --i) {
        if (n % i == 0) {
          return i;
        }
      }
      llvm_unreachable("No greatest divisor found");
    };

    llvm::SmallVector<int64_t> ret;
    if (virtualGridShape[0] > deviceGridShape[0]) {
      int64_t divisor = findMaxDivisor(virtualGridShape[0], deviceGridShape[0]);
      ret = {divisor, virtualGridShape[1]};
    } else {
      int64_t divisor = findMaxDivisor(virtualGridShape[1], deviceGridShape[1]);
      ret = {virtualGridShape[0], divisor};
    }
    assert(ret.size());
    return ret;
  }

  d2m::ToLayoutOp createToLayoutOp(PatternRewriter &rewriter, Location loc,
                                   Value input,
                                   RankedTensorType desiredType) const {
    // Create empty tensor with desired type and layout
    auto layout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
        desiredType.getEncoding());

    SmallVector<int64_t> emptyShape(desiredType.getShape());

    // compute underlying shape for virtual grid if needed
    auto gridShape = layout.getGridShape(desiredType);
    if (gridShape[0] > 8 || gridShape[1] > 8) {

      SmallVector<int64_t> tileShape = {1, 1};
      if (ttcore::isTiled(desiredType)) {
        constexpr std::array<int64_t, 2> defaultShape =
            ttcore::TileType::getDefaultShape();
        tileShape =
            SmallVector<int64_t>(defaultShape.begin(), defaultShape.end());
      }
      llvm::SmallVector<int64_t> unshardedShape =
          layout.getPhysicalShape(tileShape);

      auto bounceGridShape =
          computeVirtualGridBounceShape(gridShape, targetGridShape);
      llvm::SmallVector<int64_t> shardedShape =
          layout.getDeviceShape(bounceGridShape, tileShape);

      emptyShape = shardedShape;

      // remove index affine map from layout, preserving everything else
      auto map = rewriter.getMultiDimIdentityMap(emptyShape.size());
      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), layout.getLogicalShape(),
          layout.getDimAlignments(), layout.getCollapsedIntervals(),
          layout.getOobVal(), layout.getMemorySpace(), layout.getMemoryLayout(),
          map);
    }

    auto emptyOp = rewriter.create<d2m::EmptyOp>(
        loc, emptyShape, desiredType.getElementType(), layout);
    Value toLayoutDest = emptyOp;

    return rewriter.create<d2m::ToLayoutOp>(loc, input, toLayoutDest);
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
    auto components = op.compoundComponents();

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
            ttcore::MemorySpace::DeviceL1);
        bounce(rewriter, op, bounceType);
      } else {
        // Device -> System: need intermediate in L1 with input's
        // characteristics.
        assert(inputInfo.layout && "Input must have layout for device->system");

        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout, ttcore::MemorySpace::DeviceL1,
            /*newTensorGrid=*/{}, outputInfo.type.getElementType());
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
          inputInfo.type, *inputInfo.layout, ttcore::MemorySpace::DeviceL1);
      bounce(rewriter, op, bounceType);
    } else if (!outputInfo.isL1()) {
      // Move output to L1, preserving its grid and layout characteristics.
      auto bounceType = typeBuilder.modifyDeviceType(
          outputInfo.type, *outputInfo.layout, ttcore::MemorySpace::DeviceL1);
      bounce(rewriter, op, bounceType);
    } else if (ttcore::isTiled(inputInfo.type) !=
               ttcore::isTiled(outputInfo.type)) {
      // Format conversion
      if (ttcore::isTiled(inputInfo.type)) {
        // Tilized -> scalar: use output's layout/grid, change element type.
        auto bounceType = typeBuilder.modifyDeviceType(
            outputInfo.type, *outputInfo.layout,
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, inputInfo.type.getElementType(),
            ttcore::getTensorTileShape(inputInfo.type));
        bounce(rewriter, op, bounceType);
      } else {
        // Scalar -> tilized: use input's layout/grid, change element type.
        auto bounceType = typeBuilder.modifyDeviceType(
            inputInfo.type, *inputInfo.layout,
            /*memSpace=*/{},
            /*newTensorGrid=*/{}, outputInfo.type.getElementType(),
            ttcore::getTensorTileShape(outputInfo.type));
        bounce(rewriter, op, bounceType);
      }
    } else if (components.isLayoutChange && ttcore::isTiled(inputInfo.type)) {
      // Layout change with tiled data - bounce through scalar.
      Type scalarType = inputInfo.type.getElementType();
      if (auto tileType = mlir::dyn_cast<ttcore::TileType>(scalarType)) {
        scalarType = tileType.getElementType();
      }

      auto bounceType =
          typeBuilder.modifyDeviceType(inputInfo.type, *inputInfo.layout,
                                       /*memSpace=*/{},
                                       /*newTensorGrid=*/{}, scalarType,
                                       /*tileShape=*/std::nullopt);
      bounce(rewriter, op, bounceType);
    } else if (components.isGridChange) {
      // Grid change - create intermediate with input's grid but output's
      // layout.
      auto bounceType = typeBuilder.modifyDeviceType(
          outputInfo.type, *outputInfo.layout,
          /*memSpace=*/{}, inputInfo.getGridShape());
      bounce(rewriter, op, bounceType);
    } else {
      // Note we should eventually support DRAM <-> DRAM, or System <-> System
      // w/ format conversion via streaming supported.
      assert(false && "Unsupported compound layout change");
      return failure();
    }

    return success();
  }

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

    llvm::SmallVector<int64_t> targetGridShape = getTargetGridShape();

    patterns.add<D2MSplitCompoundLayoutRewriter>(&getContext(),
                                                 targetGridShape);
    patterns.add<D2MLowerToLayoutRewriter>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxNumRewrites(2);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
