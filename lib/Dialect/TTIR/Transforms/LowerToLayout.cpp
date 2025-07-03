// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRLOWERTOLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

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
class TTIRLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
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
      AffineMap map = mlir::tt::ttir::utils::calculateReblockMap(
          fromTy.getShape(), toTy.getShape(), ctx);
      auto baseLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(fromTy.getEncoding());

      auto enc = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getOobVal(),
          baseLayout.getMemorySpace(), map);

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
          DMAOp dma = isSrcDramOrReblock
                          ? builder.create<ttir::DMAOp>(
                                loc, viewInput, indexingMap, blockArgs[1])
                          : builder.create<ttir::DMAOp>(
                                loc, blockArgs[0], viewOutput, indexingMap);
          builder.create<ttir::DMAWaitOp>(loc, dma);
          builder.create<YieldOp>(loc, blockArgs[1]);
        },
        ThreadType::Datamovement);

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
        });

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

namespace {
class TTIRSplitCompoundLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
  // Helper struct to build intermediate bounce types.
  // This builder will always create a MetalLayoutAttr directly through the
  // primary constructor--it should never make any decisions w.r.t. to grid
  // shape, etc. (those decisions were already made in TTIRToTTIRGeneric, here
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
          memSpace, referenceLayout.getIndexAffineMap());

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
      fprintf(stderr, "++ modifyDeviceType: start\n");
      fprintf(stderr, "++ Base type ");
      baseType.dump();
      fprintf(stderr, "++ Base layout ");
      baseLayout.dump();

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
          baseLayout.getIndexAffineMap());

      ArrayRef<int64_t> tileShape;
      if (mlir::isa<ttcore::TileType>(elementType)) {
        tileShape =
            newTileShape.value_or(ttcore::getTensorTileShapeOrEmpty(baseType));
      }
      auto deviceShape = layout.getDeviceShape(tensorGrid, tileShape);

      const auto ret = RankedTensorType::get(deviceShape, elementType, layout);
      fprintf(stderr, "++ New tensor type: ");
      ret.dump();
      fprintf(stderr, "++ modifyDeviceType: finish\n");
      return ret;
    }

  private:
    MLIRContext *ctx;
  };

public:
  TTIRSplitCompoundLayoutRewriter(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  ttir::ToLayoutOp createToLayoutOp(PatternRewriter &rewriter, Location loc,
                                    Value input,
                                    RankedTensorType desiredType) const {
    // Create empty tensor with desired type and layout
    auto layout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
        desiredType.getEncoding());
    auto output = rewriter.create<ttir::EmptyOp>(
        loc, desiredType.getShape(), desiredType.getElementType(), layout);
    return rewriter.create<ttir::ToLayoutOp>(loc, input, output);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               RankedTensorType bounceType) const {
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceType);
    return rewriter
        .replaceOpWithNewOp<ttir::ToLayoutOp>(op, bounced->getResult(0),
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
};
} // namespace

namespace {
class TTIRLowerToLayout
    : public impl::TTIRLowerToLayoutBase<TTIRLowerToLayout> {
public:
  using impl::TTIRLowerToLayoutBase<TTIRLowerToLayout>::TTIRLowerToLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRSplitCompoundLayoutRewriter, TTIRLowerToLayoutRewriter>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
