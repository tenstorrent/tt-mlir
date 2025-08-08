// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRLOWERTOLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

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
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    auto inputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        inputType.getEncoding());
    auto outputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        outputType.getEncoding());

    bool inputSystem = !inputLayout || inputLayout.getMemorySpace() ==
                                           ttcore::MemorySpace::System;
    bool outputSystem = !outputLayout || outputLayout.getMemorySpace() ==
                                             ttcore::MemorySpace::System;

    assert(inputSystem != outputSystem &&
           "one of input or output must be system for now");

    if (op.getLayout()) {
      // Already lowered.
      return failure();
    }

    rewriter.replaceOpWithNewOp<ToLayoutOp>(op, op.getInput(), op.getOutput(),
                                            inputSystem ? outputLayout
                                                        : inputLayout);
    return success();
  }

  static LogicalResult lowerDatamovementGeneric(PatternRewriter &rewriter,
                                                ToLayoutOp op) {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    auto inputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        inputType.getEncoding());
    auto outputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        outputType.getEncoding());

    // Check if this is a system memory transfer.
    bool inputSystem = !inputLayout || inputLayout.getMemorySpace() ==
                                           ttcore::MemorySpace::System;
    bool outputSystem = !outputLayout || outputLayout.getMemorySpace() ==
                                             ttcore::MemorySpace::System;

    if (inputSystem || outputSystem) {
      // To/From host mem is a special case that is lowered to
      // ttmetal.enqueue_write_buffer or ttmetal.enqueue_read_buffer.
      return lowerSystemLayoutChange(rewriter, op);
    }

    // At this point, both must have layouts and be on device.
    assert(inputLayout && outputLayout &&
           "Both tensors must have layouts for device-to-device transfer");

    Value viewInput = op.getInput();

    // If grid shapes differ, we need a view to reblock.
    auto inputGridShape = inputLayout.getGridShape(inputType);
    auto outputGridShape = outputLayout.getGridShape(outputType);

    if (inputGridShape != outputGridShape) {
      viewInput = rewriter
                      .create<ViewLayoutOp>(op.getLoc(), op.getInput(),
                                            outputType.getShape())
                      .getResult();
    }

    const size_t gridRank = outputGridShape.size();

    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, gridRank);
    rewriter.replaceOpWithNewOp<GenericOp>(
        op, viewInput, op.getOutput(),
        [&](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          auto dma = builder.create<ttir::DMAOp>(
              loc, viewInput, mlir::cast<AffineMapAttr>(indexingMaps[0]),
              blockArgs[1]);
          builder.create<ttir::DMAWaitOp>(loc, dma);
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
        });

    return success();
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();

    if (components.isCompound()) {
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

class TTIRSplitCompoundLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  TTIRSplitCompoundLayoutRewriter(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  ttir::ToLayoutOp createToLayoutOp(PatternRewriter &rewriter, Location loc,
                                    Value input,
                                    RankedTensorType desiredType) const {
    // Create empty tensor with desired type and layout
    auto layout =
        mlir::cast<ttcore::MetalLayoutAttr>(desiredType.getEncoding());
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

  // Helper to create a new tensor type with modified layout
  RankedTensorType createModifiedType(
      MLIRContext *ctx, RankedTensorType baseType,
      ttcore::MetalLayoutAttr baseLayout,      // Can be null
      ttcore::MetalLayoutAttr referenceLayout, // Layout to copy collapse
                                               // behavior from (can be null)
      ArrayRef<int64_t> workerGridShape,
      std::optional<ttcore::MemorySpace> newMemSpace = {},
      std::optional<ArrayRef<int64_t>> newGrid = {},
      std::optional<Type> newElementType = {},
      std::optional<ArrayRef<int64_t>> newTileShape = {}) const {

    Type elementType = newElementType.value_or(baseType.getElementType());

    // If no base layout and no memory space override, return tensor without
    // layout.
    if (!baseLayout && !newMemSpace.has_value()) {
      return RankedTensorType::get(baseType.getShape(), elementType);
    }

    // If we have a memory space override but no base layout,
    // we're creating a layout (host -> device transition).
    if (!baseLayout) {
      assert(newMemSpace.has_value());
      assert(referenceLayout);

      // Create grid shape based on whether we're collapsing or not.
      llvm::SmallVector<int64_t> logicalGridShape;
      if (newGrid.has_value()) {
        logicalGridShape.assign(newGrid->begin(), newGrid->end());
      } else {
        auto refType = RankedTensorType::get(baseType.getShape(), elementType,
                                             referenceLayout);
        ArrayRef<int64_t> refGrid = referenceLayout.getGridShape(refType);
        logicalGridShape.assign(refGrid.begin(), refGrid.end());
      }

      auto newLayout = ttcore::MetalLayoutAttr::get(
          ctx, baseType.getShape(), workerGridShape, ttcore::OOBVal::Undef,
          *newMemSpace, referenceLayout.getCollapsedIntervals(),
          referenceLayout.getDimAlignments());

      // For physical shape derivation, use tile shape ONLY if element type is
      // tiled.
      ArrayRef<int64_t> tileShapeForPhysical;
      if (mlir::isa<ttcore::TileType>(elementType)) {
        tileShapeForPhysical = newTileShape.value_or(ArrayRef<int64_t>{});
      }

      // Calculate device shape using the logical grid shape we created;
      // getDeviceShape will handle the collapse intervals internally.
      auto deviceShape =
          newLayout.getDeviceShape(logicalGridShape, tileShapeForPhysical);

      return RankedTensorType::get(deviceShape, elementType, newLayout);
    }

    auto memSpace = newMemSpace.value_or(baseLayout.getMemorySpace());

    // We need to create an owning version of gridShape.
    SmallVector<int64_t, 2> gridShape;
    if (newGrid.has_value()) {
      gridShape.assign(newGrid->begin(), newGrid->end());
    } else {
      llvm::ArrayRef<int64_t> tempGrid = baseLayout.getGridShape(baseType);
      gridShape.assign(tempGrid.begin(), tempGrid.end());
    }

    llvm::ArrayRef<int64_t> tileShape =
        newTileShape.has_value() ? *newTileShape
                                 : ttcore::getTensorTileShapeOrEmpty(baseType);

    // Create new layout, preserving collapse intervals from base.
    auto newLayout = ttcore::MetalLayoutAttr::get(
        ctx, baseLayout.getLogicalShape(), gridShape, baseLayout.getOobVal(),
        memSpace, baseLayout.getCollapsedIntervals(),
        baseLayout.getDimAlignments());

    // For physical shape derivation, use tile shape ONLY if element type is
    // tiled.
    ArrayRef<int64_t> tileShapeForPhysical;
    if (mlir::isa<ttcore::TileType>(elementType)) {
      tileShapeForPhysical = tileShape;
    }

    // Create new device tensor shape.
    llvm::SmallVector<int64_t> deviceShape =
        newLayout.getDeviceShape(gridShape, tileShapeForPhysical);

    return RankedTensorType::get(deviceShape, elementType, newLayout);
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();

    if (!components.isCompound()) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    auto inputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        inputType.getEncoding());
    auto outputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        outputType.getEncoding());

    // Determine if we're in L1 - no layout means system memory.
    bool inputL1 = inputLayout && inputLayout.getMemorySpace() ==
                                      ttcore::MemorySpace::DeviceL1;
    bool outputL1 = outputLayout && outputLayout.getMemorySpace() ==
                                        ttcore::MemorySpace::DeviceL1;

    ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
    llvm::ArrayRef<int64_t> workerGridShape =
        deviceAttr.getWorkerGrid().getShape();

    // First prioritize moving the data into L1 so we can work with it in L1.
    if (!inputL1) {
      // Read into L1, then do other conversions.
      // When going from no layout to layout, use output's grid if available.
      auto gridShape = outputLayout ? outputLayout.getGridShape(outputType)
                                    : ArrayRef<int64_t>{};
      auto bounceType = createModifiedType(
          rewriter.getContext(), inputType, inputLayout,
          outputLayout, // reference layout
          workerGridShape, ttcore::MemorySpace::DeviceL1, gridShape);
      bounce(rewriter, op, bounceType);
    } else if (!outputL1) {
      // Convert to L1 first, then do other conversions.
      // When going from layout to no layout, preserve input's grid.
      auto gridShape = inputLayout ? inputLayout.getGridShape(inputType)
                                   : ArrayRef<int64_t>{};
      auto bounceType = createModifiedType(
          rewriter.getContext(), outputType, outputLayout,
          inputLayout, // reference layout
          workerGridShape, ttcore::MemorySpace::DeviceL1, gridShape);
      bounce(rewriter, op, bounceType);
    } else if (ttcore::isTiled(inputType) != ttcore::isTiled(outputType)) {
      // Prioritize moving tiled data.
      if (ttcore::isTiled(inputType)) {
        auto bounceType =
            createModifiedType(rewriter.getContext(), outputType, outputLayout,
                               inputLayout, // reference layout
                               workerGridShape,
                               /*memSpace=*/{},
                               /*grid=*/{}, inputType.getElementType(),
                               ttcore::getTensorTileShapeOrEmpty(inputType));
        bounce(rewriter, op, bounceType);
      } else {
        assert(ttcore::isTiled(outputType));
        auto bounceType =
            createModifiedType(rewriter.getContext(), inputType, inputLayout,
                               outputLayout, // reference layout
                               workerGridShape,
                               /*memSpace=*/{},
                               /*grid=*/{}, outputType.getElementType(),
                               ttcore::getTensorTileShape(outputType));
        bounce(rewriter, op, bounceType);
      }
    } else if (components.isLayoutChange && ttcore::isTiled(inputType)) {
      // For now to flexibly support layout changes, we need to bounce to scalar
      // first.
      Type scalarType = inputType.getElementType();
      if (auto tileType = mlir::dyn_cast<ttcore::TileType>(scalarType)) {
        scalarType = tileType.getElementType();
      }

      // Create untiled version with scalar type.
      // Use input layout as reference since we're modifying the input side.
      auto bounceType =
          createModifiedType(rewriter.getContext(), inputType, inputLayout,
                             inputLayout, // reference layout
                             workerGridShape,
                             /*memSpace=*/{}, /*grid=*/{}, scalarType,
                             /*tileShape=*/std::nullopt);
      bounce(rewriter, op, bounceType);
    } else if (components.isGridChange) {
      assert(!components.isLayoutChange &&
             "Changing layout and grid at the same time is currently "
             "not supported");
      // Keep output layout but with input's grid.
      // Handle case where input might not have a layout (use default grid).
      llvm::SmallVector<int64_t> gridShape;
      if (inputLayout) {
        auto tempGrid = inputLayout.getGridShape(inputType);
        gridShape.assign(tempGrid.begin(), tempGrid.end());
      } else {
        // Use output layout as reference for collapse behavior.
        auto tempGrid = outputLayout.getGridShape(inputType);
        gridShape.assign(tempGrid.begin(), tempGrid.end());
      }
      auto bounceType = createModifiedType(
          rewriter.getContext(), outputType, outputLayout,
          outputLayout ? outputLayout
                       : inputLayout, // Prefer output as reference
          workerGridShape,
          /*memSpace=*/{}, gridShape);
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

} // namespace mlir::tt::ttir
