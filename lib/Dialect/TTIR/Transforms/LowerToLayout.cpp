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
    ttcore::MetalLayoutAttr inputLayout = op.getOrCreateInputLayout();
    ttcore::MetalLayoutAttr outputLayout = op.getOrCreateOutputLayout();
    bool inputSystem =
        inputLayout.getMemorySpace() == ttcore::MemorySpace::System;
    bool outputSystem =
        outputLayout.getMemorySpace() == ttcore::MemorySpace::System;
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

  // Return true if the input operand to a ToLayoutOp is itself a result of a
  // device->device memspace ToLayoutOp.
  static bool producerMustBeLoweredFirst(ToLayoutOp op) {
    if (auto producer = op.getInput().getDefiningOp<ToLayoutOp>()) {
      auto inputOperandMemspace =
          producer.getOrCreateInputLayout().getMemorySpace();
      auto outputOperandMemspace =
          producer.getOrCreateOutputLayout().getMemorySpace();
      if (ttcore::isDeviceMemorySpace(inputOperandMemspace) &&
          ttcore::isDeviceMemorySpace(outputOperandMemspace)) {
        return true;
      }
    }
    return false;
  }

  static LogicalResult lowerDatamovementGeneric(PatternRewriter &rewriter,
                                                ToLayoutOp op) {
    ttcore::MetalLayoutAttr inputLayout = op.getOrCreateInputLayout();
    ttcore::MetalLayoutAttr outputLayout = op.getOrCreateOutputLayout();
    if (inputLayout.getMemorySpace() == ttcore::MemorySpace::System ||
        outputLayout.getMemorySpace() == ttcore::MemorySpace::System) {
      // To/From host mem is a special case that is lowered to
      // ttmetal.enqueue_write_buffer or ttmetal.enqueue_read_buffer
      return lowerSystemLayoutChange(rewriter, op);
    }

    // Get the shapes to determine if we need a view.
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());

    Value viewInput = op.getInput();

    // If grid shapes differ, we need a view to reblock.
    auto outputGridShape = outputLayout.getGridShape(outputType);
    auto inputGridShape = inputLayout.getGridShape(inputType);

    bool isSrcDram =
        inputLayout.getMemorySpace() == ttcore::MemorySpace::DeviceDRAM;
    bool isDstDram =
        outputLayout.getMemorySpace() == ttcore::MemorySpace::DeviceDRAM;

    // If src or dst operand is DRAM they must be remote
    // else, Lower L1->L1 reblocking as READs (view applied to the src, dst is
    // local).
    bool isSrcDramOrReblock =
        isSrcDram || (!isDstDram && (inputGridShape != outputGridShape));

    assert(!(isSrcDramOrReblock && isDstDram) &&
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
      viewInput = buildConcreteView(op.getInput(), inputType, outputType);
    }

    Value viewOutput = op.getOutput();
    if (isDstDram) {
      auto outTensorTy = mlir::cast<RankedTensorType>(op.getOutput().getType());
      viewOutput = buildConcreteView(op.getOutput(), outTensorTy, inputType);
    }

    const size_t gridRank = outputGridShape.size();

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
  RankedTensorType
  createModifiedType(MLIRContext *ctx, RankedTensorType baseType,
                     ttcore::MetalLayoutAttr baseLayout,
                     ArrayRef<int64_t> workerGridShape,
                     std::optional<ttcore::MemorySpace> newMemSpace = {},
                     std::optional<ArrayRef<int64_t>> newGrid = {},
                     std::optional<Type> newElementType = {},
                     std::optional<ArrayRef<int64_t>> newTileShape = {}) const {
    // Use existing values if not overridden
    auto memSpace = newMemSpace.value_or(baseLayout.getMemorySpace());
    auto maybeBaseTypeLayout =
        mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(baseType.getEncoding());
    const bool baseTypeHasLayout = maybeBaseTypeLayout != nullptr;

    // We need to create an owning version of gridShape for the case where we
    // default 1-fill it, which makes this more complex/ugly.
    SmallVector<int64_t, 2> gridShape;
    if (newGrid.has_value()) {
      gridShape.assign(newGrid->begin(), newGrid->end());
    } else if (baseTypeHasLayout) {
      llvm::ArrayRef<int64_t> tempGrid =
          maybeBaseTypeLayout.getGridShape(baseType);
      gridShape.assign(tempGrid.begin(), tempGrid.end());
    }

    Type elementType = newElementType.value_or(baseType.getElementType());
    llvm::ArrayRef<int64_t> tileShape =
        newTileShape.has_value() ? *newTileShape
                                 : ttcore::getTensorTileShapeOrEmpty(baseType);

    // Create new layout
    SmallVector<int64_t> squareGridShape =
        ttir::utils::getSquareTargetGrid(workerGridShape);
    auto newLayout = ttcore::MetalLayoutAttr::get(
        ctx, baseLayout.getLogicalShape(), squareGridShape,
        baseLayout.getOobVal(), memSpace, baseLayout.getCollapsedIntervals());

    // For physical shape derivation, use tile shape ONLY if element type is
    // tiled
    ArrayRef<int64_t> tileShapeForPhysical;
    if (mlir::isa<ttcore::TileType>(elementType)) {
      // Element type is tiled: need tile shape to compute tile counts
      tileShapeForPhysical = tileShape;
    } else {
      // Element type is not tiled: empty to compute element counts
      tileShapeForPhysical = {};
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
    const bool hasInputLayout = inputType.getEncoding() != nullptr;
    const bool hasOutputLayout = outputType.getEncoding() != nullptr;
    auto inputLayout = op.getOrCreateInputLayout();
    auto outputLayout = op.getOrCreateOutputLayout();

    bool inputL1 =
        inputLayout.getMemorySpace() == ttcore::MemorySpace::DeviceL1;
    bool outputL1 =
        outputLayout.getMemorySpace() == ttcore::MemorySpace::DeviceL1;

    ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
    llvm::ArrayRef<int64_t> workerGridShape =
        deviceAttr.getWorkerGrid().getShape();

    // First prioritize moving the data into L1 so we can work with it in L1
    if (!inputL1) {
      // Read into L1, then do other conversions.
      // If we're going from no grid -> grid, we need to use the output grid.
      if (!hasInputLayout && hasOutputLayout) {
        auto gridShape = outputLayout.getGridShape(outputType);
        auto bounceType = createModifiedType(
            rewriter.getContext(), inputType, inputLayout, workerGridShape,
            ttcore::MemorySpace::DeviceL1, gridShape);
        bounce(rewriter, op, bounceType);
      } else {
        // For other cases, we want to use input's current grid
        auto bounceType =
            createModifiedType(rewriter.getContext(), inputType, inputLayout,
                               workerGridShape, ttcore::MemorySpace::DeviceL1);
        bounce(rewriter, op, bounceType);
      }
    } else if (!outputL1) {
      // Convert to L1 first, then do other conversions.
      assert(inputL1 && "input should guaranteed be in L1 because of the "
                        "previous case");
      // Conversely, if we're going from grid -> no grid, we need to use the
      // input grid.
      if (!hasOutputLayout && hasInputLayout) {
        auto gridShape = inputLayout.getGridShape(inputType);
        auto bounceType = createModifiedType(
            rewriter.getContext(), outputType, outputLayout, workerGridShape,
            ttcore::MemorySpace::DeviceL1, gridShape);
        bounce(rewriter, op, bounceType);
      } else {
        // For other cases, we want to use output's current grid
        auto bounceType =
            createModifiedType(rewriter.getContext(), outputType, outputLayout,
                               workerGridShape, ttcore::MemorySpace::DeviceL1);
        bounce(rewriter, op, bounceType);
      }
    } else if (ttcore::isTiled(inputType) != ttcore::isTiled(outputType)) {
      // Prioritize moving tiled data
      if (ttcore::isTiled(inputType)) {
        auto bounceType = createModifiedType(
            rewriter.getContext(), outputType, outputLayout, workerGridShape,
            /*memSpace=*/{},
            /*grid=*/{}, inputType.getElementType(),
            ttcore::getTensorTileShapeOrEmpty(inputType));
        bounce(rewriter, op, bounceType);
      } else {
        assert(ttcore::isTiled(outputType));
        auto bounceType = createModifiedType(
            rewriter.getContext(), inputType, inputLayout, workerGridShape,
            /*memSpace=*/{},
            /*grid=*/{}, outputType.getElementType(),
            ttcore::getTensorTileShape(outputType));
        bounce(rewriter, op, bounceType);
      }
    } else if (components.isLayoutChange && ttcore::isTiled(inputType)) {
      // For now to flexibly support layout changes, we need to bounce to scalar
      // first Get scalar element type
      Type scalarType = inputType.getElementType();
      if (auto tileType = mlir::dyn_cast<ttcore::TileType>(scalarType)) {
        scalarType = tileType.getElementType();
      }

      // Create untiled version with scalar type
      auto bounceType = createModifiedType(
          rewriter.getContext(), inputType, inputLayout, workerGridShape,
          /*memSpace=*/{}, /*grid=*/{}, scalarType,
          /*tileShape=*/std::nullopt);
      bounce(rewriter, op, bounceType);
    } else if (components.isGridChange) {
      assert(!components.isLayoutChange &&
             "Changing layout and grid at the same time is currently "
             "not supported");
      // Keep output layout but with input's grid
      auto bounceType = createModifiedType(
          rewriter.getContext(), outputType, outputLayout, workerGridShape,
          /*memSpace=*/{}, inputLayout.getGridShape(inputType));
      bounce(rewriter, op, bounceType);
    } else {
      // Note we should eventually support DRAM <-> DRAM, or System <-> System
      // w/ format conversion via streaming supported
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
