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

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRLOWERTOLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  static LogicalResult lowerLayoutChange(PatternRewriter &rewriter,
                                         ToLayoutOp op) {
    llvm::errs() << "lowerLayoutChange\n";
    // assert(false &&
    //        "TODO issue https://github.com/tenstorrent/tt-mlir/issues/3037");
    return success();
  }

  static LogicalResult lowerSystemLayoutChange(PatternRewriter &rewriter,
                                               ToLayoutOp op) {
    MetalLayoutAttr inputLayout = op.getOrCreateInputLayout();
    MetalLayoutAttr outputLayout = op.getOrCreateOutputLayout();
    bool inputSystem = inputLayout.getMemorySpace() == MemorySpace::System;
    bool outputSystem = outputLayout.getMemorySpace() == MemorySpace::System;
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
    MetalLayoutAttr inputLayout = op.getOrCreateInputLayout();
    MetalLayoutAttr outputLayout = op.getOrCreateOutputLayout();
    if (inputLayout.getMemorySpace() == MemorySpace::System ||
        outputLayout.getMemorySpace() == MemorySpace::System) {
      // To/From host mem is a special case that is lowered to
      // ttmetal.enqueue_write_buffer or ttmetal.enqueue_read_buffer
      return lowerSystemLayoutChange(rewriter, op);
    }

    auto view = rewriter
                    .create<ViewLayoutOp>(op.getLoc(), op.getOutput().getType(),
                                          op.getInput())
                    .getResult();

    // New: Get rank from logical shape
    assert(inputLayout.getLogicalShape().size() ==
           outputLayout.getLogicalShape().size());
    size_t logicalRank = inputLayout.getLogicalShape().size();

    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, logicalRank);
    rewriter.replaceOpWithNewOp<GenericOp>(
        op, view, op.getOutput(),
        [&](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          auto dma = builder.create<ttir::DMAOp>(
              loc, view, mlir::cast<AffineMapAttr>(indexingMaps[0]),
              blockArgs[1]);
          builder.create<ttir::DMAWaitOp>(loc, dma);
        },
        ThreadType::Datamovement);

    return success();
  }

  static LogicalResult lowerFormatConversionGeneric(PatternRewriter &rewriter,
                                                    ToLayoutOp op) {
    op->dump();
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    bool inputTiled = tt::isTiled(inputType);
    bool outputTiled = tt::isTiled(outputType);
    // llvm::errs() << "inputTiled: " << inputTiled
    //              << ", outputTiled: " << outputTiled << ".\n";
    assert(inputTiled != outputTiled &&
           "one of input or output must be tiled for now");

    auto tmp = rewriter.replaceOpWithNewOp<GenericOp>(
        op, ValueRange{op.getInput()}, ValueRange{op.getOutput()},
        [=](OpBuilder &builder, Location loc, ValueRange blockArgs) {
          if (inputTiled) {
            builder.create<TileUntilizeBlockOp>(loc, blockArgs[0],
                                                blockArgs[1]);
          } else {
            builder.create<TileTilizeBlockOp>(loc, blockArgs[0], blockArgs[1]);
          }
        });
    llvm::errs() << "lowerFormatConversionGeneric produced: ";
    tmp->dump();

    return success();
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "TTIRLowerToLayoutRewriter::matchAndRewrite\n";
    op->dump();
    auto components = op.compoundComponents();

    llvm::errs() << "  isLayoutChange: " << components.isLayoutChange << "\n";
    llvm::errs() << "  isGridChange: " << components.isGridChange << "\n";
    llvm::errs() << "  isFormatChange: " << components.isFormatChange << "\n";
    llvm::errs() << "  isMemorySpaceChange: " << components.isMemorySpaceChange
                 << "\n";
    llvm::errs() << "  isCompound(): " << components.isCompound() << "\n";

    if (components.isCompound()) {
      return failure();
    }

    if (components.isLayoutChange) {
      // return lowerLayoutChange(rewriter, op);
      return success();
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
    auto layout = mlir::cast<MetalLayoutAttr>(desiredType.getEncoding());
    auto output = rewriter.create<ttir::EmptyOp>(
        loc, desiredType.getShape(), desiredType.getElementType(), layout);
    return rewriter.create<ttir::ToLayoutOp>(loc, input, output);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               RankedTensorType bounceType) const {
    llvm::errs() << "Bouncing: ";
    op->dump();
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceType);
    llvm::errs() << "Bounced to: ";
    bounced->dump();
    return rewriter
        .replaceOpWithNewOp<ttir::ToLayoutOp>(op, bounced->getResult(0),
                                              op.getOutput())
        ->getResult(0);
  }

  // Helper to create a new tensor type with modified layout
  RankedTensorType
  createModifiedType(MLIRContext *ctx, RankedTensorType baseType,
                     MetalLayoutAttr baseLayout,
                     std::optional<MemorySpace> newMemSpace = {},
                     std::optional<ArrayRef<int64_t>> newGrid = {},
                     std::optional<Type> newElementType = {},
                     std::optional<ArrayRef<int64_t>> newTileShape = {}) const {
    // Log inputs
    // llvm::errs() << "=== createModifiedType ===\n";
    // llvm::errs() << "baseType: " << baseType << "\n";
    // llvm::errs() << "baseType element: " << baseType.getElementType() <<
    // "\n"; llvm::errs() << "baseType shape: "; for (auto s :
    // baseType.getShape()) {
    //   llvm::errs() << s << " ";
    // }
    // llvm::errs() << "\n";
    // llvm::errs() << "baseType tiled? " << tt::isTiled(baseType) << "\n";

    // if (newElementType.has_value()) {
    //   llvm::errs() << "newElementType: " << *newElementType << "\n";
    //   llvm::errs() << "newElementType tiled? "
    //                << mlir::isa<TileType>(*newElementType) << "\n";
    // }

    // if (newTileShape.has_value()) {
    //   llvm::errs() << "newTileShape provided: ";
    //   for (auto s : *newTileShape) {
    //     llvm::errs() << s << " ";
    //   }
    //   llvm::errs() << "\n";
    // }

    // Use existing values if not overridden
    auto memSpace = newMemSpace.value_or(baseLayout.getMemorySpace());
    const bool baseTypeHasLayout = mlir::dyn_cast_or_null<MetalLayoutAttr>(
                                       baseType.getEncoding()) != nullptr;
    auto gridShape = newGrid.value_or(
        baseTypeHasLayout ? tt::getMetalTensorGridShape(baseType)
                          : ArrayRef<int64_t>{1, 1});
    auto elementType = newElementType.value_or(baseType.getElementType());
    auto tileShape =
        newTileShape.has_value()
            ? *newTileShape
            : (tt::isTiled(baseType) ? tt::getMetalTensorTileShape(baseType)
                                     : ArrayRef<int64_t>{});

    // Log derived values
    // llvm::errs() << "Derived gridShape: ";
    // for (auto s : gridShape) {
    //   llvm::errs() << s << " ";
    // }
    // llvm::errs() << "\n";

    // llvm::errs() << "Derived elementType: " << elementType << "\n";
    // llvm::errs() << "Derived tileShape: ";
    // for (auto s : tileShape) {
    //   llvm::errs() << s << " ";
    // }
    // llvm::errs() << "\n";

    // llvm::errs() << "Logical shape: ";
    // for (auto s : baseLayout.getLogicalShape()) {
    //   llvm::errs() << s << " ";
    // }
    // llvm::errs() << "\n";

    // Create new layout
    auto newLayout = MetalLayoutAttr::get(ctx, baseLayout.getLogicalShape(),
                                          baseLayout.getOobVal(), memSpace,
                                          gridShape, tileShape, elementType);

    // For physical shape derivation, use tile shape ONLY if element type is
    // tiled
    ArrayRef<int64_t> tileShapeForPhysical;
    if (mlir::isa<TileType>(elementType)) {
      // Element type is tiled: need tile shape to compute tile counts
      tileShapeForPhysical = tileShape;
    } else {
      // Element type is not tiled: empty to compute element counts
      tileShapeForPhysical = {};
    }

    // Derive physical shape
    auto physicalShape = MetalLayoutAttr::derivePhysicalShape(
        baseLayout.getLogicalShape(), gridShape, tileShapeForPhysical);
    // llvm::errs() << "physical shape: ";
    // for (auto s : physicalShape) {
    //   llvm::errs() << s << " ";
    // }
    // llvm::errs() << "\n";

    return RankedTensorType::get(physicalShape, elementType, newLayout);
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "TTIRSplitCompoundLayoutRewriter::matchAndRewrite\n";
    op->dump();
    auto components = op.compoundComponents();

    llvm::errs() << "  isLayoutChange: " << components.isLayoutChange << "\n";
    llvm::errs() << "  isGridChange: " << components.isGridChange << "\n";
    llvm::errs() << "  isFormatChange: " << components.isFormatChange << "\n";
    llvm::errs() << "  isMemorySpaceChange: " << components.isMemorySpaceChange
                 << "\n";
    llvm::errs() << "  isCompound(): " << components.isCompound() << "\n";
    if (!components.isCompound()) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto inputLayout = op.getOrCreateInputLayout();
    auto outputLayout = op.getOrCreateOutputLayout();

    // llvm::errs() << "TTIRSplitCompoundLayoutRewriter processing:\n";
    // op->dump();
    // llvm::errs() << "Input layout: " << inputLayout << "\n";
    // llvm::errs() << "Input type: " << inputType << "\n";
    // llvm::errs() << "Output layout: " << outputLayout << "\n";
    // llvm::errs() << "Output type: " << outputType << "\n";

    bool inputL1 = inputLayout.getMemorySpace() == MemorySpace::DeviceL1;
    bool outputL1 = outputLayout.getMemorySpace() == MemorySpace::DeviceL1;

    // First prioritize moving the data into L1 so we can work with it in L1
    if (!inputL1) {
      llvm::errs() << "input l1 bounce\n";
      // read first into L1, then format convert
      auto bounceType = createModifiedType(rewriter.getContext(), inputType,
                                           inputLayout, MemorySpace::DeviceL1);
      bounce(rewriter, op, bounceType);
    } else if (!outputL1) {
      llvm::errs() << "output l1 bounce\n";
      // format convert first in L1 first, then write
      assert(inputL1 && "input should guaranteed be in L1 because of the "
                        "previous case");
      auto bounceType = createModifiedType(rewriter.getContext(), outputType,
                                           outputLayout, MemorySpace::DeviceL1);
      bounce(rewriter, op, bounceType);
    } else if (tt::isTiled(inputType) != tt::isTiled(outputType)) {
      // Prioritize moving tiled data
      if (tt::isTiled(inputType)) {
        llvm::errs() << "input tiled bounce\n";
        auto bounceType = createModifiedType(
            rewriter.getContext(),
            outputType, // Use output as base to get its grid
            outputLayout,
            /*memSpace=*/{},
            /*grid=*/{},
            inputType.getElementType(), // But with input's element type
            tt::isTiled(inputType) ? tt::getMetalTensorTileShape(inputType)
                                   : ArrayRef<int64_t>{});
        // assert(bounceType == outputType);
        bounce(rewriter, op, bounceType);
      } else {
        llvm::errs() << "output tiled bounce\n";
        assert(tt::isTiled(outputType));
        auto bounceType = createModifiedType(
            rewriter.getContext(),
            inputType, // Use INPUT as base
            inputLayout,
            /*memSpace=*/{},             // Keep input's memory space
            /*grid=*/{},                 // Keep input's grid
            outputType.getElementType(), // Change to tiled element type
            tt::getMetalTensorTileShape(
                outputType) // Get tile shape from output
        );
        // assert(bounceType == outputType);
        bounce(rewriter, op, bounceType);
      }
    } else if (components.isLayoutChange && tt::isTiled(inputType)) {
      llvm::errs() << "layout change bounce\n";
      // For now to flexibly support layout changes, we need to bounce to scalar
      // first Get scalar element type
      Type scalarType = inputType.getElementType();
      if (auto tileType = mlir::dyn_cast<TileType>(scalarType)) {
        scalarType = tileType.getElementType();
      }

      // Create untiled version with scalar type
      auto bounceType =
          createModifiedType(rewriter.getContext(), inputType, inputLayout,
                             /*memSpace=*/{}, /*grid=*/{}, scalarType,
                             /*tileShape=*/std::nullopt);
      bounce(rewriter, op, bounceType);
    } else if (components.isGridChange) {
      llvm::errs() << "grid change bounce\n";
      assert(!components.isLayoutChange &&
             "Changing layout and grid at the same time is currently "
             "not supported");
      // Keep output layout but with input's grid
      auto bounceType = createModifiedType(
          rewriter.getContext(), outputType, outputLayout,
          /*memSpace=*/{}, tt::getMetalTensorGridShape(inputType));
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
    llvm::errs() << "success:\n";
    // getOperation()->dump();
  }
};

} // namespace mlir::tt::ttir
