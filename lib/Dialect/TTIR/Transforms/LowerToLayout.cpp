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
    assert(false &&
           "TODO issue https://github.com/tenstorrent/tt-mlir/issues/3037");
  }

  static LogicalResult lowerSystemLayoutChange(PatternRewriter &rewriter,
                                               ToLayoutOp op) {
    MetalLayoutAttr inputLayout = op.getInputLayout();
    MetalLayoutAttr outputLayout = op.getOutputLayout();
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
    MetalLayoutAttr inputLayout = op.getInputLayout();
    MetalLayoutAttr outputLayout = op.getOutputLayout();
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

    assert(inputLayout.getRank() == outputLayout.getRank());
    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, inputLayout.getRank());
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
    bool inputTiled = op.getInputLayout().isTiled();
    bool outputTiled = op.getOutputLayout().isTiled();
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
                                    MetalLayoutAttr desiredLayout) const {
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output = rewriter.create<ttir::EmptyOp>(
        loc, ty.getShape(), ty.getElementType(), desiredLayout);
    return rewriter.create<ttir::ToLayoutOp>(loc, input, output);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               MetalLayoutAttr bounceLayout) const {
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceLayout);
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

    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto inputLayout = op.getInputLayout();
    auto outputLayout = op.getOutputLayout();

    bool inputL1 = inputLayout.getMemorySpace() == MemorySpace::DeviceL1;
    bool outputL1 = outputLayout.getMemorySpace() == MemorySpace::DeviceL1;

    bool hostTx = inputLayout.getMemorySpace() == MemorySpace::System ||
                  outputLayout.getMemorySpace() == MemorySpace::System;

    // Allow compound memorySpaceChange && gridChange for host txns. Only these
    // two allowed together.
    if (components.isCompound() && hostTx) {
      if (components.isMemorySpaceChange && components.isGridChange &&
          !components.isFormatChange && !components.isLayoutChange) {
        return failure();
      }
    }
    // First prioritize moving the data into L1 so we can work with it in L1
    if (!inputL1 && !hostTx) {
      // read first into L1, then format convert
      bounce(rewriter, op,
             inputLayout.withMemorySpace(rewriter.getContext(),
                                         MemorySpace::DeviceL1));
    } else if (!inputL1 && hostTx) {
      // read first into L1, then format convert
      bounce(rewriter, op,
             inputLayout
                 .withMemorySpace(rewriter.getContext(), MemorySpace::DeviceL1)
                 .withGrid(rewriter.getContext(), outputType,
                           outputLayout.getGrid()));
    } else if (!outputL1 && !hostTx) {
      // format convert first in L1 first, then write
      assert(inputL1 && "input should guaranteed be in L1 because of the "
                        "previous case");
      bounce(rewriter, op,
             outputLayout.withMemorySpace(rewriter.getContext(),
                                          MemorySpace::DeviceL1));
    } else if (!outputL1 && hostTx) {
      // format convert first in L1 first, then write
      bounce(rewriter, op,
             outputLayout
                 .withMemorySpace(rewriter.getContext(), MemorySpace::DeviceL1)
                 .withGrid(rewriter.getContext(), outputType,
                           inputLayout.getGrid()));
    } else if (inputLayout.isTiled() != outputLayout.isTiled()) {
      // Prioritize moving tiled data
      if (inputLayout.isTiled()) {
        bounce(rewriter, op,
               outputLayout.withElementType(rewriter.getContext(),
                                            inputLayout.getElementType()));
      } else {
        assert(outputLayout.isTiled());
        bounce(rewriter, op,
               inputLayout.withElementType(rewriter.getContext(),
                                           outputLayout.getElementType()));
      }
    } else if (components.isLayoutChange && inputLayout.isTiled()) {
      // For now to flexibly support layout changes, we need to bounce to scalar
      // first
      bounce(rewriter, op,
             inputLayout.withElementType(rewriter.getContext(),
                                         inputLayout.getScalarElementType()));
    } else if (components.isGridChange) {
      assert(!components.isLayoutChange &&
             "Changing layout and grid at the same time is currently "
             "not supported");
      bounce(rewriter, op,
             outputLayout.withGrid(rewriter.getContext(), outputType,
                                   inputLayout.getGrid()));
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
