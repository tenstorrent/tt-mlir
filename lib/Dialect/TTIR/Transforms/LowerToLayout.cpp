// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRLOWERTOLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  static LogicalResult lowerLayoutChange(PatternRewriter &rewriter,
                                         ToLayoutOp op) {
    assert(false && "Layout change not supported yet");
    // TODO file an issue
  }

  static LogicalResult lowerDatamovementGeneric(PatternRewriter &rewriter,
                                                ToLayoutOp op) {
    assert(false && "Layout change not supported yet");
    // TODO file an issue
  }

  static LogicalResult lowerFormatConversionGeneric(PatternRewriter &rewriter,
                                                    ToLayoutOp op) {
    assert(false && "Layout change not supported yet");
    // TODO file an issue
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();
    if (components.isCompound()) {
      return failure();
    }

    if (components.isLayoutChange) {
      return lowerLayoutChange(rewriter, op);
    } else if (components.isGridChange || components.isMemorySpaceChange) {
      return lowerDatamovementGeneric(rewriter, op);
    } else if (components.isFormatChange) {
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
    return ttmlir::utils::createDPSOp<ttir::ToLayoutOp>(
        rewriter, loc, ty.getShape(), ty.getElementType(), desiredLayout,
        input);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               MetalLayoutAttr bounceLayout) const {
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceLayout);
    return rewriter
        .replaceOpWithNewOp<ttir::ToLayoutOp>(
            op, op.getOutput().getType(), bounced->getResult(0), op.getOutput())
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

    // First prioritize moving the data into L1 so we can work with it in L1
    if (!inputL1) {
      // read first into L1, then format convert
      bounce(rewriter, op,
             inputLayout.withMemorySpace(rewriter.getContext(),
                                         MemorySpace::DeviceL1));
    } else if (!outputL1) {
      // format convert first in L1 first, then write
      assert(inputL1 && "input should guaranteed be in L1 because of the "
                        "previous case");
      bounce(rewriter, op,
             outputLayout.withMemorySpace(rewriter.getContext(),
                                          MemorySpace::DeviceL1));
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
