// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace mlir::tt::ttir {
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Generic - Region pass
//===----------------------------------------------------------------------===//

class TTIRGenericRegionRewriter
    : public OpInterfaceRewritePattern<GenericRegionOp> {
public:
  TTIRGenericRegionRewriter(MLIRContext *context)
      : OpInterfaceRewritePattern<GenericRegionOp>(context) {}

  LogicalResult matchAndRewrite(GenericRegionOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    auto dps = cast<DestinationStyleOpInterface>(op.getOperation());

    // Create a generic op.
    auto [indexingMaps, iteratorTypes] = op.getIndexingMaps(rewriter);

    // For testing purposes try getting grid of the resulting tensor and put the
    // op in the grid.
    // TODO(radenko) add a proper debug/test flag.
    auto gridAttr = rewriter.getAttr<GridAttr>();
    auto resEncoding =
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding();
    if (resEncoding) {
      auto resLayout = mlir::cast<MetalLayoutAttr>(resEncoding);
      gridAttr = resLayout.getGrid();
    }

    auto genericOp = rewriter.create<ttir::GenericOp>(
        op.getLoc(), op->getResults().getTypes(), dps.getDpsInputs(),
        ValueRange() /* cbs */, dps.getDpsInits(), gridAttr, indexingMaps,
        iteratorTypes);

    // Create a new basic block for the generic op and create block arguments.
    Block *block = rewriter.createBlock(&genericOp.getRegion());
    SmallVector<Location> blockArgumentLocs(genericOp.getOperands().size(),
                                            genericOp.getLoc());
    SmallVector<Type> blockArgTypes(
        llvm::map_range(genericOp.getOperands().getTypes(), [&](Type type) {
          RankedTensorType tensorType = mlir::cast<RankedTensorType>(type);
          tt::MetalLayoutAttr layout =
              mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
          return layout.getMemref();
        }));
    block->addArguments(blockArgTypes,
                        blockArgumentLocs);

    // Convert the original op into arith/math and into the generic block.
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    op.buildGenericRegion(blockBuilder, block);
    rewriter.replaceOp(op, genericOp);
    return success();
  }
};

class TTIRGenericRegion
    : public impl::TTIRGenericRegionBase<TTIRGenericRegion> {
public:
  using impl::TTIRGenericRegionBase<TTIRGenericRegion>::TTIRGenericRegionBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericRegionRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};

} // namespace mlir::tt::ttir
