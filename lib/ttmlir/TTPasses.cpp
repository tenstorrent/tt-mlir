// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ttmlir/TTPasses.h"

namespace mlir::tt {
#define GEN_PASS_DEF_TTPACKGENERIC
#define GEN_PASS_DEF_TTTILIZE
#include "ttmlir/TTPasses.h.inc"

namespace {
class TTPackGenericRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    for (auto attr : op.getIndexingMapsAttr()) {
      AffineMap map = dyn_cast<AffineMapAttr>(attr).getValue();
      if (not map.isIdentity()) {
        return op.emitError("Unsupported affine map access pattern for tilization");
      }
    }

    SmallVector<OpFoldResult> packedSizes;
    auto tile_size = rewriter.getI64IntegerAttr(32);
    packedSizes.push_back(tile_size);
    packedSizes.push_back(tile_size);

    return linalg::pack(rewriter, op, packedSizes);
  }
};

class TTPackGeneric : public impl::TTPackGenericBase<TTPackGeneric> {
public:
  using impl::TTPackGenericBase<
      TTPackGeneric>::TTPackGenericBase;
  void runOnOperation() final {
    auto module = getOperation();
    module.dump();
    RewritePatternSet patterns(&getContext());
    patterns.add<TTPackGenericRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

class TTTilize : public impl::TTTilizeBase<TTTilize> {
public:
  using impl::TTTilizeBase<TTTilize>::TTTilizeBase;
  void runOnOperation() final {
    std::cout << "asdfhi" << std::endl;
    auto module = getOperation();
    module.dump();
#if 0
    auto module = getOperation();
    module.walk([](linalg::GenericOp generic) {
      for (auto i : generic.getInputs()) {
        i.dump();
      }
      for (auto o : generic.getOutputs()) {
        o.dump();
      }
    });
#endif
    // signalPassFailure();
  }
};
} // namespace
} // namespace mlir::tt
