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

    auto packResult = linalg::pack(rewriter, op, packedSizes);
    if (LogicalResult(packResult).failed())
      return packResult;

    // struct PackResult {
    //   SmallVector<tensor::PackOp> packOps;
    //   linalg::LinalgOp packedLinalgOp;
    //   SmallVector<tensor::UnPackOp> unPackOps;
    // };

    return packResult;
  }
};

class TTVectorizeRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    std::cout << "asdfvec" << std::endl;
    op.dump();
    /*
    /// Emit a suitable vector form for an operation. If provided,
    /// `inputVectorSizes` are used to vectorize this operation.
    /// `inputVectorSizes` must match the rank of the iteration space of the
    /// operation and the sizes must be smaller or equal than their counterpart
    /// interation space sizes, if static. `inputVectorShapes` also allows the
    /// vectorization of operations with dynamic shapes.
    LogicalResult vectorize(RewriterBase & rewriter, Operation * op,
                            ArrayRef<int64_t> inputVectorSizes = {},
                            ArrayRef<bool> inputScalableVecDims = {},
                            bool vectorizeNDExtract = false,
                            bool flatten1DDepthwiseConv = false);
    */
    SmallVector<int64_t> inputVectorSizes = {1, 1, 32, 32};
    SmallVector<bool> inputScalableVecDims = {true, true, true, true};
    return linalg::vectorize(rewriter, op, inputVectorSizes,
                             inputScalableVecDims);
  }
};

class TTPackGeneric : public impl::TTPackGenericBase<TTPackGeneric> {
public:
  using impl::TTPackGenericBase<
      TTPackGeneric>::TTPackGenericBase;
  void runOnOperation() final {
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
#if 1
    auto module = getOperation();
    module.walk([](linalg::GenericOp generic) {
      for (auto i : generic.getInputs()) {
        i.getType().dump();
      }
      for (auto o : generic.getOutputs()) {
        o.getType().dump();
      }
    });
#endif
    // signalPassFailure();
  }
};
} // namespace
} // namespace mlir::tt
