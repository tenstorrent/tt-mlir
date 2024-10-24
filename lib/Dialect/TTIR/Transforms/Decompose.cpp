// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDECOMPOSE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Decompose pass
//===----------------------------------------------------------------------===//

// Decompose IndexOp into SliceOp
//
// This transformation adjusts IndexOp attributes so that `begin`, `end`, and
// `step` become arrays, where each array element corresponds to a dimension of
// the input tensor. For dimensions other than the sliced dimension, default
// values are used.
//
struct IndexToSlicePattern : public OpRewritePattern<ttir::IndexOp> {
  using OpRewritePattern<ttir::IndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::IndexOp indexOp,
                                PatternRewriter &rewriter) const override {
    auto inputType =
        ::mlir::cast<mlir::RankedTensorType>(indexOp.getInput().getType());
    if (!inputType)
      return failure();

    int64_t rank = inputType.getRank();
    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == indexOp.getDim()) {
        begins.push_back(rewriter.getI32IntegerAttr(indexOp.getBegin()));
        ends.push_back(rewriter.getI32IntegerAttr(indexOp.getEnd()));
        steps.push_back(rewriter.getI32IntegerAttr(indexOp.getStep()));
      } else {
        begins.push_back(rewriter.getI32IntegerAttr(0));
        ends.push_back(rewriter.getI32IntegerAttr(inputType.getDimSize(i)));
        steps.push_back(rewriter.getI32IntegerAttr(1));
      }
    }

    rewriter.replaceOpWithNewOp<ttir::SliceOp>(
        indexOp, indexOp.getType(), indexOp.getInput(), indexOp.getOutput(),
        rewriter.getArrayAttr(begins), rewriter.getArrayAttr(ends),
        rewriter.getArrayAttr(steps), indexOp.getOperandConstraints());
    return success();
  }
};

class TTIRDecompose : public impl::TTIRDecomposeBase<TTIRDecompose> {
public:
  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.insert<IndexToSlicePattern>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::ttir
