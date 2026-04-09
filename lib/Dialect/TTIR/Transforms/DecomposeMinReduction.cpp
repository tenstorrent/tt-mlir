// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDECOMPOSEMINREDUCTION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Decompose min(x) into neg(max(neg(x))) so that backends without native
// reduce_min (e.g. TTMetal/D2M) can reuse the existing reduce_max tile op.
//
// This is a standalone pass following the pattern of other isolated
// decompositions (TTIRDecomposeComplexPermute, TTIRDecomposeComplexReshape).
// It is only needed by the TTMetal pipeline — the TTNN backend has native
// ttnn.min support.
class TTIRDecomposeMinReduction
    : public impl::TTIRDecomposeMinReductionBase<TTIRDecomposeMinReduction> {
public:
  using impl::TTIRDecomposeMinReductionBase<
      TTIRDecomposeMinReduction>::TTIRDecomposeMinReductionBase;

  void runOnOperation() final {
    llvm::SmallVector<MinOp> opsToDecompose;
    getOperation()->walk([&](MinOp op) { opsToDecompose.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (MinOp op : opsToDecompose) {
      if (utils::isOuterReduction(op)) {
        continue;
      }
      rewriter.setInsertionPoint(op);
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

      auto negInput =
          rewriter.create<NegOp>(op.getLoc(), inputType, op.getInput());
      auto maxOp =
          rewriter.create<MaxOp>(op.getLoc(), resultType, negInput.getResult(),
                                 op.getKeepDimAttr(), op.getDimArgAttr());
      auto negResult =
          rewriter.create<NegOp>(op.getLoc(), resultType, maxOp.getResult());
      rewriter.replaceOp(op, negResult);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
