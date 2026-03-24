// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDECOMPOSEMINREDUCTION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// True when the reduction touches a dim before the last two (tile C/R).
// Those go through the D2M outer-reduction path and must not be decomposed.
static bool isOuterReduction(MinOp op) {
  auto inputType = cast<RankedTensorType>(op.getInput().getType());
  int64_t rank = inputType.getRank();
  std::optional<ArrayAttr> dimArg = op.getDimArg();
  if (rank < 2 || !dimArg.has_value()) {
    return false;
  }
  for (Attribute dimAttr : *dimArg) {
    int64_t dim = cast<IntegerAttr>(dimAttr).getInt();
    if (dim < 0) {
      dim += rank;
    }
    if (dim < rank - 2) {
      return true;
    }
  }
  return false;
}

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
      if (isOuterReduction(op)) {
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
