// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSOFTMAXDECOMPOSITION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// softmax(x, dim) = exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)
static void decomposeSoftmax(SoftmaxOp op, IRRewriter &rewriter) {
  Location loc = op.getLoc();
  Value input = op.getInput();
  auto inputType = cast<RankedTensorType>(input.getType());
  int32_t dim = op.getDimension();

  // Reduced shape: same as input but with dim collapsed to 1.
  SmallVector<int64_t> reducedShape(inputType.getShape());
  reducedShape[dim] = 1;
  auto reducedType = RankedTensorType::get(
      reducedShape, inputType.getElementType(), inputType.getEncoding());

  auto dimAttr = rewriter.getI32ArrayAttr({dim});

  // m = max(x, dim, keep_dim=true)
  auto maxVal = rewriter.create<MaxOp>(loc, reducedType, input,
                                       rewriter.getBoolAttr(true), dimAttr);

  // x_shifted = x - m (broadcasts along reduced dim)
  auto shifted = rewriter.create<SubtractOp>(loc, inputType, input,
                                             maxVal.getResult());

  // e = exp(x_shifted)
  auto expVal = rewriter.create<ExpOp>(loc, inputType, shifted.getResult());

  // s = sum(e, dim, keep_dim=true)
  auto sumVal = rewriter.create<SumOp>(loc, reducedType, expVal.getResult(),
                                       rewriter.getBoolAttr(true), dimAttr);

  // 1/s
  auto recipSum =
      rewriter.create<ReciprocalOp>(loc, reducedType, sumVal.getResult());

  // result = e * (1/s) (broadcasts along reduced dim)
  auto result = rewriter.create<MultiplyOp>(loc, inputType, expVal.getResult(),
                                            recipSum.getResult());

  rewriter.replaceOp(op, result.getResult());
}

class TTIRSoftmaxDecomposition
    : public impl::TTIRSoftmaxDecompositionBase<TTIRSoftmaxDecomposition> {
public:
  using impl::TTIRSoftmaxDecompositionBase<
      TTIRSoftmaxDecomposition>::TTIRSoftmaxDecompositionBase;

  void runOnOperation() final {
    llvm::SmallVector<SoftmaxOp> opsToDecompose;
    getOperation()->walk(
        [&](SoftmaxOp op) { opsToDecompose.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (SoftmaxOp op : opsToDecompose) {
      rewriter.setInsertionPoint(op);
      decomposeSoftmax(op, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
