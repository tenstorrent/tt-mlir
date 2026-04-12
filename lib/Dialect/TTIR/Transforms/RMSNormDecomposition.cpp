// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRRMSNORMDECOMPOSITION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

static void decomposeRMSNorm(RMSNormOp op, IRRewriter &rewriter) {
  Location loc = op.getLoc();
  Value x = op.getInput();
  auto inputType = cast<RankedTensorType>(x.getType());
  int64_t rank = inputType.getRank();

  auto xSquared = MultiplyOp::create(rewriter, loc, inputType, x, x);

  // `normalized_shape` lists the trailing k input dims over which RMS is taken
  // (see RMSNormOp::verify). Mean of x^2 must run over all of them.
  ArrayRef<int64_t> normalizedShape = op.getNormalizedShape();
  const int64_t normRank = static_cast<int64_t>(normalizedShape.size());

  SmallVector<int64_t> reducedShape(inputType.getShape());
  for (int64_t i = 0; i < normRank; ++i) {
    reducedShape[rank - normRank + i] = 1;
  }
  auto reducedType = RankedTensorType::get(
      reducedShape, inputType.getElementType(), inputType.getEncoding());

  SmallVector<int32_t> reduceDims;
  reduceDims.reserve(normRank);
  for (int64_t i = 0; i < normRank; ++i) {
    reduceDims.push_back(static_cast<int32_t>(rank - normRank + i));
  }

  auto meanOp = MeanOp::create(rewriter, loc, reducedType, xSquared.getResult(),
                               rewriter.getBoolAttr(true),
                               rewriter.getI32ArrayAttr(reduceDims));

  float epsilon = op.getEpsilon().convertToFloat();
  auto epsOp = FullOp::create(rewriter, loc, reducedType,
                              rewriter.getF32FloatAttr(epsilon));

  auto addEps = AddOp::create(rewriter, loc, reducedType, meanOp.getResult(),
                              epsOp.getResult());

  auto rsqrt = RsqrtOp::create(rewriter, loc, reducedType, addEps.getResult());

  auto normalized =
      MultiplyOp::create(rewriter, loc, inputType, x, rsqrt.getResult());

  Value result = normalized.getResult();

  auto reshapeToInputRank = [&](Value v) -> Value {
    auto vType = cast<RankedTensorType>(v.getType());
    if (vType.getRank() == rank) {
      return v;
    }
    SmallVector<int64_t> newShape(rank - vType.getRank(), 1);
    newShape.append(vType.getShape().begin(), vType.getShape().end());
    auto reshapedType = RankedTensorType::get(newShape, vType.getElementType(),
                                              vType.getEncoding());
    SmallVector<int32_t> shapeI32(newShape.begin(), newShape.end());
    return ReshapeOp::create(rewriter, loc, reshapedType, v,
                             rewriter.getI32ArrayAttr(shapeI32));
  };

  if (op.getWeight()) {
    Value weight = reshapeToInputRank(op.getWeight());
    result = MultiplyOp::create(rewriter, loc, inputType, result, weight)
                 .getResult();
  }

  if (op.getBias()) {
    Value bias = reshapeToInputRank(op.getBias());
    result = AddOp::create(rewriter, loc, inputType, result, bias).getResult();
  }

  rewriter.replaceOp(op, result);
}

class TTIRRMSNormDecomposition
    : public impl::TTIRRMSNormDecompositionBase<TTIRRMSNormDecomposition> {
public:
  using impl::TTIRRMSNormDecompositionBase<
      TTIRRMSNormDecomposition>::TTIRRMSNormDecompositionBase;

  void runOnOperation() final {
    llvm::SmallVector<RMSNormOp> opsToDecompose;
    getOperation()->walk([&](RMSNormOp op) { opsToDecompose.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (RMSNormOp op : opsToDecompose) {
      rewriter.setInsertionPoint(op);
      decomposeRMSNorm(op, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
