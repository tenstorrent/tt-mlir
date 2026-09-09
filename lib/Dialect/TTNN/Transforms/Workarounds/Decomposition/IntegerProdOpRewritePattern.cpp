// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/IntegerProdOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static constexpr int64_t kMaxIntegerProdUnrollSize = 16;

LogicalResult
IntegerProdOpRewritePattern::matchAndRewrite(ttnn::ProdOp op,
                                             PatternRewriter &rewriter) const {
  // Only integer inputs hit the bf16 rounding bug; float prod is already
  // exact enough through the metal kernel.
  RankedTensorType inputType = op.getInput().getType();
  Type elemType = inputType.getElementType();
  if (!mlir::isa<IntegerType>(elemType)) {
    return failure();
  }

  // The integer-prod bug doesn't surface on the full-tensor reduction path in
  // practice, so we don't bother with a separate decomposition for it.
  std::optional<int64_t> dimArg = op.getDimArg();
  if (!dimArg) {
    return failure();
  }

  int64_t inputRank = inputType.getRank();
  int64_t reduceDim = *dimArg;
  if (reduceDim < 0) {
    reduceDim += inputRank;
  }
  if (reduceDim < 0 || reduceDim >= inputRank) {
    return failure();
  }

  // Dynamic / non-positive sizes can't be unrolled statically.
  ArrayRef<int64_t> inShape = inputType.getShape();
  int64_t reduceSize = inShape[reduceDim];
  if (reduceSize == ShapedType::kDynamic || reduceSize <= 0 ||
      reduceSize > kMaxIntegerProdUnrollSize) {
    return failure();
  }

  Location loc = op.getLoc();

  // Reusing the size-1 slice shape as the type of every intermediate keeps the
  // multiply chain shape-stable, so we only reshape once at the end.
  SmallVector<int64_t> sliceShape(inShape.begin(), inShape.end());
  sliceShape[reduceDim] = 1;
  RankedTensorType sliceType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, sliceShape);

  SmallVector<int32_t> stepAttr(inputRank, 1);
  auto makeSlice = [&](int64_t i) -> Value {
    SmallVector<int32_t> begins(inputRank, 0);
    SmallVector<int32_t> ends(inputRank);
    for (int64_t d = 0; d < inputRank; ++d) {
      ends[d] = static_cast<int32_t>(inShape[d]);
    }
    begins[reduceDim] = static_cast<int32_t>(i);
    ends[reduceDim] = static_cast<int32_t>(i + 1);
    return rewriter
        .create<ttnn::SliceStaticOp>(
            ttmlir::utils::appendLocationSuffix(loc, "_int_prod_slice_" +
                                                         std::to_string(i)),
            sliceType, op.getInput(), rewriter.getI32ArrayAttr(begins),
            rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(stepAttr))
        .getResult();
  };

  // A balanced tree would give shorter dependency chains, but at N <= 16 the
  // extra IR machinery isn't worth it.
  Value running = makeSlice(0);
  for (int64_t i = 1; i < reduceSize; ++i) {
    Value next = makeSlice(i);
    running = rewriter
                  .create<ttnn::MultiplyOp>(
                      ttmlir::utils::appendLocationSuffix(
                          loc, "_int_prod_mul_" + std::to_string(i)),
                      sliceType, running, next)
                  .getResult();
  }

  // `running` already has the reduce dim as size 1, so dropping it is a pure
  // metadata reshape with no data movement.
  if (!op.getKeepDim()) {
    RankedTensorType outType = op.getResult().getType();
    SmallVector<int32_t> outShapeI32;
    outShapeI32.reserve(outType.getRank());
    for (int64_t d : outType.getShape()) {
      outShapeI32.push_back(static_cast<int32_t>(d));
    }
    running =
        rewriter
            .create<ttnn::ReshapeOp>(
                ttmlir::utils::appendLocationSuffix(loc, "_int_prod_reshape"),
                outType, running, rewriter.getI32ArrayAttr(outShapeI32))
            .getResult();
  }

  rewriter.replaceOp(op, running);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
