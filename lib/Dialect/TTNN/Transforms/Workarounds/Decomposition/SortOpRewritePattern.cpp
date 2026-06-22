// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SortOpRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Tile width for the O(n^2) rank-by-comparison: the inner reduction runs in
// blocks and accumulates, so peak memory is [R,n,kArgsortTile] not [R,n,n] --
// no length cap.
static constexpr int64_t kArgsortTile = 512;

// Every intermediate here changes shape and/or dtype, so each needs a fresh
// TTNN layout encoding derived from `base` (shape first, then dtype).
static RankedTensorType typedShape(RankedTensorType base,
                                   ArrayRef<int64_t> shape,
                                   ttcore::DataType dt) {
  RankedTensorType withShape =
      ttnn::utils::RankedTensorTypeFactory::create(base, shape);
  return ttnn::utils::RankedTensorTypeFactory::create(withShape, dt);
}

LogicalResult
SortOpRewritePattern::matchAndRewrite(ttnn::SortOp op,
                                      PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();

  // Float keys sort correctly; only integer keys hit the bf16-compare bug, and
  // only the indices output is wrong, so there is nothing to fix otherwise.
  if (!mlir::isa<IntegerType>(inputType.getElementType())) {
    return failure();
  }
  if (op.getIndices().use_empty()) {
    return failure();
  }

  int64_t rank = inputType.getRank();
  int64_t dim = op.getDim();
  if (dim < 0) {
    dim += rank;
  }
  // Last-dim only keeps the reshape-to-2D below trivial; other dims are rare
  // for integer argsort and fall back to the kernel.
  if (dim != rank - 1) {
    return failure();
  }

  ArrayRef<int64_t> inShape = inputType.getShape();
  int64_t n = inShape[dim];
  if (n <= 0) {
    return failure();
  }
  int64_t R = inputType.getNumElements() / n;

  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  bool descending = op.getDescending();
  const auto f32 = ttcore::DataType::Float32;

  RankedTensorType idxType = op.getIndices().getType();
  ttcore::DataType inDt =
      ttcore::elementTypeToDataType(inputType.getElementType());
  GetDeviceOp device = ttnn::utils::getOrInsertDevice(rewriter, op);

  // Small construction helpers; everything past the input typecast is f32.
  auto i32 = [&](ArrayRef<int64_t> v) {
    return rewriter.getI32ArrayAttr(SmallVector<int32_t>(v.begin(), v.end()));
  };
  auto reshape = [&](Value v, ArrayRef<int64_t> shape) -> Value {
    return rewriter.create<ttnn::ReshapeOp>(
        loc, typedShape(inputType, shape, f32), v, i32(shape));
  };
  auto repeat = [&](Value v, ArrayRef<int64_t> outShape,
                    ArrayRef<int64_t> factors) -> Value {
    return rewriter.create<ttnn::RepeatOp>(
        loc, typedShape(inputType, outShape, f32), v,
        ttnn::ShapeAttr::get(ctx, SmallVector<int64_t>(factors)));
  };
  auto arange = [&](int64_t start, int64_t end) -> Value {
    RankedTensorType t = typedShape(inputType, {end - start}, f32);
    ttnn::LayoutAttr lay = ttnn::LayoutAttr::get(
        ctx, mlir::cast<ttnn::TTNNLayoutAttr>(t.getEncoding()).getLayout());
    return rewriter.create<ttnn::ArangeOp>(loc, t, device, start, end,
                                           /*step=*/1, lay);
  };
  auto sliceCols = [&](Value v2d, int64_t start, int64_t len) -> Value {
    return rewriter.create<ttnn::SliceStaticOp>(
        loc, typedShape(inputType, {R, len}, f32), v2d, i32({0, start}),
        i32({R, start + len}), i32({1, 1}));
  };

  // f32 holds the integer keys exactly (<= 2^24), so comparisons never collide
  // like ttnn.sort's bf16 compare. Flatten batch dims to a 2D row.
  Value in2D = rewriter.create<ttnn::ReshapeOp>(
      loc, typedShape(inputType, {R, n}, inDt), op.getInput(), i32({R, n}));
  Value xf = rewriter.create<ttnn::TypecastOp>(
      loc, typedShape(inputType, {R, n}, f32), in2D);

  RankedTensorType t2f = typedShape(inputType, {R, n}, f32);
  Value xiCol = reshape(xf, {R, n, 1});           // x[i] down the rows
  Value iColN = reshape(arange(0, n), {1, n, 1}); // index along dim-1 (i or k)

  // rank[r,i] = #keys ordered before i (its sorted position), summed in
  // j-tiles.
  Value rank2D;
  for (int64_t jb = 0; jb < n; jb += kArgsortTile) {
    int64_t t = std::min(kArgsortTile, n - jb);
    SmallVector<int64_t, 3> blk{R, n, t};
    Value xiB = repeat(xiCol, blk, {1, 1, t}); // x[i]
    Value xjRow = reshape(sliceCols(xf, jb, t), {R, 1, t});
    Value xjB = repeat(xjRow, blk, {1, n, 1});   // x[jb + j]
    Value iIota = repeat(iColN, blk, {R, 1, t}); // i
    Value jRow = reshape(arange(jb, jb + t), {1, 1, t});
    Value jIota = repeat(jRow, blk, {R, n, 1}); // jb + j (global)

    RankedTensorType t3 = typedShape(inputType, blk, f32);
    Value cmp =
        descending
            ? rewriter.create<ttnn::GreaterThanOp>(loc, t3, xjB, xiB)
                  .getResult()
            : rewriter.create<ttnn::LessThanOp>(loc, t3, xjB, xiB).getResult();
    // Break ties by original index (j < i) so equal keys sort stably.
    Value eq = rewriter.create<ttnn::EqualOp>(loc, t3, xjB, xiB);
    Value jLtI = rewriter.create<ttnn::LessThanOp>(loc, t3, jIota, iIota);
    Value tie = rewriter.create<ttnn::MultiplyOp>(loc, t3, eq, jLtI);
    Value lessOrTie = rewriter.create<ttnn::MaximumOp>(loc, t3, cmp, tie);
    Value partial = rewriter.create<ttnn::SumOp>(loc, t2f, lessOrTie,
                                                 /*keep_dim=*/false, i32({2}));
    rank2D = rank2D ? rewriter.create<ttnn::AddOp>(loc, t2f, rank2D, partial)
                          .getResult()
                    : partial;
  }

  // Invert: argsort[r,k] = sum_i i * (rank[r,i] == k), summed in i-tiles.
  Value argF2D;
  for (int64_t ib = 0; ib < n; ib += kArgsortTile) {
    int64_t t = std::min(kArgsortTile, n - ib);
    SmallVector<int64_t, 3> blk{R, n, t};
    Value rankRow = reshape(sliceCols(rank2D, ib, t), {R, 1, t});
    Value rankB = repeat(rankRow, blk, {1, n, 1}); // rank[i] over k
    Value kIota = repeat(iColN, blk, {R, 1, t});   // k
    Value iRow = reshape(arange(ib, ib + t), {1, 1, t});
    Value iGlobal = repeat(iRow, blk, {R, n, 1}); // ib + i (global)

    RankedTensorType t3 = typedShape(inputType, blk, f32);
    Value onehot = rewriter.create<ttnn::EqualOp>(loc, t3, rankB, kIota);
    Value prod = rewriter.create<ttnn::MultiplyOp>(loc, t3, onehot, iGlobal);
    Value partial = rewriter.create<ttnn::SumOp>(loc, t2f, prod,
                                                 /*keep_dim=*/false, i32({2}));
    argF2D = argF2D ? rewriter.create<ttnn::AddOp>(loc, t2f, argF2D, partial)
                          .getResult()
                    : partial;
  }

  ttcore::DataType idxDt =
      ttcore::elementTypeToDataType(idxType.getElementType());
  Value argIdx2D = rewriter.create<ttnn::TypecastOp>(
      loc, typedShape(inputType, {R, n}, idxDt), argF2D);
  SmallVector<int32_t> outShapeI32(inShape.begin(), inShape.end());
  Value indicesND = rewriter.create<ttnn::ReshapeOp>(
      loc, idxType, argIdx2D, rewriter.getI32ArrayAttr(outShapeI32));

  // Values are correct from ttnn.sort, so keep it for them. Its own indices
  // are unused, so use_empty() above stops this pattern from re-matching it.
  auto valuesSort = rewriter.create<ttnn::SortOp>(
      loc, TypeRange{op.getValues().getType(), op.getIndices().getType()},
      op.getInput(), op.getDimAttr(), op.getDescendingAttr(),
      op.getStableAttr());

  rewriter.replaceOp(op, {valuesSort.getValues(), indicesND});
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
