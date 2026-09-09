// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_FILLCACHEINPUTPADREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_FILLCACHEINPUTPADREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which zero-pads the cache-write input's seq_len (dim -2) to
// the next tile multiple before `ttnn.fill_cache` / `ttnn.paged_fill_cache`.
// Without this, the kernel iterates over the input's padded shape and
// copies whatever (potentially +/-Inf / NaN) sits in the implicit tile-pad
// rows verbatim into the KV cache, later leaking NaNs into SDPA via the
// causal mask's `0 * Inf` path.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/42779
//
// Templated on the cache-write op; instantiated for `FillCacheOp` and
// `PagedFillCacheOp`, which share an `$input` operand with identical shape
// semantics (4D, dim -2 = seq_len).
template <typename CacheWriteOpT>
class FillCacheInputPadRewritePattern : public OpRewritePattern<CacheWriteOpT> {
public:
  using OpRewritePattern<CacheWriteOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(CacheWriteOpT op,
                                PatternRewriter &rewriter) const override {
    auto type = mlir::dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!type || type.getRank() < 2) {
      return failure();
    }
    ArrayRef<int64_t> shape = type.getShape();
    int64_t seqDim = static_cast<int64_t>(shape.size()) - 2;
    int64_t seqLen = shape[seqDim];
    if (seqLen % TILE_HEIGHT == 0) {
      return failure();
    }
    int64_t paddedSeqLen = llvm::divideCeil(seqLen, TILE_HEIGHT) * TILE_HEIGHT;

    // `FillCacheOp` requires input.dim(-2) <= cache.dim(-2); skip when
    // padding would exceed cache.dim(-2). `PagedFillCacheOp` is unconstrained
    // here -- its cache.dim(-2) is page-block size, not seq_len.
    if constexpr (std::is_same_v<CacheWriteOpT, ttnn::FillCacheOp>) {
      auto cacheShape =
          mlir::cast<RankedTensorType>(op.getCache().getType()).getShape();
      if (paddedSeqLen > cacheShape[seqDim]) {
        return failure();
      }
    }

    SmallVector<int64_t> paddedShape(shape);
    paddedShape[seqDim] = paddedSeqLen;
    SmallVector<int32_t> padding(2 * shape.size(), 0);
    padding[2 * seqDim + 1] = static_cast<int32_t>(paddedSeqLen - seqLen);

    Value paddedInput = rewriter.create<ttnn::PadOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_scrub_input_pad"),
        utils::RankedTensorTypeFactory::create(type, paddedShape),
        op.getInput(), padding, /*pad_value=*/mlir::APFloat(0.0f),
        /*use_multicore=*/true);

    rewriter.modifyOpInPlace(
        op, [&]() { op.getInputMutable().assign(paddedInput); });
    return success();
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_FILLCACHEINPUTPADREWRITEPATTERN_H
