// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallPtrSet.h"

// Workaround which adds padding to RotaryEmbedding seq_len
// dimension to make it a multiple of tile size.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/31567
namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Returns true if `v` (or any value reachable through tile-preserving
// forwarding ops like `ttnn.slice_static`) is consumed by a prefill-side cache
// write op (`ttnn.fill_cache` / `ttnn.paged_fill_cache`).
//
// This is intentionally scoped to *prefill* cache writes. Decode-side
// `ttnn.update_cache` / `ttnn.paged_update_cache` perform read-modify-write
// on a single row at `update_idx % TILE_HEIGHT` and never read the input's
// implicit tile-pad rows, so they don't need the scrub.
//
// If the check returns true, the implicit tile-pad rows of the rotary input
// must be scrubbed to zero so that the rotary kernel — which iterates over
// padded_shape — does not propagate garbage bytes (e.g. ±Inf / NaN) through
// to fill_cache, which itself iterates over padded_shape and writes them
// verbatim into the cache (see tt-metal#42779).
bool transitivelyFeedsPrefillCacheWrite(Value v) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  llvm::SmallVector<Value, 8> worklist;
  worklist.push_back(v);
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second) {
        continue;
      }
      if (isa<FillCacheOp, PagedFillCacheOp>(user)) {
        return true;
      }
      // Walk through ops that only re-window or re-label tile-pad rows.
      // slice_static carries the producer's tile bytes through to its
      // consumers, so if it feeds a fill_cache the scrub is still needed.
      if (isa<SliceStaticOp>(user)) {
        for (Value result : user->getResults()) {
          worklist.push_back(result);
        }
      }
    }
  }
  return false;
}

} // namespace

std::optional<std::pair<RotaryEmbeddingOp, SliceStaticOp>>
getWorkaroundedOp(RotaryEmbeddingOp ropeOp, PatternRewriter &rewriter) {
  RankedTensorType resultType = ropeOp.getType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape.size() < 2) {
    return std::nullopt;
  }

  int64_t originalSeqLen = resultShape[resultShape.size() - 2];
  if (originalSeqLen % TILE_HEIGHT == 0) {
    return std::nullopt;
  }

  SmallVector<int64_t> paddedResultShape(resultShape);
  paddedResultShape[paddedResultShape.size() - 2] =
      llvm::divideCeil(originalSeqLen, TILE_HEIGHT) * TILE_HEIGHT;

  auto paddedType =
      utils::RankedTensorTypeFactory::create(resultType, paddedResultShape);

  // If the rotary output transitively feeds a prefill-side cache write
  // (fill_cache / paged_fill_cache), the rotary input's implicit tile-pad
  // rows (which may contain ±Inf / NaN from upstream typecasts) would be
  // processed by the per-row rotary kernel into garbage output rows, then
  // copied through the per-user slice chain into the KV cache. SDPA later
  // reads those positions and (via the causal mask's 0 * Inf path) can leak
  // NaNs into valid output lanes — the `lane % 4 == 3` PCC corruption seen
  // in Llama prefill.
  //
  // To prevent this, replace the rotary's input with a PadOp that explicitly
  // zero-fills rows [S_logical, S_padded) in the seq_len dim. The rotary then
  // produces output rows [S_logical, S_padded) = cos*0 + sin*rotate(0) = 0,
  // which flow through downstream slices into a zero-clean KV cache.
  //
  // Decode-side update_cache / paged_update_cache do read-modify-write on a
  // single row and don't need this scrub; the Q rotary path (no cache write)
  // is also unaffected.
  //
  // Once https://github.com/tenstorrent/tt-metal/issues/42779 is fixed in the
  // fill_cache kernel (partial-tile write preserving rows
  // [S_logical, S_padded) of the cache), this scrub becomes unnecessary.
  Value rotaryInput = ropeOp.getInput();
  if (transitivelyFeedsPrefillCacheWrite(ropeOp.getResult())) {
    auto inputType = mlir::cast<RankedTensorType>(rotaryInput.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t> paddedInputShape(inputShape);
    int64_t seqDim = static_cast<int64_t>(inputShape.size()) - 2;
    paddedInputShape[seqDim] = paddedResultShape[seqDim];

    auto paddedInputType =
        utils::RankedTensorTypeFactory::create(inputType, paddedInputShape);

    SmallVector<int32_t> padding(2 * inputShape.size(), 0);
    padding[2 * seqDim + 1] =
        static_cast<int32_t>(paddedInputShape[seqDim] - inputShape[seqDim]);

    rotaryInput = rewriter.create<ttnn::PadOp>(
        ttmlir::utils::appendLocationSuffix(ropeOp.getLoc(),
                                            "_scrub_implicit_pad"),
        paddedInputType, rotaryInput, padding,
        /*pad_value=*/mlir::APFloat(0.0f),
        /*use_multicore=*/true);
  }

  auto paddedOp = rewriter.create<RotaryEmbeddingOp>(
      ropeOp.getLoc(), paddedType, rotaryInput, ropeOp.getCosCache(),
      ropeOp.getSinCache(), ropeOp.getTokenIndexAttr(),
      ropeOp.getComputeConfigAttr());

  // Slice to original shape.
  SmallVector<int32_t> begins(resultShape.size(), 0);
  SmallVector<int32_t> ends(paddedResultShape.begin(), paddedResultShape.end());
  SmallVector<int32_t> steps(resultShape.size(), 1);
  ends[ends.size() - 2] = originalSeqLen;

  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      ropeOp.getLoc(), resultType, paddedOp.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

  return std::make_pair(paddedOp, sliceOp);
}

LogicalResult RotaryEmbeddingOpRewritePattern::matchAndRewrite(
    RotaryEmbeddingOp srcOp, PatternRewriter &rewriter) const {
  auto workaround = getWorkaroundedOp(srcOp, rewriter);
  if (!workaround) {
    return failure();
  }

  auto [paddedOp, sliceOp] = *workaround;
  rewriter.replaceOp(srcOp, sliceOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
