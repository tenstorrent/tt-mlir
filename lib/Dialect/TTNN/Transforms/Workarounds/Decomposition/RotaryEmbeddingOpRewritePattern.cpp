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

// Right-pad `v`'s seq_len dim (rank-2 from the end) with zeros up to
// `paddedSeqLen`. Returns the padded SSA value. If `v`'s seq_len is
// already >= paddedSeqLen this is a no-op (returns `v` unchanged); the
// kernel reads only the first paddedSeqLen rows in that case so the
// implicit tile-pad rows above paddedSeqLen are not read.
Value zeroPadSeqLen(Value v, int64_t paddedSeqLen, Location loc,
                    StringRef locSuffix, PatternRewriter &rewriter) {
  auto type = mlir::cast<RankedTensorType>(v.getType());
  ArrayRef<int64_t> shape = type.getShape();
  int64_t seqDim = static_cast<int64_t>(shape.size()) - 2;
  if (shape[seqDim] >= paddedSeqLen) {
    return v;
  }

  SmallVector<int64_t> paddedShape(shape);
  paddedShape[seqDim] = paddedSeqLen;
  auto paddedType = utils::RankedTensorTypeFactory::create(type, paddedShape);

  SmallVector<int32_t> padding(2 * shape.size(), 0);
  padding[2 * seqDim + 1] =
      static_cast<int32_t>(paddedSeqLen - shape[seqDim]);

  return rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(loc, locSuffix), paddedType, v,
      padding, /*pad_value=*/mlir::APFloat(0.0f),
      /*use_multicore=*/true);
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
  // To prevent this, explicitly zero-pad the rotary's three operands
  // (input, cos, sin) along seq_len up to S_padded before invoking the
  // rotary kernel. The kernel asserts cos_seq_len >= input_seq_len, so all
  // three must be padded together (otherwise the cache cos/sin keep their
  // original S_logical and the kernel rejects the larger input). For rows
  // [S_logical, S_padded) the padded input/cos/sin are all 0, so the rotary
  // produces clean zero output rows that flow through the downstream
  // per-user slice chain into a zero-clean KV cache.
  //
  // Decode-side update_cache / paged_update_cache do read-modify-write on a
  // single row and don't need this scrub; the Q rotary path (no cache write)
  // is also unaffected.
  //
  // Once https://github.com/tenstorrent/tt-metal/issues/42779 is fixed in the
  // fill_cache kernel (partial-tile write preserving rows
  // [S_logical, S_padded) of the cache), this scrub becomes unnecessary.
  Value rotaryInput = ropeOp.getInput();
  Value cosCache = ropeOp.getCosCache();
  Value sinCache = ropeOp.getSinCache();
  int64_t paddedSeqLen = paddedResultShape[paddedResultShape.size() - 2];
  if (transitivelyFeedsPrefillCacheWrite(ropeOp.getResult())) {
    rotaryInput = zeroPadSeqLen(rotaryInput, paddedSeqLen, ropeOp.getLoc(),
                                "_scrub_input_pad", rewriter);
    cosCache = zeroPadSeqLen(cosCache, paddedSeqLen, ropeOp.getLoc(),
                             "_scrub_cos_pad", rewriter);
    sinCache = zeroPadSeqLen(sinCache, paddedSeqLen, ropeOp.getLoc(),
                             "_scrub_sin_pad", rewriter);
  }

  auto paddedOp = rewriter.create<RotaryEmbeddingOp>(
      ropeOp.getLoc(), paddedType, rotaryInput, cosCache, sinCache,
      ropeOp.getTokenIndexAttr(), ropeOp.getComputeConfigAttr());

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
