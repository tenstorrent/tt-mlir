// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_RINGSDPAFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_RINGSDPAFUSINGPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Replaces an exposed sequence-parallel K/V all-gather feeding a plain SDPA
// with the ring variant, which streams the K/V blocks around the ring and
// folds each into the running attention as it arrives.
//
// Matches:  scaled_dot_product_attention(q,
//                                        all_gather(k, dim=D, cluster_axis=A),
//                                        all_gather(v, dim=D, cluster_axis=A))
// Produces: exp_ring_joint_scaled_dot_product_attention(q, k, v,
//                                                       dim=D, cluster_axis=A)
//           with the persistent buffers and semaphore pool left unbound for
//           the prelude passes to fill in.
//
// Unlike SDPAFusing, which builds an SDPA out of softmax(QK^T)V primitives and
// is therefore rooted on MatmulOp, this pattern rewrites an SDPA that already
// exists -- which is the form frontends produce, via
// `stablehlo.composite "tenstorrent.scaled_dot_product_attention"`.
class RingSDPAFusing
    : public mlir::OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  using mlir::OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ScaledDotProductAttentionOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;

private:
  // The all-gather feeding `v`, when it is a single-use all-gather that agrees
  // with `keyGather` on every CCL attribute. Null otherwise.
  static AllGatherOp matchPairedGather(Value v, AllGatherOp keyGather);

  // Peels an optional padding slice off `v`, returning the value behind it.
  //
  // Frontends that pad the sequence to `TILE_SIZE * SP` have to slice the
  // gathered K/V back to the true length before every block, because plain SDPA
  // cannot tell padded keys from real ones. The ring op takes `logical_n`
  // instead, so the slice becomes the carrier of the true length rather than
  // something to delete separately.
  //
  // Sets `slice` to the peeled op (null when there is none) and only accepts a
  // pure prefix on the sequence axis: `begins` all zero, unit `step`, and every
  // non-sequence dim full. Anything else is some other slice and is left alone.
  static Value peelPaddingSlice(Value v, int64_t seqDim, SliceStaticOp &slice);

  // True when two peeled padding slices describe the same trim.
  static bool slicesAgree(SliceStaticOp a, SliceStaticOp b);

  // Builds the program config the ring kernel requires. Unlike plain SDPA,
  // tt-metal takes this by value with no default, so the compiler has to
  // choose one.
  static SDPAProgramConfigAttr
  buildProgramConfig(ScaledDotProductAttentionOp srcOp, int64_t localSeqLen,
                     int64_t gatheredSeqLen);
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_RINGSDPAFUSINGPATTERN_H
