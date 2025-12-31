// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Fuses a matmul followed by slices and reshapes into a single
// SplitQueryKeyValueAndSplitHeadsOp. Detects the common pattern where
// a fused QKV projection (matmul) is followed by three slices to separate
// Q, K, V components, each followed by a reshape to split heads.
//
// Pattern matched:
//   matmul -> slice[0:q_size] -> reshape[B,H,S,D] (Q)
//          -> slice[q_size:q_size+k_size] -> reshape[B,H,S,D] (K)
//          -> slice[q_size+k_size:total] -> reshape[B,H,S,D] (V)
//
// Replaced with:
//   matmul -> reshape[B,S,total] -> SplitQueryKeyValueAndSplitHeadsOp -> Q, K,
//   V
template <typename MatMulOpType>
class SplitQueryKeyValueAndSplitHeadsFusing
    : public mlir::OpRewritePattern<MatMulOpType> {
public:
  using mlir::OpRewritePattern<MatMulOpType>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatMulOpType matmulOp,
                  mlir::PatternRewriter &rewriter) const final;
};

// Fuses a SplitQueryKeyValueAndSplitHeadsOp followed by permute [2,0,1,3]
// on all three outputs into a single NLPCreateQKVHeadsDecodeOp, which is
// optimized for the decode case (S=1).
//
// Pattern matched:
//   split_query_key_value_and_split_heads [B,1,hidden]
//     -> Q[B,H,1,D] -> permute[2,0,1,3] -> [1,B,H,D]
//     -> K[B,Hkv,1,D] -> permute[2,0,1,3] -> [1,B,Hkv,D]
//     -> V[B,Hkv,1,D] -> permute[2,0,1,3] -> [1,B,Hkv,D]
//
// Replaced with:
//   reshape [B,1,hidden] -> [1,1,B,hidden]
//     -> nlp_create_qkv_heads_decode -> Q[1,B,H,D], K[1,B,Hkv,D],
//     V[1,B,Hkv,D]
class NLPCreateQKVHeadsDecodeFusing
    : public mlir::OpRewritePattern<SplitQueryKeyValueAndSplitHeadsOp> {
public:
  using OpRewritePattern<SplitQueryKeyValueAndSplitHeadsOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SplitQueryKeyValueAndSplitHeadsOp splitOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H
