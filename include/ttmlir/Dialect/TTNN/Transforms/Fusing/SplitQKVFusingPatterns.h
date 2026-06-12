// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

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
  SplitQueryKeyValueAndSplitHeadsFusing(
      mlir::MLIRContext *context,
      const OpValidationConfig &validationConfig = {})
      : mlir::OpRewritePattern<MatMulOpType>(context),
        validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(MatMulOpType matmulOp,
                  mlir::PatternRewriter &rewriter) const final;

private:
  OpValidationConfig validationConfig;
};

// Fuses a SplitQueryKeyValueAndSplitHeadsOp into a single
// NLPCreateQKVHeadsDecodeOp for the decode case (S=1), when each output reaches
// the decode layout change [B,H,1,D] -> [1,B,H,D].
//
// The layout change is accepted in either equivalent form:
//   * an explicit permute [2,0,1,3], or
//   * the equivalent reshape [B,H,1,D] -> [1,B,H,D] (pure relabel when S==1;
//     compilers often emit this instead of a permute).
//
// An output need not feed the layout change directly. It may first pass through
// an optional per-head QK-RMSNorm and an optional "rotate first `half`, pass the
// rest" partial-RoPE (slice/rotary/slice/concat), all matched purely by op kinds
// and shapes (no model-specific constants). Such a chain is rebuilt in the
// [1,B,H,D] decode layout after the fused op; the partial-RoPE's cos/sin caches
// are repeated across the (now heads-) sequence axis so the rotary op applies
// the single decode position to every head.
//
// Pattern matched (V shown direct, Q/K shown with the optional chain):
//   split_query_key_value_and_split_heads [B,1,hidden]
//     -> V[B,Hkv,1,D]               -> permute/reshape -> [1,B,Hkv,D]
//     -> Q[B,H,1,D]   -> rms_norm -> partial-RoPE -> reshape -> [1,B,H,D]
//     -> K[B,Hkv,1,D] -> rms_norm -> partial-RoPE -> reshape -> [1,B,Hkv,D]
//
// Replaced with:
//   reshape [B,1,hidden] -> [1,1,B,hidden]
//     -> nlp_create_qkv_heads_decode -> Q[1,B,H,D], K[1,B,Hkv,D], V[1,B,Hkv,D]
//     -> (rebuilt rms_norm + partial-RoPE in decode layout for Q/K)
class NLPCreateQKVHeadsDecodeFusing
    : public mlir::OpRewritePattern<SplitQueryKeyValueAndSplitHeadsOp> {
public:
  NLPCreateQKVHeadsDecodeFusing(mlir::MLIRContext *context,
                                const OpValidationConfig &validationConfig = {})
      : OpRewritePattern<SplitQueryKeyValueAndSplitHeadsOp>(context),
        validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(SplitQueryKeyValueAndSplitHeadsOp splitOp,
                  mlir::PatternRewriter &rewriter) const override;

private:
  OpValidationConfig validationConfig;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H
