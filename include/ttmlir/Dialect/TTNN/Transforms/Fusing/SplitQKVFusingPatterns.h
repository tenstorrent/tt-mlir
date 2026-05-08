// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

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
      const FusionValidationConfig &validationConfig = {})
      : mlir::OpRewritePattern<MatMulOpType>(context),
        validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(MatMulOpType matmulOp,
                  mlir::PatternRewriter &rewriter) const final;

private:
  FusionValidationConfig validationConfig;
};

// Cross-attention K/V concat fusion (Pilot 3.6 — fuse_kv_split).
//
// Self-attention has a single fused QKV matmul whose output is sliced into
// three equal portions; the existing SplitQueryKeyValueAndSplitHeadsFusing
// pattern collapses that. Cross-attention is structurally different — Q comes
// from the decoder hidden state and K, V come from the encoder hidden state,
// so K and V share an LHS but Q does not. The end-to-end IR has three
// SEPARATE matmuls/linears, each with a reshape -> permute chain feeding one
// SDPA operand:
//
//   Q = matmul(decoder_h, W_q) -> reshape -> permute -> SDPA.query
//   K = matmul(encoder_h, W_k) -> reshape -> permute -> SDPA.key
//   V = matmul(encoder_h, W_v) -> reshape -> permute -> SDPA.value
//
// where K.LHS == V.LHS == encoder_h and Q.LHS == decoder_h (Q.LHS != K.LHS).
//
// This pattern fuses the K and V matmul outputs into a single
// SplitQueryKeyValueAndSplitHeadsOp by:
//   - Reshaping Q matmul output to 3D [B, S_q, num_heads * head_dim] and
//     passing it as input_tensor.
//   - Concatenating K and V matmul outputs along the last dim to produce
//     [B, S_kv, 2 * num_kv_heads * head_dim] and passing it as
//     kv_input_tensor.
//
// The op-level kv_input_tensor operand has existed in TTNNOps.td since the
// op was introduced; this pattern just wires three sibling matmuls into it.
//
// Models affected: T5, BART, BLIP, Whisper, bge_m3 (encoder cross-attention).
template <typename MatMulOpType>
class CrossAttnSplitQKVFusing : public mlir::OpRewritePattern<MatMulOpType> {
public:
  CrossAttnSplitQKVFusing(mlir::MLIRContext *context,
                          const FusionValidationConfig &validationConfig = {})
      : mlir::OpRewritePattern<MatMulOpType>(context),
        validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(MatMulOpType matmulOp,
                  mlir::PatternRewriter &rewriter) const final;

private:
  FusionValidationConfig validationConfig;
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
  NLPCreateQKVHeadsDecodeFusing(
      mlir::MLIRContext *context,
      const FusionValidationConfig &validationConfig = {})
      : OpRewritePattern<SplitQueryKeyValueAndSplitHeadsOp>(context),
        validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(SplitQueryKeyValueAndSplitHeadsOp splitOp,
                  mlir::PatternRewriter &rewriter) const override;

private:
  FusionValidationConfig validationConfig;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVFUSINGPATTERNS_H
