// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Fuses Scaled Dot Product Attention from component ops into a single
// ScaledDotProductAttentionOp (prefill) or ScaledDotProductAttentionDecodeOp
// (decode).
//
// Matches:  matmul(softmax((Q @ K^T) * scale + mask), V)
// Produces: scaled_dot_product_attention(Q, K, V, mask, scale)
//           or scaled_dot_product_attention_decode(...) when q_seq == 1
class SDPAFusing : public mlir::OpRewritePattern<MatmulOp> {
public:
  SDPAFusing(mlir::MLIRContext *context,
             const FusionValidationConfig &validationConfig = {})
      : OpRewritePattern<MatmulOp>(context),
        validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;

private:
  FusionValidationConfig validationConfig;

  struct SDPAComponents;

  // Layout / Transpose Utilities
  static bool isTransposeOnLastTwoDims(ArrayRef<int64_t> perm);
  bool isKeyTransposed(Value key, Value query, Value value) const;

  // Constant Extraction
  std::optional<float> extractConstant(Value v) const;
  std::pair<Value, std::optional<float>>
  extractMultiplyWithConstant(Value v) const;

  // Pattern Matching
  bool matchSoftmaxPath(Value v, SDPAComponents &c) const;
  bool matchScoreComputation(Value v, SDPAComponents &c) const;
  bool matchScoreChain(Value v, SDPAComponents &c) const;

  // Input Canonicalization
  std::pair<Value, std::optional<float>> analyzeQ(Value v) const;
  std::tuple<Value, bool, std::optional<float>> analyzeK(Value v) const;
  Value analyzeV(Value v) const;
  bool prepareInputsForSDPA(SDPAComponents &c, PatternRewriter &rewriter) const;

  // Key Un-transpose
  Value unTransposeKeyIfNeeded(Value query, Value key, Value value,
                               mlir::PatternRewriter &rewriter,
                               Location loc) const;

  // Validation
  bool validateShapes(Value query, Value key, Value value) const;
  bool validateSemantics(const SDPAComponents &c) const;

  // Op Creation
  mlir::LogicalResult createSDPAOp(mlir::PatternRewriter &rewriter,
                                   SDPAComponents &c) const;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H
