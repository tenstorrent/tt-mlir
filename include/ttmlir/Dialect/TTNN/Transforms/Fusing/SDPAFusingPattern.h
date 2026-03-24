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

  // Q/K Extraction with Scale Handling
  std::pair<Value, std::optional<float>> extractTensorWithScale(Value v) const;
  bool extractQKWithScales(Value a, Value b, SDPAComponents &c) const;

  // Pattern Matching
  bool matchSoftmaxPath(Value v, SDPAComponents &c) const;
  bool matchScoreComputation(Value v, SDPAComponents &c) const;
  bool matchScoreChain(Value v, SDPAComponents &c) const;

  // Input Canonicalization
  static Value castToBF16IfNeeded(Value v, PatternRewriter &rewriter);
  static Value restoreElementTypeIfNeeded(Value v, Type elementType,
                                          PatternRewriter &rewriter);
  static Type getTargetElementType(Value v);
  std::pair<Value, Type> analyzeQ(Value v) const;
  std::tuple<Value, Type, bool> analyzeK(Value v) const;
  std::pair<Value, Type> analyzeV(Value v) const;
  Value prepareMask(Value v) const;
  void prepareInputsForSDPA(SDPAComponents &c, PatternRewriter &rewriter) const;

  // Key Un-transpose
  Value unTransposeKeyIfNeeded(Value query, Value key, Value value,
                               mlir::PatternRewriter &rewriter,
                               Location loc) const;

  // Rank-3 → Rank-4 unsqueezing for single-head attention patterns.
  static Value unsqueezeToRank4(Value v, PatternRewriter &rewriter,
                                Location loc);
  bool tryUnsqueezeInputs(SDPAComponents &c, PatternRewriter &rewriter) const;

  // Validation
  bool validateShapes(Value query, Value key, Value value) const;
  bool validateSemantics(const SDPAComponents &c) const;

  // Op Creation
  mlir::LogicalResult createSDPAOp(mlir::PatternRewriter &rewriter,
                                   SDPAComponents &c,
                                   bool squeezeOutput = false) const;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H
