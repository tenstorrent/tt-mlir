// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Transformer ops: SDPA, PagedSDPA, NLPConcatHeadsDecode, ConcatenateHeads
//
// Output hints:
//   SDPA/PagedSDPA/NLPConcatHeadsDecode: NULL hint only.
//     tt-metal requires K/V in DRAM, output DRAM-interleaved.
//     NLPConcatHeadsDecode: probing sharded output crashes
//     compute_output_specs.
//   ConcatenateHeads: non-sharded only.
//
// Input layout restrictions:
//   ConcatenateHeads: reject all sharded inputs.
//     https://github.com/tenstorrent/tt-mlir/issues/7145
//
// Reshard exploration: disabled for all transformer ops.
//===----------------------------------------------------------------------===//

/// ConcatenateHeads: reject all sharded inputs, non-sharded output, no
/// reshards.
struct ConcatenateHeadsRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// SDPA decode / NLPConcatHeadsDecode / ScaledDotProductAttention:
/// NULL hint only, no reshards.
struct SDPARuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// ScaledDotProductAttentionDecodeOp / PagedScaledDotProductAttentionDecodeOp:
/// Like SDPARuleBook, but also rejects sharded inputs.
/// tt-metal sdpa_decode_device_operation validates that K and V (and all
/// inputs after Q) reside in DRAM.  The OpModel bypasses this check because
/// mock tensors in NO_DISPATCH mode lack real device buffers, so without an
/// explicit input layout filter the optimizer can assign V=L1-block-sharded,
/// which then fails the TTNN verifier (keyType != valueType).
struct SDPADecodeRuleBook : SDPARuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
};

/// RotaryEmbedding / RotaryEmbeddingLlama:
/// NULL hint only, no reshards. Rejects width-sharded and block-sharded
/// inputs (only height-sharded or interleaved accepted).
/// Cache tensors are DRAM-interleaved; resharding them is wasteful.
struct RotaryEmbeddingRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// SplitQueryKeyValueAndSplitHeads: NULL hint only, no reshards.
/// The sharded create_qkv_heads kernel (BLOCK_SHARDED → HEIGHT_SHARDED)
/// corrupts data when the sequence dimension is non-tile-aligned (e.g. 197).
/// https://github.com/tenstorrent/tt-metal/issues/41526
struct SplitQKVRuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H
