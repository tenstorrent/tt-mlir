// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Transformer ops: SDPA, PagedSDPA, NLPConcatHeadsDecode, ConcatenateHeads,
//                  (Paged)UpdateCache, (Paged)FillCache
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
/// Per-operand input layout filtering.
/// - Q (operand 0): DRAM (any) or L1-sharded -- L1-interleaved rejected
///   ("Q tensor buffer type must be DRAM when not sharded").
/// - K, V, and cache tensors (operand >= 1): DRAM-interleaved only.
/// The OpModel bypasses tt-metal's input validation because mock tensors in
/// NO_DISPATCH mode lack real device buffers, so without these filters the
/// optimizer can assign V=L1-block-sharded (TTNN verifier fails with
/// keyType != valueType) or Q=L1-interleaved (kernel-side assert).
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

/// PagedUpdateCache constraint: the fill-value (operand 1) must be L1
/// height-sharded.
struct PagedUpdateCacheRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
};

/// FillCache / PagedFillCache constraint: the cache buffer (operand 0) is
/// modified in-place and must remain in DRAM interleaved storage. If the
/// beam search picks an L1 layout, the optimizer inserts a to_memory_config
/// that copies the cache into a temporary L1 buffer; the in-place fill writes
/// then go to that scratch copy and are silently discarded, leaving the real
/// DRAM cache uninitialized.
struct FillCacheRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
};

struct PagedFillCacheRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H
