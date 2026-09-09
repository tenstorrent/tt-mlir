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

/// TTML SDPA forward:
/// The TTML kernel requires tiled, interleaved inputs; filtering them and
/// disabling reshards prune unsupported candidates rather than tune
/// performance. The backend derives both output layouts from the query, so
/// only the NULL output hint is valid.
struct TTMLSDPAForwardRuleBook : SDPARuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
};

/// TTML SDPA backward:
/// The TTML kernels require tiled inputs but support both interleaved and
/// sharded memory. Grad-Q/K/V inherit the Q/K/V memory configs, so only the
/// NULL output hint is valid. Reshard exploration is disabled solely to bound
/// the seven-operand search, at the cost of not optimizing input sharding.
struct TTMLSDPABackwardRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// TTML layer norm forward:
/// All operands must be tiled and DRAM-interleaved. The backend derives the
/// output and optional statistics layouts from the input.
struct TTMLLayerNormForwardRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
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

/// PagedUpdateCache operand contract: cache (0) DRAM-interleaved, value (1) L1
/// HeightSharded (via reshards), update_idxs (2) / page_table (3) ROW_MAJOR
/// (via RowMajor input siblings). The optimizer reaches these at opt >= 2,
/// where the paged_update_cache workaround is gated off.
struct PagedUpdateCacheRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool generatesRowMajorInputSiblings(unsigned operandIdx) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_TRANSFORMERRULES_H
