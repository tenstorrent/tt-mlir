// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TransformerRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ConcatenateHeadsRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn ConcatenateHeadsRuleBook::getInputLayoutFilter() const {
  // ConcatenateHeads: cannot consume any sharded inputs.
  // https://github.com/tenstorrent/tt-mlir/issues/7145
  return layout_filter_utils::rejectAllSharded;
}

bool ConcatenateHeadsRuleBook::shouldExploreReshards() const { return false; }

OutputHints ConcatenateHeadsRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // ConcatenateHeads: non-sharded only.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// SDPARuleBook
//===----------------------------------------------------------------------===//

bool SDPARuleBook::shouldExploreReshards() const { return false; }

OutputHints SDPARuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  // SDPA decode: tt-metal requires K/V in DRAM, Q either
  // height-sharded or DRAM, output either height-sharded or DRAM
  // (GQA further restricts output to DRAM-only).  In practice the
  // only valid output is DRAM-interleaved via the NULL hint --
  // every sharded output hint is rejected. Use NULL hint only,
  // no fallbacks.
  // NLPConcatHeadsDecode: tt-metal requires height-sharded L1 input
  // (set by workaround pass) and always outputs DRAM-interleaved.
  // Probing sharded output configs crashes compute_output_specs.
  return layout_filter_utils::nullHintOnly();
}

//===----------------------------------------------------------------------===//
// SDPADecodeRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn SDPADecodeRuleBook::getInputLayoutFilter() const {
  // K, V, and cache tensors must be DRAM-interleaved.
  return layout_filter_utils::requireDRAMInterleaved;
}

//===----------------------------------------------------------------------===//
// RotaryEmbeddingRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn RotaryEmbeddingRuleBook::getInputLayoutFilter() const {
  return layout_filter_utils::allowOnlyShardingType(
      TensorMemoryLayout::HeightSharded);
}

bool RotaryEmbeddingRuleBook::shouldExploreReshards() const { return false; }

OutputHints RotaryEmbeddingRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  return layout_filter_utils::nullHintOnly();
}

//===----------------------------------------------------------------------===//
// SplitQKVRuleBook
//===----------------------------------------------------------------------===//

bool SplitQKVRuleBook::shouldExploreReshards() const { return false; }

OutputHints SplitQKVRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  // The sharded create_qkv_heads kernel (BLOCK_SHARDED input →
  // HEIGHT_SHARDED output) corrupts data when the sequence dimension
  // is non-tile-aligned (e.g. ViT sequence length 197).
  // Use NULL hint only to keep input/output DRAM-interleaved.
  // https://github.com/tenstorrent/tt-metal/issues/41526
  return layout_filter_utils::nullHintOnly();
}

} // namespace mlir::tt::ttnn
