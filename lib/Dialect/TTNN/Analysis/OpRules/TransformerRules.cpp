// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TransformerRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ConcatenateHeadsRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn
ConcatenateHeadsRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
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

LayoutFilterFn
SDPADecodeRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  // Q (operand 0): kernel asserts "Q tensor buffer type must be DRAM when
  // not sharded" -- accept DRAM (any) or L1-sharded; reject L1-interleaved.
  // K, V, and cache tensors (operand >= 1): must be DRAM-interleaved.
  if (operandIdx == 0) {
    return layout_filter_utils::rejectL1Interleaved;
  }
  return layout_filter_utils::requireDRAMInterleaved;
}

//===----------------------------------------------------------------------===//
// RotaryEmbeddingRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn
RotaryEmbeddingRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
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

//===----------------------------------------------------------------------===//
// PagedUpdateCacheRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn
PagedUpdateCacheRuleBook::getInputLayoutFilter(RankedTensorType inputType,
                                               unsigned operandIdx) const {
  // Operand 1 (fill value) must be L1 height-sharded with num_cores ==
  // numUsers (= input1.shape[1]); other grids silently produced PCC=0 for
  // upper users (https://github.com/tenstorrent/tt-metal/issues/44923).
  //
  // tt-metal PR #45016 enforces this with a TT_FATAL
  // (`grid.num_cores() == padded_shape()[1]`).  The kernel flattens the
  // grid via corerange_to_cores(row_major) and assigns cores[i] to user
  // i, so any single-axis grid of the right size is valid -- the
  // canonical shape is {numUsers, 1}, but {1, numUsers} (and any other 2D
  // shape whose product equals numUsers) is assumed to work with correct
  // PCC.  We mirror that invariant here: gridShape[0] * gridShape[1] ==
  // numUsers.
  //
  // TODO(bmalesevic): Once PR #45016 is uplifted, tripwire will fail if the
  // filter is violated, so we can remove these rules.
  // test/unittests/OpModel/TTNN/Lib/TestOpModelLibTripwires.cpp.
  if (operandIdx == 1) {
    int64_t numUsers = inputType.getShape()[1];
    return [numUsers](TTNNLayoutAttr layout) -> bool {
      auto ml = layout.getMemLayout();
      assert(ml && "expected non-null memory layout");
      auto gridShape = layout.getGridShape();
      assert(gridShape.size() == 2 && "expected 2D grid shape");
      return layout.hasL1BufferType() &&
             ml.getValue() == TensorMemoryLayout::HeightSharded &&
             gridShape[0] * gridShape[1] == numUsers;
    };
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// FillCache / PagedFillCache: cache buffer (operand 0) must stay in DRAM
//===----------------------------------------------------------------------===//

// Cache buffer: DRAM interleaved only.  These are in-place ops; the cache
// modifications must land in the actual DRAM-resident KV cache, not in a
// temporary L1 scratch copy that the optimizer would insert if it picks an
// L1 layout.
static LayoutFilterFn cacheBufferDramOnlyFilter() {
  return [](TTNNLayoutAttr layout) -> bool {
    auto ml = layout.getMemLayout();
    return layout.hasDRAMBufferType() && ml &&
           ml.getValue() == TensorMemoryLayout::Interleaved;
  };
}

LayoutFilterFn
FillCacheRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  if (operandIdx == 0) {
    return cacheBufferDramOnlyFilter();
  }
  return nullptr;
}

LayoutFilterFn
PagedFillCacheRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  if (operandIdx == 0) {
    return cacheBufferDramOnlyFilter();
  }
  return nullptr;
}

} // namespace mlir::tt::ttnn
