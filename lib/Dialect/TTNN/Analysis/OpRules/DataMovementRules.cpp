// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ConcatRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn ConcatRuleBook::getInputLayoutFilter() const {
  // Concat: cannot consume any sharded inputs.
  // https://github.com/tenstorrent/tt-mlir/issues/7145
  return layout_filter_utils::rejectAllSharded;
}

bool ConcatRuleBook::shouldExploreReshards() const { return false; }

OutputHints ConcatRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // ConcatOp: sharded output causes device close hang in tt-metal.
  // https://github.com/tenstorrent/tt-metal/issues/39419
  // TODO(rpavlovicTT): re-enable sharded concat once tt-metal fixes it.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// SliceRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn SliceRuleBook::getInputLayoutFilter() const {
  // Slice: sharded inputs produce incorrect results in tt-metal.
  // https://github.com/tenstorrent/tt-metal/issues/39074
  return layout_filter_utils::rejectAllSharded;
}

bool SliceRuleBook::shouldExploreReshards() const { return false; }

OutputHints
SliceRuleBook::getOutputHints(Operation * /*op*/,
                              const std::vector<OpConfig> &legalConfigs) const {
  // SliceStaticOp, SliceDynamicOp: disabled due to
  // https://github.com/tenstorrent/tt-metal/issues/38016
  // TODO(rpavlovicTT): re-enable slice ops.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// ReshapeRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn ReshapeRuleBook::getInputLayoutFilter() const {
  // Reshape/Permute: width-sharded inputs round-trip through interleaved
  // internally in tt-metal; height/block sharding is handled natively.
  return layout_filter_utils::rejectWidthSharded;
}

bool ReshapeRuleBook::shouldExploreReshards() const { return false; }

OutputHints ReshapeRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // ReshapeOp, PermuteOp: width-sharded inputs round-trip through interleaved
  // internally; sharded output not beneficial.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// PadRuleBook
//===----------------------------------------------------------------------===//

bool PadRuleBook::shouldExploreReshards() const { return false; }

OutputHints
PadRuleBook::getOutputHints(Operation * /*op*/,
                            const std::vector<OpConfig> &legalConfigs) const {
  // PadOp: ttnn::pad only supports sharded output for HEIGHT_SHARDED +
  // ROW_MAJOR inputs. Its tile-layout path (invoke_tile) crashes when given a
  // sharded output memory config with interleaved input because it
  // unconditionally dereferences input_tensor.shard_spec()->grid, which is
  // nullopt for interleaved tensors. The op model validation doesn't catch
  // this since it only checks sharding constraints when the input is already
  // sharded.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

} // namespace mlir::tt::ttnn
