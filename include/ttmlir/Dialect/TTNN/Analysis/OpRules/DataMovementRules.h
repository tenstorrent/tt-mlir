// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_DATAMOVEMENTRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_DATAMOVEMENTRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Data movement ops: Reshape, Permute, Concat, Slice, Pad
//
// Input layout restrictions:
//   ConcatOp: reject all sharded inputs
//     https://github.com/tenstorrent/tt-mlir/issues/7145
//   SliceStaticOp, SliceDynamicOp: reject all sharded inputs
//     https://github.com/tenstorrent/tt-metal/issues/39074
//   ReshapeOp, PermuteOp: reject width-sharded inputs
//     (round-trip through interleaved internally in tt-metal)
//
// Output hints:
//   All: non-sharded only
//     PadOp: invoke_tile crashes with sharded output + interleaved input
//     https://github.com/tenstorrent/tt-metal/issues/40898
//     SliceOps: https://github.com/tenstorrent/tt-metal/issues/38016
//     ConcatOp: device close hang,
//     https://github.com/tenstorrent/tt-metal/issues/39419
//
// Reshard exploration: disabled for all data movement ops except ConcatOp.
//===----------------------------------------------------------------------===//

/// ConcatOp: sharded inputs/output re-enabled after tt-metal hang fix.
/// https://github.com/tenstorrent/tt-metal/pull/39882
struct ConcatRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  bool isValidInputCombination(
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// SliceStaticOp, SliceDynamicOp: allow HEIGHT_SHARDED ROW_MAJOR input for
/// width-trim slices (begins=[0,...,0], only last dim sliced); reject all
/// other sharded inputs. Non-sharded output by default; HEIGHT_SHARDED ROW_MAJOR
/// output proposed as fallback for width-trim SliceStaticOp. No reshards.
struct SliceRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// ReshapeOp, PermuteOp: reject width-sharded inputs, no reshards.
/// View-eligible reshapes use NULL hint only (inherit input memory config).
/// Non-view reshapes use non-sharded output hints.
struct ReshapeRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// PadOp: non-sharded output only, no reshards.
/// Note: tt-metal pad supports HS RM output when input is HS RM, but the
/// reshard path is blocked by the tile→RM constraint in validateReshard
/// (block_sharded TILE → HS RM requires a to_layout step first).
/// https://github.com/tenstorrent/tt-metal/issues/40898
struct PadRuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// ToLayoutOp: reject HEIGHT_SHARDED inputs to prevent the HS cascade from
/// propagating through layout conversion into reshape/conv2d chains that
/// require TILED tensors. Without this filter the optimizer threads HS-RM
/// through to_layout → HS-TILE output → reshape fails on HS-TILE → inserts a
/// wasteful DRAM→HS→DRAM round-trip.
struct ToLayoutRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  OutputHints getOutputHints(Operation *op,
                             const std::vector<OpConfig> &legalConfigs) const override;
};

/// UpsampleOp (bilinear): proposes HEIGHT_SHARDED output with the exact core
/// count that compute_bilinear_autoshard_memory_config will use at runtime:
/// num_cores = min(max_cores, N*H*W_input).
/// For nearest-mode, falls back to default (NULL hint + sharded fallbacks).
struct UpsampleRuleBook : OpRuleBook {
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_DATAMOVEMENTRULES_H
