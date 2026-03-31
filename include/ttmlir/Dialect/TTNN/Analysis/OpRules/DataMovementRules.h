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
// Reshard exploration: disabled for all data movement ops.
//===----------------------------------------------------------------------===//

/// ConcatOp: sharded inputs/output re-enabled after tt-metal hang fix.
/// https://github.com/tenstorrent/tt-metal/pull/39882
struct ConcatRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter() const override;
  bool shouldExploreReshards() const override;
};

/// SliceStaticOp, SliceDynamicOp: reject all sharded inputs, non-sharded
/// output, no reshards.
struct SliceRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter() const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// ReshapeOp, PermuteOp: reject width-sharded inputs, non-sharded output,
/// no reshards.
struct ReshapeRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter() const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// PadOp: non-sharded output only, no reshards.
struct PadRuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_DATAMOVEMENTRULES_H
