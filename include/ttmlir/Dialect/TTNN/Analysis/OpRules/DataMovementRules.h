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

/// RepeatOp: pin the output sharding grid to the input's grid.
/// With a NULL/gridless output memory config, tt-metal's repeat re-derives the
/// output shard spec onto the full compute grid, so the grid it reports no
/// longer matches the physically-allocated (trimmed) grid and downstream
/// consumers read the wrong cores (tt-xla#5605). Force an explicit sharded
/// output on the input's memory layout and grid.
struct RepeatRuleBook : OpRuleBook {
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// SliceStaticOp, SliceDynamicOp: reject all sharded inputs, non-sharded
/// output, no reshards.
struct SliceRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// True if a `ttnn.reshape` is realized by tt-metal as a zero-copy view of
/// its input (same buffer, same address). Mirrors the `this_is_view`
/// condition in tt-metal's reshape.cpp.
bool canReshapeBeView(Operation *op);

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
struct PadRuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// MeshPartitionOp: DRAM-interleaved output only, no reshards.
/// Downstream consumers (paged_update_cache, paged_sdpa_decode, mesh_shard)
/// all require DRAM-interleaved input; allowing the optimizer to pick L1
/// for the producer leaves multi-consumer reshards in inconsistent states.
struct MeshPartitionRuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// AllGatherOp / ReduceScatterOp / AllReduceOp: keep CCL op inputs and outputs
/// interleaved (L1 or DRAM), never sharded.
///
/// The op-model constraint query validates a CCL op's sharded tensor spec
/// against the PER-DEVICE tensor shape (the shape carried in the TTNN IR). At
/// runtime, a mesh tensor that remains sharded on a non-cluster mesh axis
/// carries the GLOBAL logical shape, and metal builds the CCL output TensorSpec
/// from that global shape. So a sharded CCL input/output the query accepts
/// (per-device shard grid fits) can fail metal's shard-grid-fit at runtime
/// (global shard grid overflows the core rows/cols; TT_FATAL @
/// tensor_spec.cpp). The concrete crash observed is L1-sharded, but
/// DRAM-sharded is rejected for the same reason; only interleaved layouts are
/// safe. Until tt-mlir tracks per-value mesh distribution and can validate the
/// global shard grid (TODO(rpavlovicTT, #9070)), keep CCL layouts interleaved
/// so the optimizer never offers a layout the runtime can't build.
struct CCLRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_DATAMOVEMENTRULES_H
