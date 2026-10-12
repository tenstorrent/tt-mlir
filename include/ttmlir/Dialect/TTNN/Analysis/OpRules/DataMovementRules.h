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
  using OpRuleBook::getInputLayoutFilter;
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  LayoutFilterFn getInputLayoutFilter(Operation *op,
                                      unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  bool generatesTiledInputSiblings(Operation *op,
                                   unsigned operandIdx) const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
  std::vector<OpConfig> inputDerivedOutputHints(
      Operation *op,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
};

/// True for a static slice that only narrows the last (channel) dimension of
/// a rank >= 2 tensor with unit steps: every shard keeps its rows and loses
/// columns, so the op can run shard-local on a height-sharded input.
bool isLastDimOnlySlice(Operation *op);

/// True if a `ttnn.reshape` is realized by tt-metal as a zero-copy view of
/// its input (same buffer, same address). Mirrors the `this_is_view`
/// condition in tt-metal's reshape.cpp.
bool canReshapeBeView(Operation *op);

/// ReshapeOp, PermuteOp: reject width-sharded inputs, no reshards.
/// View-eligible reshapes use NULL hint only (inherit input memory config).
/// Non-view reshapes use non-sharded output hints.
struct ReshapeRuleBook : OpRuleBook {
  using OpRuleBook::generatesRowMajorInputSiblings;
  using OpRuleBook::getInputLayoutFilter;
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  LayoutFilterFn getInputLayoutFilter(Operation *op,
                                      unsigned operandIdx) const override;
  bool generatesRowMajorInputSiblings(Operation *op,
                                      unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

/// True for a `ttnn.reshape` that copies in TILE layout but is a free view in
/// ROW_MAJOR (last dim unchanged, penultimate dims not tile-aligned) and whose
/// every user is a permute: untilizing first turns copy + tiled permute into
/// view + row-major permute.
bool reshapePrefersRowMajorInput(Operation *op);

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

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_DATAMOVEMENTRULES_H
