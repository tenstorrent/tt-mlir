// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_GRIDSAMPLERULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_GRIDSAMPLERULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// GridSample rules:
//
// The kernel reads row-major operands only, interleaved or L1 height-sharded,
// and places its output itself (height-sharded L1). Every tiled candidate of
// the image and of the grid is cloned into its row-major siblings: the
// interleaved one and, for an L1 height-sharded producer, the one that
// untilizes on the producer's own shards, so the image stays in L1 and the
// sampler reads it there. The tiled originals are dropped by the input
// filter; output hints are left to the backend.
//===----------------------------------------------------------------------===//

struct GridSampleRuleBook : OpRuleBook {
  using OpRuleBook::generatesRowMajorInputSiblings;
  using OpRuleBook::getInputLayoutFilter;
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
  bool generatesRowMajorInputSiblings(unsigned operandIdx) const override;
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_GRIDSAMPLERULES_H
