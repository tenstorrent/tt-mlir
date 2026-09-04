// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPTIMIZERRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPTIMIZERRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Training optimizer step ops: AdamW.
//
// AdamW is a zero-result in-place op registered in
// optimizer_utils::isSinkOp.
//
// The ttml device operation validates every operand
// (adamw_device_operation.cpp:20-90) and TT_FATALs unless each is
//   - BufferType::DRAM
//   - TensorMemoryLayout::INTERLEAVED
//   - Layout::TILE
//===----------------------------------------------------------------------===//

/// AdamW: every operand must be tiled and DRAM-interleaved (see above).
struct AdamWRuleBook : OpRuleBook {
  LayoutFilterFn getInputLayoutFilter(unsigned operandIdx) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_OPTIMIZERRULES_H
