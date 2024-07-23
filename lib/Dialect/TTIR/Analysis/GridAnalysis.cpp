// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GridAnalysis.h"

namespace mlir::tt::ttir {

bool GridAnalysis::applyOverrides() {
  // Lookup grid size overrides based on location information for current
  // operation.
  //
  if (analysis_input.grid_size_overrides && op->getLoc().isa<NameLoc>()) {
    StringRef loc_str_op_name = op->getLoc().cast<NameLoc>().getName();
    auto grid_override =
        analysis_input.grid_size_overrides->find(loc_str_op_name);
    if (grid_override != analysis_input.grid_size_overrides->end()) {
      analysis_result.target_rows = grid_override->second[0];
      analysis_result.target_columns = grid_override->second[1];
      return true;
    }
  }

  return false;
}

void GridAnalysis::analysisImplementation() {
  // Placeholder. For now result of analysis is maximum supported grid size.
  //
  analysis_result.target_rows = analysis_input.max_supported_rows;
  analysis_result.target_columns = analysis_input.max_supported_columns;
}
} // namespace mlir::tt::ttir
