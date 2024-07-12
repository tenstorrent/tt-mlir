// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GridAnalysis.h"

namespace mlir::tt::ttir {
void GridAnalysis::analysisImplementation() {
  // Placeholder. For now result of analysis is maximum supported grid size.
  //
  analysis_result.target_rows = analysis_input.max_supported_rows;
  analysis_result.target_columns = analysis_input.max_supported_columns;
}

void GridAnalysis::handleOverride(llvm::json::Object *override) {
  // Need to now collect the grid parameter from the override:
  // Grid Override Syntax: [rows, cols]
  llvm::json::Array newGrid = *(override->getArray("grid"));

  analysis_result.target_rows = newGrid[0].getAsInteger().value();
  analysis_result.target_columns = newGrid[1].getAsInteger().value();
}

} // namespace mlir::tt::ttir
