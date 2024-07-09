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
} // namespace mlir::tt::ttir
