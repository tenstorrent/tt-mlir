// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_GRIDANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_GRIDANALYSIS_H

#include "ttmlir/Dialect/TTIR/Analysis/TTIRAnalysis.h"

namespace mlir::tt::ttir {

struct GridAnalysisResult {
  int target_rows = 1;
  int target_columns = 1;
};

struct GridAnalysisInput {
  int max_supported_rows;
  int max_supported_columns;

  GridAnalysisInput() : max_supported_rows(1), max_supported_columns(1) {}

  GridAnalysisInput(int max_supported_rows, int max_supported_columns)
      : max_supported_rows(max_supported_rows),
        max_supported_columns(max_supported_columns) {}

  bool operator==(const GridAnalysisInput &rhs) const {
    return max_supported_rows == rhs.max_supported_rows &&
           max_supported_columns == rhs.max_supported_columns;
  }

  bool operator!=(const GridAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// Determine target grid size for each op.
//
class GridAnalysis
    : public TTIRAnalysis<GridAnalysisInput, GridAnalysisResult> {

private:
  void analysisImplementation() override;
  void handleOverride(llvm::json::Object *override) override;

public:
  GridAnalysis(Operation *op) : TTIRAnalysis(op) {}
};
} // namespace mlir::tt::ttir

#endif
