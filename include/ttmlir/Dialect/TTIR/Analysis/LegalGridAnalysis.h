// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_LEGALGRIDANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_LEGALGRIDANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Analysis/TTIRAnalysis.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::tt::ttir {

struct LegalGridAnalysisInput {
  ChipDescAttr chipDesc;
  GridAttr maxGrid;
  RankedTensorType tensorType;
  llvm::StringMap<SmallVector<int64_t, 2>> *gridSizeOverrides;

  LegalGridAnalysisInput()
      : chipDesc(nullptr), maxGrid(nullptr), tensorType(nullptr),
        gridSizeOverrides(nullptr) {}

  LegalGridAnalysisInput(
      ChipDescAttr chipDesc, GridAttr maxGrid, RankedTensorType tensorType,
      llvm::StringMap<SmallVector<int64_t, 2>> *gridSizeOverrides)
      : chipDesc(chipDesc), maxGrid(maxGrid), tensorType(tensorType),
        gridSizeOverrides(gridSizeOverrides) {}

  bool operator==(const LegalGridAnalysisInput &rhs) const {
    return chipDesc == rhs.chipDesc && maxGrid == rhs.maxGrid &&
           tensorType == rhs.tensorType &&
           gridSizeOverrides == rhs.gridSizeOverrides;
  }

  bool operator!=(const LegalGridAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

class LegalGridAnalysis
    : public TTIRAnalysis<LegalGridAnalysisInput, std::vector<GridAttr>> {
private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  LegalGridAnalysis(Operation *op) : TTIRAnalysis(op) {}
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_LEGALGRIDANALYSIS_H
