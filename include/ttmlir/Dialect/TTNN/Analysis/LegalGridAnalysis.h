// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALGRIDANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALGRIDANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/OverrideParams.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::tt::ttnn {

struct LegalGridAnalysisInput {
  ChipDescAttr chipDesc;
  GridAttr maxGrid;
  RankedTensorType tensorType;
  int64_t maxShardedGrids;
  llvm::StringMap<LayoutOverrideParams> *outputLayoutOverrides;

  LegalGridAnalysisInput()
      : chipDesc(nullptr), maxGrid(nullptr), tensorType(nullptr),
        outputLayoutOverrides(nullptr) {}

  LegalGridAnalysisInput(
      ChipDescAttr chipDesc, GridAttr maxGrid, RankedTensorType tensorType,
      int64_t maxShardedGrids,
      llvm::StringMap<LayoutOverrideParams> *outputLayoutOverrides)
      : chipDesc(chipDesc), maxGrid(maxGrid), tensorType(tensorType),
        maxShardedGrids(maxShardedGrids),
        outputLayoutOverrides(outputLayoutOverrides) {}

  bool operator==(const LegalGridAnalysisInput &rhs) const {
    return chipDesc == rhs.chipDesc && maxGrid == rhs.maxGrid &&
           tensorType == rhs.tensorType &&
           outputLayoutOverrides == rhs.outputLayoutOverrides;
  }

  bool operator!=(const LegalGridAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

class LegalGridAnalysis
    : public TTNNAnalysis<LegalGridAnalysisInput, std::vector<tt::LayoutAttr>> {
private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  LegalGridAnalysis(Operation *op) : TTNNAnalysis(op) {}
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALGRIDANALYSIS_H
