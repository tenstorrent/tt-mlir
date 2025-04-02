// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALLAYOUTANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALLAYOUTANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "llvm/ADT/StringMap.h"

namespace mlir::tt::ttnn {

struct LegalLayoutAnalysisInput {
  ChipDescAttr chipDesc;
  GridAttr maxGrid;
  RankedTensorType tensorType;
  int64_t maxShardedConfigs;
  llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides;
  llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides;
  bool rowMajorEnabled;

  LegalLayoutAnalysisInput()
      : chipDesc(nullptr), maxGrid(nullptr), tensorType(nullptr),
        outputLayoutOverrides(nullptr), conv2dConfigOverrides(nullptr) {}

  LegalLayoutAnalysisInput(
      ChipDescAttr chipDesc, GridAttr maxGrid, RankedTensorType tensorType,
      int64_t maxShardedConfigs,
      llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides,
      llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides,
      bool rowMajorEnabled)
      : chipDesc(chipDesc), maxGrid(maxGrid), tensorType(tensorType),
        maxShardedConfigs(maxShardedConfigs),
        outputLayoutOverrides(outputLayoutOverrides),
        conv2dConfigOverrides(conv2dConfigOverrides),
        rowMajorEnabled(rowMajorEnabled) {}

  bool operator==(const LegalLayoutAnalysisInput &rhs) const {
    return chipDesc == rhs.chipDesc && maxGrid == rhs.maxGrid &&
           tensorType == rhs.tensorType &&
           outputLayoutOverrides == rhs.outputLayoutOverrides &&
           conv2dConfigOverrides == rhs.conv2dConfigOverrides;
  }

  bool operator!=(const LegalLayoutAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

class LegalLayoutAnalysis
    : public TTNNAnalysis<LegalLayoutAnalysisInput, std::vector<OpConfig>> {
private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  LegalLayoutAnalysis(Operation *op) : TTNNAnalysis(op) {}
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALLAYOUTANALYSIS_H
