// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPCONFIGANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPCONFIGANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include <vector>

namespace mlir::tt::ttnn {

struct LegalOpConfigAnalysisInput {
  // Legal configs found by LegalOpLayoutAnalysis.
  std::vector<OpConfig> legalConfigs;

  // Conv2d config overrides.
  llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides;

  LegalOpConfigAnalysisInput() : conv2dConfigOverrides(nullptr) {}

  LegalOpConfigAnalysisInput(
      std::vector<OpConfig> legalConfigs,
      llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides)
      : legalConfigs(legalConfigs),
        conv2dConfigOverrides(conv2dConfigOverrides) {}

  bool operator==(const LegalOpConfigAnalysisInput &rhs) const {
    return legalConfigs == rhs.legalConfigs &&
           conv2dConfigOverrides == rhs.conv2dConfigOverrides;
  }

  bool operator!=(const LegalOpConfigAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// This analysis takes legal configs found by LegalOpLayoutAnalysis and applies
// op config overrides (such as conv2d config overrides). Also, it searches
// through all legal op specific attributes and applies cartesian product of
// them with legal output layouts.
class LegalOpConfigAnalysis
    : public TTNNAnalysis<LegalOpConfigAnalysisInput, std::vector<OpConfig>> {
public:
  using TTNNAnalysis<LegalOpConfigAnalysisInput,
                     std::vector<OpConfig>>::TTNNAnalysis;

  LegalOpConfigAnalysis(Operation *op) : TTNNAnalysis(op) {}

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

  void fillOpSpecificAttrs();

  // Search space for conv2d config. Shared across all conv2d ops.
  Conv2dConfigSearchSpace searchSpace = Conv2dConfigSearchSpaceFactory::get();
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPCONFIGANALYSIS_H
