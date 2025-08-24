// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"

namespace mlir::tt::ttnn {

bool OpConfigAnalysis::applyOverrides() {

  // Placeholder, no overrides for now.
  //
  return false;
}

void OpConfigAnalysis::analysisImplementation() {

  // Future entrypoint for picking optimal op config.
  // Placeholder: pick the first legal config.
  //
  for (auto opConfigs : analysisInput.legalConfigs) {
    analysisResult[opConfigs.first] = opConfigs.second[0];
  }
}
} // namespace mlir::tt::ttnn
