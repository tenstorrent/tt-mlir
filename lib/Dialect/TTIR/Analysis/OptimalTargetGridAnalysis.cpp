// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/OptimalTargetGridAnalysis.h"

namespace mlir::tt::ttir {

bool OptimalTargetGridAnalysis::applyOverrides() {

  // Placeholder, no overrides for now.
  //
  return false;
}

void OptimalTargetGridAnalysis::analysisImplementation() {

  // Implement GraphSolver like algorithm to eliminate illegal grid combinations
  // from graph globaly.
  // Entry point for graphsolving.
  //

  // Balancer/GridPicker implementation.
  // Future entrypoint for balancing and picking optimal grid.
  // Placeholder: pick the first legal grid.
  //
  for (auto opGrids : analysisInput.legalGrids) {
    analysisResult[opGrids.first] = opGrids.second[0];
  }
}
} // namespace mlir::tt::ttir
