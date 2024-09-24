// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTIR/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include <llvm/Support/Casting.h>

namespace mlir::tt::ttir {

bool OpConfigAnalysis::applyOverrides() {

  // Placeholder, no overrides for now.
  //
  return false;
}

void OpConfigAnalysis::analysisImplementation() {

  // Future entrypoint for picking optimal op config.
  // Placeholder: pick the first legal grid.
  //
  for (auto opGrids : analysisInput.legalGrids) {
    Operation *op = opGrids.first;
    const std::vector<LayoutAttr> &legalLayouts = opGrids.second;
    if (not opGrids.second.empty()) {
      analysisResult[op] = legalLayouts[0];
    }
  }
}
} // namespace mlir::tt::ttir
