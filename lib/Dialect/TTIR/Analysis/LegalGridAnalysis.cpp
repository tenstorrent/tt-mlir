// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/LegalGridAnalysis.h"

namespace mlir::tt::ttir {

bool LegalGridAnalysis::applyOverrides() {
  // Lookup grid size overrides based on location information for current
  // operation.
  //
  if (analysisInput.gridSizeOverrides && isa<NameLoc>(op->getLoc())) {
    StringRef loc_str_op_name = mlir::cast<NameLoc>(op->getLoc()).getName();
    auto gridOverride = analysisInput.gridSizeOverrides->find(loc_str_op_name);
    if (gridOverride != analysisInput.gridSizeOverrides->end()) {
      analysisResult.push_back(GridAttr::get(
          op->getContext(), ArrayRef<int64_t>(gridOverride->second)));
      analysisResult.push_back(
          GridAttr::get(op->getContext(),
                        {gridOverride->second[0], gridOverride->second[1]}));
      return true;
    }
  }

  return false;
}

void LegalGridAnalysis::analysisImplementation() {
  // Placeholder, needs to be implemented. Go through all the grid sizes and
  // check if they are legal based on tensor type and device/chip attributes.
  // For now result of analysis is maximum supported grid size.
  //
  analysisResult.push_back(
      GridAttr::get(op->getContext(), analysisInput.maxGrid.getShape()));
}
} // namespace mlir::tt::ttir
