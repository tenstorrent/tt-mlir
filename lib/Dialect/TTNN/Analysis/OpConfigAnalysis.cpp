// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"

#include <cassert>

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
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Picking op config for op {}",
                 opConfigs.first->getName());
    for (auto config : opConfigs.second) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "  Candidate config {}",
                   config.outputLayout);
    }
    TTNNLayoutAttr chosenLayout = opConfigs.second[0].outputLayout;

    RankedTensorType outputTensor =
        mlir::cast<RankedTensorType>(opConfigs.first->getResultTypes()[0]);
    TTNNLayoutAttr existingLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputTensor.getEncoding());
    assert(existingLayout && "Expected existing layout on op result");

    // If existing layout is tiled, keep it, if existing layout is RM, try to
    // pick a RM layout.
    for (auto config : opConfigs.second) {
      if (config.outputLayout.isTiled() == existingLayout.isTiled()) {
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "  Picking {} layout {}",
                     existingLayout.isTiled() ? "tiled" : "RM",
                     config.outputLayout);
        chosenLayout = config.outputLayout;
        break;
      }
    }

    analysisResult[opConfigs.first] = chosenLayout;
  }
}
} // namespace mlir::tt::ttnn
