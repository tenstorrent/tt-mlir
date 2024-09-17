// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mlir/IR/BuiltinAttributes.h>

#include "ttmlir/Dialect/TTIR/Analysis/GenericAnalysis.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir {

bool GenericAnalysis::applyOverrides() { return false; }

static SmallVector<int64_t> chooseGridShape(DeviceAttr device,
                                            RankedTensorType tensorType) {
  ArrayRef<int64_t> deviceGridShape = device.getWorkerGrid().getShape();
  auto layout = mlir::cast<tt::LayoutAttr>(tensorType.getEncoding());
  SmallVector<int64_t> tiledShape = layout.getTiledShape(tensorType.getShape());
  assert(deviceGridShape.size() == tiledShape.size());

  SmallVector<int64_t> maxGridShape;
  maxGridShape.reserve(deviceGridShape.size());
  for (size_t i = 0; i < deviceGridShape.size(); ++i) {
    maxGridShape.push_back(std::min(deviceGridShape[i], tiledShape[i]));
  }

  return maxGridShape;
}

void GenericAnalysis::analysisImplementation() {
  op->walk([&](GenericOp op) {
    auto result = op.getResult(0);
    analysisResult.gridShapes[op] = chooseGridShape(
        analysisInput.device, cast<RankedTensorType>(result.getType()));
  });
}
} // namespace mlir::tt::ttir
