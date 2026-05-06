// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/SDPAUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tt::ttnn::utils {

std::optional<float> extractScalarConstant(Value v) {
  v = ttmlir::utils::lookThrough<TypecastOp>(v);

  if (auto fullOp = v.getDefiningOp<FullOp>()) {
    if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
      return attr.getValue().convertToFloat();
    }
  }

  if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
    auto callee = loadCached.getCallee();
    auto moduleOp = loadCached->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return std::nullopt;
    }

    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
    if (!funcOp) {
      return std::nullopt;
    }

    std::optional<float> result;
    funcOp.walk([&](FullOp fullOp) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        result = attr.getValue().convertToFloat();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  return std::nullopt;
}

std::pair<Value, std::optional<float>>
extractMultiplyWithScalarConstant(Value v) {
  if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
    if (auto s = extractScalarConstant(mulOp.getRhs())) {
      return {mulOp.getLhs(), s};
    }
    if (auto s = extractScalarConstant(mulOp.getLhs())) {
      return {mulOp.getRhs(), s};
    }
  }
  return {v, std::nullopt};
}

} // namespace mlir::tt::ttnn::utils
