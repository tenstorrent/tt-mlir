// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/SDPAUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/CallInterfaces.h"

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

  // Trace-mode hoisting case: the scalar constant has been pulled out of this
  // function and replaced by a block argument tagged with
  // `ttcore.argument_type = #ttcore.argument_type<constant>`. The annotation
  // is a contract that all callers pass the same compile-time-known value, so
  // we can recover it by walking to any caller and recursing on the
  // corresponding operand.
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(v)) {
    auto funcOp =
        mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!funcOp) {
      return std::nullopt;
    }

    unsigned argIdx = blockArg.getArgNumber();
    auto argTypeAttr = funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
        argIdx, ttcore::ArgumentTypeAttr::name);
    if (!argTypeAttr ||
        argTypeAttr.getValue() != ttcore::ArgumentType::Constant) {
      return std::nullopt;
    }

    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return std::nullopt;
    }

    Value callerArg;
    bool found = false;
    moduleOp.walk([&](mlir::CallOpInterface callOp) -> WalkResult {
      auto callee = callOp.getCallableForCallee();
      auto sym = mlir::dyn_cast<SymbolRefAttr>(callee);
      if (!sym || sym.getRootReference() != funcOp.getNameAttr()) {
        return WalkResult::advance();
      }
      auto operands = callOp.getArgOperands();
      if (argIdx >= operands.size()) {
        return WalkResult::advance();
      }
      callerArg = operands[argIdx];
      found = true;
      return WalkResult::interrupt();
    });
    if (!found) {
      return std::nullopt;
    }

    return extractScalarConstant(callerArg);
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
