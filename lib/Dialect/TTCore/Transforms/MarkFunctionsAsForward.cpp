// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttcore {
#define GEN_PASS_DEF_TTCOREMARKFUNCTIONSASFORWARDPASS
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

class TTCoreMarkFunctionsAsForwardPass
    : public impl::TTCoreMarkFunctionsAsForwardPassBase<
          TTCoreMarkFunctionsAsForwardPass> {
public:
  using impl::TTCoreMarkFunctionsAsForwardPassBase<
      TTCoreMarkFunctionsAsForwardPass>::TTCoreMarkFunctionsAsForwardPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk through all functions in the module and mark public ones as forward
    // functions. This pass runs early in the pipeline before any const-eval or
    // CPU hoisting transforms, so all public functions at this point are
    // forward functions.
    module.walk([&](func::FuncOp funcOp) {
      // Skip if already tagged.
      if (ttmlir::utils::getFunctionType(funcOp).has_value()) {
        return;
      }

      // Skip private functions.
      if (funcOp.isPrivate()) {
        return;
      }

      // Tag this function as a forward device function.
      ttmlir::utils::setFunctionType(
          funcOp, ttmlir::utils::FunctionType::ForwardDevice);
    });
  }
};

} // namespace mlir::tt::ttcore
