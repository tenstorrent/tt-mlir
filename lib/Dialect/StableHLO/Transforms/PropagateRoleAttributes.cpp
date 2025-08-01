// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_PROPAGATEROLEATTRIBUTESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class PropagateRoleAttributesPass
    : public impl::PropagateRoleAttributesPassBase<
          PropagateRoleAttributesPass> {
public:
  using impl::PropagateRoleAttributesPassBase<
      PropagateRoleAttributesPass>::PropagateRoleAttributesPassBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Helper function to recursively propagate tt.input_role attribute upward
    // through call chain
    std::function<void(Value, StringAttr)> propagateRoleAttribute =
        [&](Value value, StringAttr roleAttr) -> void {
      if (auto *definingOp = value.getDefiningOp()) {
        // Set the attribute on the defining operation
        definingOp->setAttr("tt.input_role", roleAttr);

        // If this is a call operation, propagate to its arguments
        if (auto callOp = mlir::dyn_cast<func::CallOp>(definingOp)) {
          for (Value operand : callOp.getOperands()) {
            propagateRoleAttribute(operand, roleAttr);
          }
        }
      } else if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
        // If it's a block argument, set the attribute on the parent function
        auto *parentOp = blockArg.getOwner()->getParentOp();
        auto argIndex = blockArg.getArgNumber();

        if (auto parentFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)) {
          parentFuncOp.setArgAttr(argIndex, "tt.input_role", roleAttr);

          // Find all call sites of this function and propagate upward
          auto funcName = parentFuncOp.getSymName();
          module.walk([&](func::CallOp walkCallOp) {
            if (walkCallOp.getCallee() == funcName &&
                argIndex < walkCallOp.getNumOperands()) {
              Value callerArg = walkCallOp.getOperand(argIndex);
              propagateRoleAttribute(callerArg, roleAttr);
            }
          });
        }
      }
    };

    // Walk through all func.call operations and propagate tt.input_role
    // attributes
    module.walk([&](func::CallOp callOp) {
      auto roleAttr = callOp->getAttrOfType<StringAttr>("tt.input_role");
      if (roleAttr) {
        // Propagate tt.input_role attribute from call site to each of its call
        // arguments
        for (Value operand : callOp.getOperands()) {
          propagateRoleAttribute(operand, roleAttr);
        }
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
