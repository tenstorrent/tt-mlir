// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_TAGCONSTEVALOPS
#include "ttmlir/Transforms/Passes.h.inc"

namespace {
class TagConstEvalOps : public impl::TagConstEvalOpsBase<TagConstEvalOps> {
public:
  using impl::TagConstEvalOpsBase<TagConstEvalOps>::TagConstEvalOpsBase;

  void runOnOperation() final {
    mlir::ModuleOp module = this->getOperation();
    mlir::MLIRContext *context = &getContext();

    // Walk through all functions in the module
    module.walk([&](func::FuncOp funcOp) {
      // Check if this function has the const_eval attribute
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        tagOpsInFunction(funcOp, context);
      }
    });
  }

private:
  void tagOpsInFunction(func::FuncOp funcOp, mlir::MLIRContext *context) {
    OpBuilder builder(context);

    // Walk through all operations in the function
    funcOp.getBody().walk([&](Operation *op) {
      // Skip terminator operations (like func.return)
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return;
      }

      if (op->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>()) {
        return;
      }

      // Check if the operation already has the ttir.should_hoist attribute
      if (!op->hasAttr("ttir.should_hoist")) {
        // Add the ttir.should_hoist attribute
        op->setAttr("ttir.should_hoist", builder.getUnitAttr());
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::transforms
