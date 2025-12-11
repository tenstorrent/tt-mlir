// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNEXTRACTISOLATEDTOTESTMODULE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttnn

namespace {

struct TTNNExtractIsolatedToTestModulePass
    : public ::mlir::tt::ttnn::impl::TTNNExtractIsolatedToTestModuleBase<
          TTNNExtractIsolatedToTestModulePass> {
  using ::mlir::tt::ttnn::impl::TTNNExtractIsolatedToTestModuleBase<
      TTNNExtractIsolatedToTestModulePass>::TTNNExtractIsolatedToTestModuleBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect all isolated functions
    SmallVector<func::FuncOp> isolatedFuncs;
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr("isolated")) {
        isolatedFuncs.push_back(funcOp);
      }
    });

    // If no isolated functions found, nothing to do
    if (isolatedFuncs.empty()) {
      return;
    }

    // Create a test module at the end of the main module
    builder.setInsertionPointToEnd(module.getBody());
    auto testModule = builder.create<ModuleOp>(
        module.getLoc(), StringAttr::get(builder.getContext(), "test_module"));

    // Move all isolated functions to the test module
    Block *testModuleBody = testModule.getBody();
    for (func::FuncOp funcOp : isolatedFuncs) {
      // Clone the function into the test module
      builder.setInsertionPointToEnd(testModuleBody);
      builder.clone(*funcOp.getOperation());

      // Erase from original module
      funcOp.erase();
    }
  }
};

} // namespace
