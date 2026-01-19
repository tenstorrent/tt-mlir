// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNMATERIALIZED2M
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNMaterializeD2M
    : public impl::TTNNMaterializeD2MBase<TTNNMaterializeD2M> {
public:
  using impl::TTNNMaterializeD2MBase<
      TTNNMaterializeD2M>::TTNNMaterializeD2MBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SmallVector<ttnn::DispatchD2MOp> toCompile;
    moduleOp.walk([&](ttnn::DispatchD2MOp dispatchD2MOp) {
      toCompile.push_back(dispatchD2MOp);
    });

    if (toCompile.empty()) {
      return;
    }

    for (ttnn::DispatchD2MOp dispatchOp : toCompile) {
      if (failed(compileViaD2M(dispatchOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult compileViaD2M(ttnn::DispatchD2MOp dispatchOp) {
    // Get the entry function from the body region
    func::FuncOp mainFunc = dispatchOp.lookupD2MMainFunc();
    if (!mainFunc) {
      return dispatchOp.emitOpError("could not find D2M function '")
             << dispatchOp.getD2mFunc() << "' in body region";
    }

    // Create a temporary module and clone function body into it.
    OpBuilder builder(dispatchOp.getContext());
    auto tempModule = ModuleOp::create(dispatchOp.getLoc());
    builder.setInsertionPointToEnd(tempModule.getBody());
    IRMapping mapping;
    builder.clone(*mainFunc.getOperation(), mapping);

    llvm::errs() << "\n=== D2M Input Module ===\n";
    tempModule.print(llvm::errs());
    llvm::errs() << "\n=== End D2M Input Module ===\n\n";

    auto D2MPm = PassManager::on<ModuleOp>(tempModule.getContext());
    D2MPm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
        /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
        /*printModuleScope=*/false, // true requires single-threading
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false,
        /*out=*/llvm::errs());

    ttmetal::TTIRToTTMetalPipelineOptions d2mOptions;
    d2mOptions.ttnnMode = true;

    ttmetal::createTTIRToTTMetalFrontendPipeline(D2MPm, d2mOptions);
    ttmetal::createTTIRToTTMetalMiddleendPipeline(D2MPm, d2mOptions);
    ttmetal::createTTIRToTTMetalBackendPipeline(D2MPm, d2mOptions);

    if (failed(D2MPm.run(tempModule))) {
      tempModule.erase();
      return dispatchOp.emitOpError("D2M pipeline failed");
    }

    llvm::errs() << "\n=== D2M Output Module ===\n";
    tempModule.print(llvm::errs());
    llvm::errs() << "\n=== End D2M Output Module ===\n\n";

    // Replace the body region content with the compiled result
    Region &body = dispatchOp.getBody();
    body.front().clear();
    builder.setInsertionPointToEnd(&body.front());

    IRMapping newMapping;
    for (Operation &op : tempModule.getBody()->getOperations()) {
      // if (op.hasTrait<OpTrait::IsTerminator>()) {
      //   continue;
      // }
      builder.clone(op, newMapping);
    }

    tempModule.erase();
    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
