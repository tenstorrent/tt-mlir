// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRHOISTCPUOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Constant as fill pass
//===----------------------------------------------------------------------===//

class TTIRHoistCPUOpsRewriter : public OpRewritePattern<MaximumOp> {
public:
  using OpRewritePattern<MaximumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaximumOp op,
                                PatternRewriter &rewriter) const final {

    // Retrieve the parent module of the current operation
    auto parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Skip transformation if the op is already in the CPU module
    if (parentModule) {
      if (parentModule->hasAttr("ttir.cpu_module")) {
        return failure();
      }
    }
    auto loc = op.getLoc();

    // Check if a "cpu_module" already exists
    mlir::ModuleOp cpuModule;
    for (auto &op : parentModule.getBody()->getOperations()) {
      if (auto module = llvm::dyn_cast<mlir::ModuleOp>(&op)) {
        if (module->hasAttr("ttir.cpu_module")) {
          cpuModule = module;
          break;
        }
      }
    }

    // If no CPU module exists, create one
    if (!cpuModule) {
      rewriter.setInsertionPointToEnd(parentModule.getBody());
      cpuModule = rewriter.create<mlir::ModuleOp>(loc);
      cpuModule->setAttr("ttir.cpu_module", rewriter.getUnitAttr());
    }

    auto resultTy = op.getResultTypes();
    auto hoistFuncTy = rewriter.getFunctionType({op.getOperand(0).getType(),
                                                 op.getOperand(1).getType(),
                                                 op.getOperand(2).getType()},
                                                {resultTy});

    // define hoisted func, w placeholder attr for CPU execution
    rewriter.setInsertionPointToEnd(cpuModule.getBody());
    auto hoistFunc =
        rewriter.create<func::FuncOp>(loc, "cpu_maximum_func", hoistFuncTy);
    hoistFunc->setAttr("target", rewriter.getStringAttr("CPU"));

    // Create the function entry block, defining the operands as function
    // arguments
    auto entryBlock = hoistFunc.addEntryBlock();
    auto operand0 = entryBlock->addArgument(op.getOperand(0).getType(), loc);
    auto operand1 = entryBlock->addArgument(op.getOperand(1).getType(), loc);
    auto operand2 = entryBlock->addArgument(op.getOperand(2).getType(), loc);

    rewriter.setInsertionPointToEnd(entryBlock); // Set to the entry block
    auto cpuMaxOp = rewriter.create<MaximumOp>(
        loc, TypeRange{resultTy},
        ArrayRef<Value>{operand0, operand1, operand2});
    cpuMaxOp->setAttrs(op->getAttrs());
    rewriter.create<func::ReturnOp>(loc, cpuMaxOp.getResults());

    rewriter.setInsertionPoint(op);
    auto funcAttr =
        FlatSymbolRefAttr::get(rewriter.getContext(), hoistFunc.getName());
    auto callOp = rewriter.create<func::CallOp>(
        loc, funcAttr, TypeRange{resultTy}, op.getOperands().drop_back(1));
    rewriter.replaceOp(op, callOp.getResults());

    return success();
  }
};

class TTIRHoistCPUOps : public impl::TTIRHoistCPUOpsBase<TTIRHoistCPUOps> {
public:
  using impl::TTIRHoistCPUOpsBase<TTIRHoistCPUOps>::TTIRHoistCPUOpsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRHoistCPUOpsRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir
