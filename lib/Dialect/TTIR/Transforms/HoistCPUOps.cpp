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
    auto resultTy = op.getResultTypes();
    auto loc = op.getLoc();
    auto hoistFuncTy = rewriter.getFunctionType({resultTy}, {resultTy});

    // define hoisted func, w placeholder attr for CPU execution
    auto hoistFunc =
        rewriter.create<func::FuncOp>(loc, "cpu_maximum_func", hoistFuncTy);
    hoistFunc->setAttr("target", rewriter.getStringAttr("CPU"));

    rewriter.setInsertionPointToEnd(hoistFunc.addEntryBlock());
    // auto cpuMaxOp = rewriter.create<MaximumOp>(loc, resultTy,
    // op.getOperand(0),
    //                                            op.getOperand(1));
    rewriter.create<func::ReturnOp>(loc, op.getResults());

    SymbolRefAttr funcAttr =
        SymbolRefAttr::get(rewriter.getContext(), hoistFunc.getName());
    auto callOp = rewriter.create<func::CallOp>(loc, funcAttr, op.getOperand(0),
                                                op.getOperand(1));
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
