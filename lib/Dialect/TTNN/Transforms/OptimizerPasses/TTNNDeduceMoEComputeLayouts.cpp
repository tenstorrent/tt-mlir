// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDEDUCEMOECOMPUTELAYOUTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeduceMoEComputeLayouts
    : public impl::TTNNDeduceMoEComputeLayoutsBase<
          TTNNDeduceMoEComputeLayouts> {
public:
  using impl::TTNNDeduceMoEComputeLayoutsBase<
      TTNNDeduceMoEComputeLayouts>::TTNNDeduceMoEComputeLayoutsBase;

  // If a deduced result feeds directly into a func.return, the enclosing
  // function's result type must be refreshed to match. (Other consumers read
  // the type through use-def so they don't need an explicit fix-up.)
  void refreshEnclosingFuncReturnType(mlir::Value refinedResult) {
    for (mlir::OpOperand &use : refinedResult.getUses()) {
      auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(use.getOwner());
      if (!retOp) {
        continue;
      }
      auto funcOp = retOp->getParentOfType<mlir::func::FuncOp>();
      if (!funcOp) {
        continue;
      }
      llvm::SmallVector<mlir::Type> newResultTypes(
          funcOp.getFunctionType().getResults().begin(),
          funcOp.getFunctionType().getResults().end());
      newResultTypes[use.getOperandNumber()] = refinedResult.getType();
      funcOp.setFunctionType(mlir::FunctionType::get(
          funcOp.getContext(), funcOp.getFunctionType().getInputs(),
          newResultTypes));
    }
  }

  // The weight-prep ops are created with placeholder result types. Replace them
  // with the device-derived specs OpModel computes via graph-captured query of
  // the matching tt-metal invocation. Consumers read the deduced types via
  // use-def.
  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal("TTNNDeduceMoEComputeLayouts requires "
                                    "OpModel support to be enabled.");
#else
    op_model::ScopedSingletonDeviceGuard deviceGuard(getOperation());
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](ttnn::PrepareMoEComputeW0W1WeightsOp op) {
      op.getResult().setType(
          op_model::getPreparedMoEComputeW0W1WeightsOutputType(&op));
      refreshEnclosingFuncReturnType(op.getResult());
    });

    moduleOp.walk([&](ttnn::PrepareMoEComputeW2WeightsOp op) {
      op.getResult().setType(
          op_model::getPreparedMoEComputeW2WeightsOutputType(&op));
      refreshEnclosingFuncReturnType(op.getResult());
    });
#endif // TTMLIR_ENABLE_OPMODEL
  }
};

} // namespace mlir::tt::ttnn
