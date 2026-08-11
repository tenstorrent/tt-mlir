// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

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

      // When the function is const-eval-hoisted, it is invoked via
      // ttcore.load_cached; its results map 1:1 to the function returns, so
      // refine the matching cached result too.
      auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
      if (!moduleOp) {
        continue;
      }
      moduleOp.walk([&](ttcore::LoadCachedOp loadCachedOp) {
        if (loadCachedOp.getCallee() == funcOp.getName()) {
          loadCachedOp.getResult(use.getOperandNumber())
              .setType(refinedResult.getType());
        }
      });
    }
  }

  // Reshard a moe_compute tilize input (expert indices or scores) onto the
  // device-derived tilize-drain core. The fused kernel allocates the indices/
  // scores circular buffers against an L1 buffer on that single core but never
  // checks placement, so we pin it here in IR. The reshard is ROW_MAJOR + L1 +
  // single-core HEIGHT_SHARDED: the kernel reads these untilized, and a sharded
  // TILE spec needs a 32x32-aligned shard the small last dim (select_experts_k)
  // can't satisfy.
  void reshardTilizeInputToDrainCore(ttnn::MoeComputeOp op, unsigned operandIdx,
                                     CoreRangeSetAttr drainCoreRangeSet) {
    mlir::IRRewriter rewriter(op.getContext());
    rewriter.setInsertionPoint(op);

    auto inputValue = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
        op->getOperand(operandIdx));
    mlir::RankedTensorType inputType = inputValue.getType();

    llvm::SmallVector<int64_t, 2> gridShape{1, 1};
    TTNNLayoutAttr drainLayout =
        TTNNLayoutAttr::Builder(inputType)
            .setLayout(Layout::RowMajor)
            .setBufferType(BufferType::L1)
            .setMemoryLayout(TensorMemoryLayoutAttr::get(
                op.getContext(), TensorMemoryLayout::HeightSharded))
            .setGridShape(gridShape)
            .setCoreRangeSet(drainCoreRangeSet)
            .build();
    mlir::RankedTensorType drainType =
        utils::RankedTensorTypeFactory::create(inputType, drainLayout);

    auto toMemCfg = rewriter.create<ttnn::ToMemoryConfigOp>(
        op.getLoc(), drainType, inputValue);

    rewriter.modifyOpInPlace(
        op, [&]() { op->setOperand(operandIdx, toMemCfg.getResult()); });
  }

  // The weight-prep ops are created with placeholder result types. Replace them
  // with the device-derived specs OpModel computes via graph-captured query of
  // the matching tt-metal invocation. Consumers read the deduced types via
  // use-def. Also reshards moe_compute's expert indices/scores inputs onto the
  // device-derived tilize drain core.
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

    moduleOp.walk([&](ttnn::MoeComputeOp op) {
      CoreRangeSetAttr drainCoreRangeSet =
          op_model::getMoeTilizeDrainCoreRangeSet(&op);
      // Operands 1 (expert indices) and 2 (expert scores) are read by the
      // kernel from CBs allocated on the drain core.
      reshardTilizeInputToDrainCore(op, /*operandIdx=*/1, drainCoreRangeSet);
      reshardTilizeInputToDrainCore(op, /*operandIdx=*/2, drainCoreRangeSet);
    });
#endif // TTMLIR_ENABLE_OPMODEL
  }
};

} // namespace mlir::tt::ttnn
