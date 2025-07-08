// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_WRAPUNDERMANUALCOMPUTATIONPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Wrap all operations within a module under a manual computation op.
static mlir::LogicalResult wrapFunctionBodyInManualComputationOp(
    MLIRContext *context, mlir::OpBuilder &builder,
    mlir::sdy::MeshOp &globalMeshOp, func::FuncOp &funcOp) {
  // Get in_shardings of current function.
  mlir::sdy::TensorShardingPerValueAttr inShardings =
      mlir::sdy::TensorShardingPerValueAttr::get(
          context,
          shardy_utils::getInShardingAttrs(context, funcOp, globalMeshOp));

  // Get out_shardings of current function.
  mlir::sdy::TensorShardingPerValueAttr outShardings =
      mlir::sdy::TensorShardingPerValueAttr::get(
          context,
          shardy_utils::getOutShardingAttrs(context, funcOp, globalMeshOp));

  // Create sdy.manual_computation op
  mlir::FunctionType funcType = funcOp.getFunctionType();
  mlir::Block &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);
  mlir::sdy::ManualComputationOp manualComputationOp =
      builder.create<mlir::sdy::ManualComputationOp>(
          builder.getUnknownLoc(), funcType.getResults(), funcOp.getArguments(),
          inShardings, outShardings, llvm::SmallVector<mlir::StringAttr>());

  // Determine the argumentTypes and argumentLocations that need to get
  // added to the new region in manualComputationOp.
  llvm::SmallVector<mlir::Type> argumentTypes;
  llvm::SmallVector<mlir::Location> argumentLocations;

  for (auto arg : funcOp.getArguments()) {
    // All arguments must be annotated with sdy.sharding attribute at this
    // point.
    mlir::RankedTensorType oldType =
        mlir::cast<mlir::RankedTensorType>(arg.getType());
    argumentTypes.push_back(oldType);
    argumentLocations.push_back(arg.getLoc());
  }

  // Create a new block and new region in manualComputationOp since it is
  // a region based op.
  mlir::Block &sourceBlock = funcOp.getBody().front();
  mlir::Region &targetRegion = manualComputationOp->getRegion(0);
  std::unique_ptr<mlir::Block> newBlock = std::make_unique<mlir::Block>();
  targetRegion.push_back(
      newBlock.release()); // Dynamically created block memory moves
                           // ownership into region.
  mlir::Block &targetBlock = targetRegion.front();
  targetBlock.addArguments(argumentTypes, argumentLocations);

  // Migrate all the ops currently in funcOp into the manualComputationOp
  // region. This will also copy func.ReturnOp.
  mlir::Block::iterator it = ++mlir::Block::iterator(manualComputationOp);
  targetBlock.getOperations().splice(
      targetBlock.end(), sourceBlock.getOperations(), it, sourceBlock.end());

  // Create a new func.ReturnOp in the original func.funcOp that takes the
  // manualComputationOp as it's operand.
  builder.setInsertionPointAfter(manualComputationOp);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                       manualComputationOp->getResults());

  // Update old arguments with new arguments inside of the
  // manualComputationBlock.
  for (uint32_t i = 0; i < sourceBlock.getArguments().size(); i++) {
    auto oldArg = sourceBlock.getArgument(i);
    auto newArg = targetBlock.getArgument(i);

    for (auto &targetOp : targetBlock.getOperations()) {
      for (auto &operand : targetOp.getOpOperands()) {
        if (operand.get() == oldArg) {
          operand.set(newArg);
        }
      }
    }
  }

  // Update all func.return ops in manualComputationOp with sdy.return op.
  // This is because the manualComputationOp requires a sdy.returnOp.
  for (Operation &op :
       llvm::make_early_inc_range(targetBlock.getOperations())) {
    if (!mlir::isa<mlir::func::ReturnOp>(op)) {
      continue;
    }

    mlir::func::ReturnOp returnOp = mlir::cast<mlir::func::ReturnOp>(op);
    builder.setInsertionPoint(returnOp);
    builder.create<mlir::sdy::ReturnOp>(builder.getUnknownLoc(),
                                        returnOp->getOperands());
    returnOp->erase();
  }

  return mlir::success();
}

class WrapUnderManualComputationPass
    : public impl::WrapUnderManualComputationPassBase<
          WrapUnderManualComputationPass> {
public:
  using impl::WrapUnderManualComputationPassBase<
      WrapUnderManualComputationPass>::WrapUnderManualComputationPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);
    mlir::PassManager pm(context);

    // Check if the graph is already solved by shardy. If so, we will remove
    // all sdy tensor shardings from the arguments and results.
    if (shardy_utils::isGraphSolved(rootModule)) {
      rootModule.walk([&](func::FuncOp funcOp) {
        shardy_utils::removeSdyTensorShardings(context, funcOp);
      });

      return;
    }

    // If the module has gspmd annotations, we skip this pass.
    bool gspmdAnnotationsExist = gspmd_utils::gspmdAnnotationsExist(rootModule);
    if (gspmdAnnotationsExist) {
      rootModule.emitWarning("Wrapping under manual computation pass does not "
                             "support GSPMD annotated module for now");
      return;
    }

    // Get the shardy mesh op in the root module.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(rootModule);

    if (parsedMeshOps.size() == 0) {
      rootModule.emitError(
          "Pass requires a shardy mesh op to be present in the root module");
      signalPassFailure();
      return;
    }

    if (parsedMeshOps.size() > 1) {
      rootModule.emitError("Pass currently only support a single shardy mesh "
                           "op in the module");
      signalPassFailure();
      return;
    }

    globalMeshOp = parsedMeshOps[0];

    // Sdy does not have support for a custom call for FullToShard or
    // ShardToFull like gspmd. Therefore, wrap all operations in each function
    // under a sdy.manual_computation op. This is required to enable the
    // conversion from sdy into ttir.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(wrapFunctionBodyInManualComputationOp(context, builder,
                                                       globalMeshOp, funcOp))) {
        rootModule.emitError(
            "Could not wrap function body inside a manual computation op");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
