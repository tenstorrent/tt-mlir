// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Utils.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_PROPAGATECOMPOSITEOPERANDSHARDINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class PropagateCompositeOperandShardingPass
    : public impl::PropagateCompositeOperandShardingPassBase<
          PropagateCompositeOperandShardingPass> {
public:
  using impl::PropagateCompositeOperandShardingPassBase<
      PropagateCompositeOperandShardingPass>::
      PropagateCompositeOperandShardingPassBase;

  StringRef getArgument() const final {
    return "sdy-propagate-composite-operand-sharding";
  }

  StringRef getDescription() const final {
    return "Propagate sharding attributes from stablehlo.composite operands "
           "into their decomposition functions, then re-run aggressive "
           "propagation on those functions.";
  }

  void runOnOperation() final {
    llvm::errs() << "Running PropagateCompositeOperandShardingPass\n";

    // Walk all stablehlo.composite ops in the module.
    mlir::ModuleOp moduleOp = getOperation();
    mlir::SymbolTable symbolTable(moduleOp);

    moduleOp.walk([&](mlir::stablehlo::CompositeOp compositeOp) {
      // 1) this composite must have decomposition name (string)
      llvm::StringRef calleeName = compositeOp.getDecomposition();
      if (calleeName.empty()) {
        return mlir::WalkResult::advance();
      }

      // 2) lookup callee
      mlir::Operation *calleeOp = symbolTable.lookup(calleeName);
      if (!calleeOp) {
        compositeOp.emitError()
            << "failed to find callee function for composite: @" << calleeName;
        return mlir::WalkResult::interrupt();
      }
      mlir::func::FuncOp calleeFunc = llvm::cast<mlir::func::FuncOp>(calleeOp);

      // 3) only handle composites with exactly 1 operand for now
      if (compositeOp.getNumOperands() != 1) {
        compositeOp.emitError()
            << "expected composite with exactly 1 operand, got "
            << compositeOp.getNumOperands();
        return mlir::WalkResult::interrupt();
      }

      mlir::Value operand = compositeOp.getOperand(0);
      // mlir::MLIRContext *context = operand.getContext();

      // 결과로 만들 TensorShardingAttr
      mlir::sdy::TensorShardingAttr tensorSharding = nullptr;

      // 4) case A: operand is a block argument → take sdy.sharding from func
      // arg
      if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
        auto parentFunc = llvm::dyn_cast<mlir::func::FuncOp>(parentOp);
        if (!parentFunc) {
          compositeOp.emitError() << "composite operand is a block argument "
                                     "but parent is not func.func";
          return mlir::WalkResult::interrupt();
        }

        tensorSharding =
            parentFunc.getArgAttrOfType<mlir::sdy::TensorShardingAttr>(
                blockArg.getArgNumber(), "sdy.sharding");
        if (!tensorSharding) {
          compositeOp.emitError()
              << "block argument feeding composite has no 'sdy.sharding' attr";
          return mlir::WalkResult::interrupt();
        }
      } else {
        // 5) case B: operand is produced by an op → op result has ONLY
        // per_value
        mlir::Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          compositeOp.emitError() << "composite operand has no defining op and "
                                     "is not a block argument";
          return mlir::WalkResult::interrupt();
        }

        // 1) try the “new” name
        auto perValueAttr =
            defOp->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
                "sdy.sharding_per_value");

        // 2) your IR uses this:  {sdy.sharding = #sdy.sharding_per_value[...] }
        if (!perValueAttr) {
          perValueAttr =
              defOp->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
                  "sdy.sharding");
        }

        if (!perValueAttr) {
          compositeOp.emitError()
              << "expected defining op of composite operand to have per-value "
                 "sharding "
                 "('sdy.sharding_per_value' or 'sdy.sharding')";
          return mlir::WalkResult::interrupt();
        }

        // per-value: take first entry
        auto perValueShardings = perValueAttr.getShardings();
        if (perValueShardings.empty()) {
          compositeOp.emitError()
              << "'sdy.sharding_per_value' is present but has no entries";
          return mlir::WalkResult::interrupt();
        }

        llvm::errs() << "perValueShardings : " << perValueShardings.front()
                     << "\n";
        llvm::errs() << "perValueAttr : " << perValueAttr << "\n";

        tensorSharding = llvm::dyn_cast<mlir::sdy::TensorShardingAttr>(
            perValueShardings.front());

        //   auto firstEntry = llvm::dyn_cast<mlir::ArrayAttr>(
        //       perValueShardings.front());
        //   if (!firstEntry) {
        //     compositeOp.emitError()
        //         << "first entry of 'sdy.sharding_per_value' is not
        //         ArrayAttr";
        //     return mlir::WalkResult::interrupt();
        //   }

        //   // build DimensionShardingAttr list
        //   llvm::SmallVector<mlir::sdy::DimensionShardingAttr, 4>
        //   dimensionShardings; dimensionShardings.reserve(firstEntry.size());
        //   for (mlir::Attribute attr : firstEntry) {
        //     auto dimSharding =
        //     llvm::dyn_cast<mlir::sdy::DimensionShardingAttr>(attr); if
        //     (!dimSharding) {
        //       compositeOp.emitError()
        //           << "all elements in per-value sharding entry must be "
        //              "DimensionShardingAttr";
        //       return mlir::WalkResult::interrupt();
        //     }
        //     dimensionShardings.push_back(dimSharding);
        //   }

        //   auto shardings = perValueAttr.getShardings();
        //   if (shardings.empty()) {
        //     compositeOp.emitError()
        //         << "'sdy.sharding_per_value' is present but has no entries";
        //     return mlir::WalkResult::interrupt();
        //   }
        //     auto firstAsTensor =
        //     llvm::dyn_cast<mlir::sdy::TensorShardingAttr>(
        //       shardings.front());
        //   if (!firstAsTensor) {
        //     compositeOp.emitError()
        //         << "expected first entry of sdy.sharding_per_value to be
        //         TensorShardingAttr";
        //     return mlir::WalkResult::interrupt();
        //   }

        //   tensorSharding = firstAsTensor;
      }

      // 6) attach to callee arg0
      mlir::Block &entryBlock = calleeFunc.getBody().front();
      if (entryBlock.getNumArguments() < 1) {
        compositeOp.emitError()
            << "callee @" << calleeName << " has no arguments";
        return mlir::WalkResult::interrupt();
      }
      calleeFunc.setArgAttr(/*argIndex=*/0, "sdy.sharding", tensorSharding);
      calleeFunc.setResultAttr(0, "sdy.sharding", tensorSharding);
      // 7) run aggressive propagation only on this callee
      {
        mlir::OwningOpRef<mlir::ModuleOp> tmpModule =
            mlir::ModuleOp::create(calleeFunc.getLoc());

        // 1) copy over all sdy.mesh ops from the original module, because
        //    aggressive propagation expects the mesh to exist at module scope.
        for (mlir::Operation &op : moduleOp.getBody()->getOperations()) {
          if (auto meshOp = llvm::dyn_cast<mlir::sdy::MeshOp>(&op)) {
            mlir::Operation *cloned = meshOp.clone();
            tmpModule->push_back(cloned);
          }
        }

        mlir::func::FuncOp clonedFunc =
            llvm::cast<mlir::func::FuncOp>(calleeFunc.clone());
        tmpModule->push_back(clonedFunc);

        mlir::PassManager pm(tmpModule.get()->getName());
        mlir::sdy::PropagationOptions propagationOptions;
        mlir::sdy::PropagationStrategy propagationStrategy =
            mlir::sdy::PropagationStrategy::Aggressive;
        propagationOptions.conservativePropagation = true;
        pm.addPass(mlir::sdy::createAggressivePropagationPass(
            /*propagationOptions=*/propagationOptions,
            /*propagationStrategy=*/propagationStrategy));

        if (mlir::failed(pm.run(*tmpModule))) {
          compositeOp.emitError()
              << "failed to run sdy-aggressive-propagate on @" << calleeName;
          return mlir::WalkResult::interrupt();
        }

        // replace body
        auto &origBody = calleeFunc.getBody();
        while (!origBody.empty()) {
          origBody.front().erase();
        }
        origBody.takeBody(clonedFunc.getBody());
        calleeFunc.setFunctionType(clonedFunc.getFunctionType());
      }

      return mlir::WalkResult::advance();
    });
  }
};

} // namespace mlir::tt::stablehlo
