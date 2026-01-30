// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt {
#define GEN_PASS_DEF_EMITPYSPLITFILES
#include "ttmlir/Conversion/Passes.h.inc"

class EmitPySplitFiles : public impl::EmitPySplitFilesBase<EmitPySplitFiles> {

  struct FuncOpAnalyser {
    func::FuncOp funcOp;
    llvm::SetVector<Operation *> constevalDefUseChain;
    llvm::SmallVector<func::CallOp> crossFileCallOps;
    mlir::Value globalDict = nullptr;
    mlir::Value outerDictKey = nullptr;
    mlir::Value deviceArg = nullptr;
    llvm::DenseMap<Operation *, unsigned> opIndices;
  };

public:
  using impl::EmitPySplitFilesBase<EmitPySplitFiles>::EmitPySplitFilesBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    // Create main and consteval file ops.
    builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    auto mainFile = builder.create<emitpy::FileOp>(moduleOp->getLoc(), "main");
    auto constevalFile =
        builder.create<emitpy::FileOp>(moduleOp->getLoc(), "consteval");

    // Categorize top-level module operations.
    ModuleOpAnalyser moduleAnalyser;
    categorizeModuleOperations(moduleOp, moduleAnalyser);

    // Move 'pure' consteval functions (isolated segments of consteval logic) to
    // the consteval file.
    for (auto *funcOp : moduleAnalyser.constevalFuncOps) {
      funcOp->moveBefore(&constevalFile.getBodyRegion().front(),
                         constevalFile.getBodyRegion().front().end());
    }

    llvm::SmallVector<llvm::StringRef> executeFuncNames;
    // Process uncategorized operations (functions that are not 'pure' consteval
    // functions).
    for (auto *uncategorizedOp : moduleAnalyser.uncategorizedOps) {
      if (auto funcOp = dyn_cast<func::FuncOp>(uncategorizedOp)) {
        std::string funcName = funcOp.getName().str();
        FuncOpAnalyser funcOpAnalyser;
        populateFuncOpAnalyser(funcOp, funcOpAnalyser);

        // If there are no consteval operations to process for this function,
        // skip it. This function will be part of the main file in its entirety.
        if (!hasConstevalLogic(funcOpAnalyser)) {
          continue;
        }

        // Replace func::CallOp with emitpy::CallOpaqueOp for cross-file
        // functions calls to avoid symbol resolution errors (i.e. calls to
        // hoisted functions).
        for (auto callOp :
             llvm::make_early_inc_range(funcOpAnalyser.crossFileCallOps)) {
          replaceCallWithCallOpaque(builder, callOp);
        }

        auto strType = emitpy::OpaqueType::get(&getContext(), "str");
        auto tensorListType =
            emitpy::OpaqueType::get(&getContext(), "[ttnn.Tensor]");
        auto deviceType = emitpy::OpaqueType::get(&getContext(), "ttnn.Device");
        auto innerDictType =
            emitpy::DictType::get(&getContext(), strType, tensorListType);
        std::string executeFuncName = "execute_" + funcName + "_consteval";
        executeFuncNames.push_back(
            builder.getStringAttr(executeFuncName).getValue());

        // Create execute function that will store complete consteval logic of
        // the current funcOp. This function will be a part of the
        // consteval file and will be called from the main file.
        auto executeConstevalFunc = createExecuteConstevalFunction(
            builder, funcOpAnalyser, constevalFile);
        populateExecuteConstevalFunctionBody(builder, executeConstevalFunc,
                                             funcOpAnalyser);

        // Insert call of the execute function at the beginning of the original
        // function.
        llvm::SmallVector<Value> executeCallOperands = {funcOp.getArgument(0)};
        if (funcOpAnalyser.deviceArg) {
          executeCallOperands.push_back(funcOpAnalyser.deviceArg);
        }
        auto callExecuteOp =
            createCallAtFunctionStart(builder, funcOp, executeFuncName,
                                      executeCallOperands, innerDictType);

        // Map non-consteval subscript operations to appropriate key constants
        // to transform constEvalFuncWrapperResult[i] ->
        // executeFuncResult[key][i].
        llvm::DenseMap<emitpy::ConstantOp,
                       llvm::SmallVector<emitpy::SubscriptOp>>
            subscriptMapping;

        for (Operation *op : constevalUseDefChain) {
          auto wrapperFuncCall = dyn_cast<emitpy::CallOpaqueOp>(op);
          if (!wrapperFuncCall ||
              !wrapperFuncCall.getCallee().contains("constEvalFuncWrapper")) {
            continue;
          }

          // Get inner dictionary key from the wrapper function operands.
          // The inner dict key is the last operand, unless a device argument
          // is present (target-module=true), in which case it's second-to-last.
          unsigned numOperands = wrapperFuncCall->getNumOperands();
          auto deviceType =
              emitpy::OpaqueType::get(&getContext(), "ttnn.Device");
          auto lastOperand = wrapperFuncCall->getOperand(numOperands - 1);
          unsigned innerDictKeyIdx = (lastOperand.getType() == deviceType)
                                         ? numOperands - 2
                                         : numOperands - 1;
          auto innerDictKeyOp = wrapperFuncCall->getOperand(innerDictKeyIdx)
                                    .getDefiningOp<emitpy::ConstantOp>();
          if (!innerDictKeyOp) {
            wrapperFuncCall->emitError("constEvalFuncWrapper call missing "
                                       "inner dictionary key operand");
            signalPassFailure();
            return;
          }

          // Collect all non-consteval subscripts that use current wrapper
          // function result.
          for (auto &use : wrapperFuncCall->getResult(0).getUses()) {
            if (auto subOp = dyn_cast<emitpy::SubscriptOp>(use.getOwner())) {
              if (!constevalUseDefChain.contains(subOp)) {
                subscriptMapping[innerDictKeyOp].push_back(subOp);
              }
            }
          }
        }

        // Transform collected subscript operations in deterministic order.
        auto sortedEntries = llvm::to_vector(subscriptMapping);
        llvm::sort(sortedEntries, [&](auto &a, auto &b) {
          return opIndices[a.first.getOperation()] >
                 opIndices[b.first.getOperation()];
        });

        for (auto &[innerDictKeyOp, subscripts] : sortedEntries) {
          builder.setInsertionPointAfter(callExecuteOp);
          auto *clonedKeyOp = builder.clone(*innerDictKeyOp.getOperation());
          auto innerDictValue = builder.create<emitpy::GetValueForDictKeyOp>(
              funcOp.getLoc(), tensorListType, callExecuteOp.getResult(0),
              clonedKeyOp->getResult(0));
          for (auto subscriptOp : subscripts) {
            builder.setInsertionPoint(subscriptOp);
            auto newSubscriptOp = builder.create<emitpy::SubscriptOp>(
                subscriptOp.getLoc(), subscriptOp.getType(),
                innerDictValue.getResult(), subscriptOp.getIndex());
            newSubscriptOp->setAttrs(subscriptOp->getAttrs());
            subscriptOp.getResult().replaceAllUsesWith(
                newSubscriptOp.getResult());
            subscriptOp->erase();
          }
        }

        // Erase explicitly marked consteval operations from the original
        // function body.
        for (Operation *op :
             llvm::reverse(funcOpAnalyser.constevalDefUseChain)) {
          op->erase();
        }
      }
    }

    // Create import statement for execute functions in the main file op.
    if (!executeFuncNames.empty()) {
      builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
      builder.create<emitpy::ImportOp>(
          moduleOp.getLoc(), builder.getStringAttr("consteval"),
          /*module_alias=*/nullptr,
          /*members_to_import=*/builder.getStrArrayAttr(executeFuncNames),
          /*member_aliases=*/
          builder.getArrayAttr(llvm::SmallVector<mlir::Attribute>(
              executeFuncNames.size(), builder.getStringAttr(""))),
          /*import_all=*/nullptr);
    }

    // Copy all import ops from the top-level module to the beginning of both
    // file ops. Erase the original import ops from the top-level module.
    for (auto *importOp : llvm::make_early_inc_range(importOps)) {
      builder.setInsertionPointToStart(&constevalFile.getBodyRegion().front());
      builder.clone(*importOp);

      builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
      builder.clone(*importOp);

      importOp->erase();
    }

    // Move uncategorized operations to the main file op.
    for (auto *uncategorizedOp : uncategorizedOps) {
      uncategorizedOp->moveBefore(&mainFile.getBodyRegion().front(),
                                  mainFile.getBodyRegion().front().end());
    }
  }

private:
  struct ModuleOpAnalyser {
    llvm::SmallVector<Operation *> importOps;
    llvm::SmallVector<Operation *> constevalMarkedOps;
    llvm::SmallVector<Operation *> constevalFuncOps;
    llvm::SmallVector<Operation *> uncategorizedOps;
  };

  llvm::SetVector<Operation *> getSortedConstevalUseDefChain(
      const llvm::SmallVector<Operation *> &constevalOps) {
    llvm::SmallVector<Operation *> workload(constevalOps.begin(),
                                            constevalOps.end());
    llvm::SetVector<Operation *> useDefChain;

    while (!workload.empty()) {
      Operation *op = workload.pop_back_val();
      if (!useDefChain.insert(op)) {
        continue;
      }

      for (auto operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          workload.push_back(defOp);
        }
      }
    }

    return topologicalSort(useDefChain);
  }

  void categorizeModuleOperations(ModuleOp moduleOp,
                                  ModuleOpAnalyser &moduleAnalyser) {
    for (auto &op : moduleOp.getOps()) {
      if (isa<emitpy::FileOp>(op)) {
        continue;
      }
      if (isa<emitpy::ImportOp>(op)) {
        moduleAnalyser.importOps.push_back(&op);
        continue;
      }
      if (op.hasAttr("emitpy.consteval")) {
        moduleAnalyser.constevalMarkedOps.push_back(&op);
        continue;
      }
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (ttmlir::utils::isConstEvalFunc(funcOp) ||
            ttmlir::utils::isForwardCPUFunc(funcOp)) {
          moduleAnalyser.constevalFuncOps.push_back(&op);
          continue;
        }
      }
      moduleAnalyser.uncategorizedOps.push_back(&op);
    }
  }

  void populateFuncOpAnalyser(func::FuncOp funcOp,
                              FuncOpAnalyser &funcOpAnalyser) {
    ModuleOp moduleOp = getOperation();
    llvm::SmallVector<Operation *> constevalOps;
    unsigned index = 0;

    funcOpAnalyser.funcOp = funcOp;
    // Search for globalDict and outerDictKey in the function body.
    // Collect all operations that are explicitly marked with attribute as
    // consteval.
    funcOp.walk([&](Operation *op) {
      funcOpAnalyser.opIndices[op] = index++;
      if (isa<emitpy::GlobalStatementOp>(op)) {
        funcOpAnalyser.globalDict = op->getResult(0);
      }
      if (auto constantOp = dyn_cast<emitpy::ConstantOp>(op)) {
        // Look for the constant with a value equal to the function name.
        auto funcNameAttr = emitpy::OpaqueAttr::get(
            &getContext(), "\"" + funcOp.getName().str() + "\"");
        if (constantOp.getValue() == funcNameAttr) {
          funcOpAnalyser.outerDictKey = constantOp->getResult(0);
        }
      }
      if (op->hasAttr("emitpy.consteval")) {
        constevalOps.push_back(op);
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (auto calledFunc =
                moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
          if (ttmlir::utils::isForwardCPUFunc(calledFunc)) {
            funcOpAnalyser.crossFileCallOps.push_back(callOp);
          }
        }
      }
    });

    funcOpAnalyser.constevalDefUseChain =
        getSortedConstevalUseDefChain(constevalOps);

    auto deviceType = emitpy::OpaqueType::get(&getContext(), "ttnn.Device");
    if (funcOp.getNumArguments() > 0) {
      auto lastArg = funcOp.getArgument(funcOp.getNumArguments() - 1);
      if (lastArg.getType() == deviceType) {
        funcOpAnalyser.deviceArg = lastArg;
      }
    }
  }

  bool hasConstevalLogic(const FuncOpAnalyser &funcOpAnalyser) {
    return !funcOpAnalyser.constevalDefUseChain.empty();
  }

  void replaceCallWithCallOpaque(OpBuilder &builder, func::CallOp callOp) {
    builder.setInsertionPoint(callOp);
    auto callOpaqueOp = builder.create<emitpy::CallOpaqueOp>(
        callOp.getLoc(), callOp.getResultTypes(), callOp.getCallee().str(),
        callOp.getOperands());
    callOpaqueOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
    callOp.replaceAllUsesWith(callOpaqueOp);
    callOp->erase();
  }

  func::FuncOp
  createExecuteConstevalFunction(OpBuilder &builder,
                                 const FuncOpAnalyser &funcOpAnalyser,
                                 emitpy::FileOp constevalFile) {
    auto tensorListType =
        emitpy::OpaqueType::get(&getContext(), "[ttnn.Tensor]");
    auto deviceType = emitpy::OpaqueType::get(&getContext(), "ttnn.Device");
    llvm::SmallVector<Type> executeInputTypes = {tensorListType};
    if (funcOpAnalyser.deviceArg) {
      executeInputTypes.push_back(deviceType);
    }
    std::string funcName = funcOpAnalyser.funcOp.getName().str();
    std::string executeFuncName = "execute_" + funcName + "_consteval";
    builder.setInsertionPointToEnd(&constevalFile.getBodyRegion().front());
    auto executeConstevalFunc = builder.create<func::FuncOp>(
        funcOpAnalyser.funcOp.getLoc(), executeFuncName,
        builder.getFunctionType(executeInputTypes, {innerDictType}));
    executeConstevalFunc.addEntryBlock();
    return executeConstevalFunc;
  }

  void
  populateExecuteConstevalFunctionBody(OpBuilder &builder,
                                       func::FuncOp executeConstevalFunc,
                                       const FuncOpAnalyser &funcOpAnalyser) {
    Block &entryBlock = executeConstevalFunc.getBody().front();
    FuncOp funcOp = funcOpAnalyser.funcOp;
    mlir::Value originalDeviceArg = funcOpAnalyser.deviceArg;
    auto strType = emitpy::OpaqueType::get(&getContext(), "str");
    auto tensorListType =
        emitpy::OpaqueType::get(&getContext(), "[ttnn.Tensor]");
    auto innerDictType =
        emitpy::DictType::get(&getContext(), strType, tensorListType);
    // Map original funcOp input to the input of the new
    // execute function.
    mlir::IRMapping mapping;
    mlir::Value originalInput = funcOp.getArgument(0);
    mapping.map(originalInput, entryBlock.getArgument(0));
    if (originalDeviceArg) {
      mapping.map(originalDeviceArg, entryBlock.getArgument(1));
      executeConstevalFunc.setArgAttr(1, "emitpy.name",
                                      builder.getStringAttr("device"));
    }

    // Clone all operations in the consteval use-def chain into the execute
    // function. Operations that are not explicity marked as consteval
    // should stay in the original function and be cloned in the execute
    // function. Operations that are explicitly marked as consteval should
    // be erased from original function.
    builder.setInsertionPointToStart(&entryBlock);
    for (Operation *op : funcOpAnalyser.constevalDefUseChain) {
      Operation *cloned = builder.clone(*op, mapping);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), cloned->getResult(i));
      }
    }

    // Add return operation to the execute function.
    auto dictResult = builder.create<emitpy::GetValueForDictKeyOp>(
        funcOp.getLoc(), innerDictType,
        mapping.lookup(funcOpAnalyser.globalDict),
        mapping.lookup(funcOpAnalyser.outerDictKey));
    builder.create<func::ReturnOp>(funcOp.getLoc(), dictResult.getResult());
  }

  emitpy::CallOpaqueOp createCallAtFunctionStart(
      OpBuilder &builder, func::FuncOp funcOp, std::string calleeName,
      llvm::SmallVector<Value> calleeOperands, Type calleeResultType) {
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    builder.create<emitpy::CallOpaqueOp>(funcOp.getLoc(), calleeResultType,
                                         calleeName, calleeOperands);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createEmitPySplitFilesPass() {
  return std::make_unique<EmitPySplitFiles>();
}

} // namespace mlir::tt
