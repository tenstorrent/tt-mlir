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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt {
#define GEN_PASS_DEF_EMITPYSPLITFILES
#include "ttmlir/Conversion/Passes.h.inc"

class EmitPySplitFiles : public impl::EmitPySplitFilesBase<EmitPySplitFiles> {

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

    llvm::SmallVector<Operation *> importOps;
    llvm::SmallVector<Operation *> uncategorizedOps;

    // Categorize top-level module operations.
    for (auto &op : llvm::make_early_inc_range(moduleOp.getOps())) {
      if (isa<emitpy::FileOp>(op)) {
        continue;
      }
      if (isa<emitpy::ImportOp>(op)) {
        importOps.push_back(&op);
        continue;
      }
      if (op.hasAttr("emitpy.consteval")) {
        op.moveBefore(&constevalFile.getBodyRegion().front(),
                      constevalFile.getBodyRegion().front().end());
        continue;
      }
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (ttmlir::utils::isConstEvalFunc(funcOp) ||
            ttmlir::utils::isForwardCPUFunc(funcOp)) {
          op.moveBefore(&constevalFile.getBodyRegion().front(),
                        constevalFile.getBodyRegion().front().end());
          continue;
        }
      }
      uncategorizedOps.push_back(&op);
    }

    llvm::SmallVector<llvm::StringRef> executeFuncNames;
    // Process uncategorized operations.
    for (auto *uncategorizedOp : uncategorizedOps) {
      if (auto funcOp = dyn_cast<func::FuncOp>(uncategorizedOp)) {
        std::string funcName = funcOp.getName().str();
        llvm::SmallVector<Operation *> constevalOps;
        llvm::SmallVector<func::CallOp> hoistedCallOps;
        mlir::Value globalDict = nullptr;
        mlir::Value outerDictKey = nullptr;

        // Assign sequential indices to operations for deterministic ordering.
        llvm::DenseMap<Operation *, unsigned> opIndices;
        unsigned index = 0;

        // Search for globalDict and outerDictKey in the function body.
        // Collect all operations that are explicitly marked with attribute as
        // consteval.
        funcOp.walk([&](Operation *op) {
          opIndices[op] = index++;
          if (isa<emitpy::GlobalStatementOp>(op)) {
            globalDict = op->getResult(0);
          }
          if (auto constantOp = dyn_cast<emitpy::ConstantOp>(op)) {
            // Look for the constant with a value equal to the function name
            auto funcNameAttr =
                emitpy::OpaqueAttr::get(&getContext(), "\"" + funcName + "\"");
            if (constantOp.getValue() == funcNameAttr) {
              outerDictKey = constantOp->getResult(0);
            }
          }
          if (op->hasAttr("emitpy.consteval")) {
            constevalOps.push_back(op);
          }
          if (auto callOp = dyn_cast<func::CallOp>(op)) {
            if (auto calledFunc = constevalFile.lookupSymbol<func::FuncOp>(
                    callOp.getCallee())) {
              if (ttmlir::utils::isForwardCPUFunc(calledFunc)) {
                hoistedCallOps.push_back(callOp);
              }
            }
          }
        });

        // Replace func::CallOp with emitpy::CallOpaqueOp for hoisted functions
        // to avoid symbol resolution errors.
        for (auto callHoistedOp : llvm::make_early_inc_range(hoistedCallOps)) {
          builder.setInsertionPoint(callHoistedOp);
          auto callOpaqueHoistedOp = builder.create<emitpy::CallOpaqueOp>(
              callHoistedOp.getLoc(), callHoistedOp.getResultTypes(),
              callHoistedOp.getCallee().str(), callHoistedOp.getOperands());
          callOpaqueHoistedOp->setDiscardableAttrs(
              callHoistedOp->getDiscardableAttrDictionary());
          callHoistedOp.replaceAllUsesWith(callOpaqueHoistedOp);
          callHoistedOp->erase();
        }

        // If there are no globalDict or outerDictKey, there are no
        // consteval operations to process for this function.
        if (!globalDict || !outerDictKey) {
          continue;
        }

        auto strType = emitpy::OpaqueType::get(&getContext(), "str");
        auto tensorListType =
            emitpy::OpaqueType::get(&getContext(), "[ttnn.Tensor]");
        auto innerDictType =
            emitpy::DictType::get(&getContext(), strType, tensorListType);
        std::string executeFuncName = "execute_" + funcName + "_consteval";
        executeFuncNames.push_back(
            builder.getStringAttr(executeFuncName).getValue());

        // Create execute function that will store complete consteval logic of
        // the current funcOp. This function will be a part of the
        // consteval file and will be called from the main file.
        builder.setInsertionPointToEnd(&constevalFile.getBodyRegion().front());
        auto executeConstevalFunc = builder.create<func::FuncOp>(
            funcOp.getLoc(), executeFuncName,
            builder.getFunctionType({tensorListType}, {innerDictType}));
        executeConstevalFunc.addEntryBlock();
        Block &entryBlock = executeConstevalFunc.getBody().front();

        // Map original funcOp input to the input of the new execute function.
        mlir::IRMapping mapping;
        mlir::Value originalInput = funcOp.getArgument(0);
        mapping.map(originalInput, entryBlock.getArgument(0));

        auto constevalUseDefChain = getSortedConstevalUseDefChain(constevalOps);

        // Clone all operations in the consteval use-def chain into the execute
        // function. Operations that are not explicity marked as consteval
        // should stay in the original function and be cloned in the execute
        // function. Operations that are explicitly marked as consteval should
        // be erased from original function.
        builder.setInsertionPointToStart(&entryBlock);
        for (Operation *op : constevalUseDefChain) {
          Operation *cloned = builder.clone(*op, mapping);
          for (unsigned i = 0; i < op->getNumResults(); ++i) {
            mapping.map(op->getResult(i), cloned->getResult(i));
          }
        }

        // Add return operation to the execute function.
        auto dictResult = builder.create<emitpy::GetValueForDictKeyOp>(
            funcOp.getLoc(), innerDictType, mapping.lookup(globalDict),
            mapping.lookup(outerDictKey));
        builder.create<func::ReturnOp>(funcOp.getLoc(), dictResult.getResult());

        // Insert call of the execute function at the beginning of the original
        // function.
        builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
        auto callExecuteOp = builder.create<emitpy::CallOpaqueOp>(
            funcOp.getLoc(), innerDictType, executeFuncName,
            mlir::ValueRange{originalInput});

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

          // Get inner dictionary key from the last operand of the
          // wrapper function.
          auto innerDictKeyOp =
              wrapperFuncCall->getOperand(wrapperFuncCall->getNumOperands() - 1)
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
        for (Operation *op : llvm::reverse(constevalOps)) {
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
};

std::unique_ptr<OperationPass<ModuleOp>> createEmitPySplitFilesPass() {
  return std::make_unique<EmitPySplitFiles>();
}

} // namespace mlir::tt
