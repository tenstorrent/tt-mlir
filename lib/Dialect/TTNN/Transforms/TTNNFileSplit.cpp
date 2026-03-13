// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFILESPLIT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
//
// TTNN File Split Pass
//
// This pass organizes TTNN IR into FileOp containers (emitpy::FileOp or
// emitc::FileOp, selected via the `target` pass option) before lowering
// to a target dialect.
//
// It creates two FileOps:
//   @"consteval" - complete consteval logic
//   @"main"      - model logic and calls to the consteval logic
//
// This pass effectively does the following:
//   - Moves the consteval functions to the consteval file.
//   - Moves the cpu-hoisted declarations to the consteval file.
//   - For each forward function with LoadCachedOps, creates a wrapper function
//   in the consteval file that will perform the consteval logic for that
//   function. Inserts a call to the wrapper function at the beginning of the
//   forward function in the main file.
//
//
//===----------------------------------------------------------------------===//

namespace {

constexpr const char *kMainFileName = "main";
constexpr const char *kConstevalFileName = "consteval";
constexpr const char *kConstevalPrefix = "consteval_";

// Collect all ops in the def chain of the given LoadCachedOps.
static llvm::SetVector<Operation *>
collectDefChain(ArrayRef<ttcore::LoadCachedOp> loadCachedOps, Block *block) {
  llvm::SetVector<Operation *> result;
  llvm::SmallVector<Operation *> workload;
  for (auto loadCachedOp : loadCachedOps) {
    for (auto input : loadCachedOp.getInputs()) {
      if (auto *defOp = input.getDefiningOp();
          defOp && defOp->getBlock() == block) {
        workload.push_back(defOp);
      }
    }
  }

  while (!workload.empty()) {
    auto *op = workload.pop_back_val();
    if (op->getBlock() != block || !result.insert(op)) {
      continue;
    }
    for (auto operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        workload.push_back(defOp);
      }
    }
  }
  return result;
}

// Collect the caching-pattern ops that consume the given LoadCachedOps'
// results.
static llvm::SetVector<Operation *>
collectCachingOps(ArrayRef<ttcore::LoadCachedOp> loadCachedOps, Block *block) {
  llvm::SetVector<Operation *> ops;
  for (auto loadCachedOp : loadCachedOps) {
    for (auto result : loadCachedOp.getResults()) {
      for (auto *user : result.getUsers()) {
        if (isa<ttcore::SetKeyValueOp>(user) && user->getBlock() == block) {
          ops.insert(user);
        }
      }
    }
  }
  return ops;
}

class TTNNFileSplit : public impl::TTNNFileSplitBase<TTNNFileSplit> {
public:
  using TTNNFileSplitBase::TTNNFileSplitBase;

  void runOnOperation() override {
    switch (target) {
    case FileSplitTarget::EmitPy:
      runImpl<emitpy::FileOp>();
      break;
    case FileSplitTarget::EmitC:
      runImpl<emitc::FileOp>();
      break;
    }
  }

private:
  template <typename FileOpTy>
  void runImpl() {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    // If there is no consteval logic, do not perform file split.
    bool hasConstevalLogic =
        llvm::any_of(moduleOp.getOps<func::FuncOp>(), [](func::FuncOp funcOp) {
          return ttmlir::utils::isConstEvalFunc(funcOp) ||
                 ttmlir::utils::isForwardCPUDeclarationFunc(funcOp);
        });
    if (!hasConstevalLogic) {
      return;
    }

    // Create the file containers.
    builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    auto mainFile = FileOpTy::create(builder, moduleOp.getLoc(), kMainFileName);
    auto constevalFile =
        FileOpTy::create(builder, moduleOp.getLoc(), kConstevalFileName);

    // Move const-eval functions to the consteval file. Clone
    // CPU-hoisted declarations into both files so that func.call ops
    // in both files can resolve the symbol.
    llvm::SmallVector<func::FuncOp> constevalDefs;
    llvm::SmallVector<func::FuncOp> cpuDecls;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        constevalDefs.push_back(funcOp);
      } else if (ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
        cpuDecls.push_back(funcOp);
      }
    }
    for (auto constevalOp : constevalDefs) {
      constevalOp->moveBefore(&constevalFile.getBodyRegion().front(),
                              constevalFile.getBodyRegion().front().end());
    }
    for (auto cpuDecl : cpuDecls) {
      builder.setInsertionPointToStart(&constevalFile.getBodyRegion().front());
      builder.clone(*cpuDecl.getOperation());
      builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
      builder.clone(*cpuDecl.getOperation());
      cpuDecl->erase();
    }

    // For each forward function with LoadCachedOps, create a wrapper in
    // the consteval file that will perform the const-eval logic.
    // Insert a call to the wrapper function at the beginning of the forward
    // function in the main file.
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      llvm::SmallVector<ttcore::LoadCachedOp> loadCachedOps;
      funcOp.walk(
          [&](ttcore::LoadCachedOp op) { loadCachedOps.push_back(op); });
      if (loadCachedOps.empty()) {
        continue;
      }
      if (failed(createWrapperAndPatch(builder, constevalFile, mainFile, funcOp,
                                       loadCachedOps))) {
        return signalPassFailure();
      }
    }

    // Move remaining top-level operations to the main file.
    for (auto &op :
         llvm::make_early_inc_range(moduleOp.getBody()->getOperations())) {
      if (isa<FileOpTy>(&op)) {
        continue;
      }
      op.moveBefore(&mainFile.getBodyRegion().front(),
                    mainFile.getBodyRegion().front().end());
    }
  }

  template <typename FileOpTy>
  LogicalResult createWrapperAndPatch(
      OpBuilder &builder, FileOpTy constevalFile, FileOpTy mainFile,
      func::FuncOp forwardFunc,
      llvm::SmallVector<ttcore::LoadCachedOp> &loadCachedOps) {

    // Create the wrapper function in the consteval file.
    std::string wrapperName = kConstevalPrefix + forwardFunc.getName().str();
    auto dictType = ttcore::DictType::get(&getContext());
    llvm::SmallVector<Type> wrapperArgTypes;
    wrapperArgTypes.push_back(dictType);
    wrapperArgTypes.append(forwardFunc.getArgumentTypes().begin(),
                           forwardFunc.getArgumentTypes().end());
    auto wrapperFuncType = builder.getFunctionType(wrapperArgTypes, {dictType});

    builder.setInsertionPointToEnd(&constevalFile.getBodyRegion().front());
    auto wrapperFunc = func::FuncOp::create(builder, forwardFunc.getLoc(),
                                            wrapperName, wrapperFuncType);
    wrapperFunc.addEntryBlock();

    Block &forwardBody = forwardFunc.getBody().front();
    Block &wrapperBody = wrapperFunc.getBody().front();
    builder.setInsertionPointToStart(&wrapperBody);

    ttcore::GetGlobalOp cacheDict = nullptr;
    for (auto &op : forwardBody) {
      if (auto getGlobalOp = dyn_cast<ttcore::GetGlobalOp>(&op);
          getGlobalOp && getGlobalOp->hasAttr(kCachingDictAttr)) {
        cacheDict = getGlobalOp;
        getGlobalOp->removeDiscardableAttr(kCachingDictAttr);
        break;
      }
    }

    // Map arguments from the forward function to the wrapper's
    // arguments.
    IRMapping mapping;
    mapping.map(cacheDict.getResult(), wrapperBody.getArgument(0));
    for (unsigned i = 0; i < forwardFunc.getNumArguments(); ++i) {
      wrapperFunc.setArgAttrs(i + 1, forwardFunc.getArgAttrDict(i));
      mapping.map(forwardBody.getArgument(i), wrapperBody.getArgument(i + 1));
    }

    // Collect all ops from the forward function to clone into the wrapper.
    llvm::SetVector<Operation *> opsToClone =
        collectDefChain(loadCachedOps, &forwardBody);
    opsToClone.set_union(loadCachedOps);
    opsToClone.set_union(collectCachingOps(loadCachedOps, &forwardBody));

    for (auto *op : opsToClone) {
      Operation *clonedOp = builder.clone(*op, mapping);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), clonedOp->getResult(i));
      }
    }

    // Create the return operation in the wrapper function.
    func::ReturnOp::create(builder, forwardFunc.getLoc(),
                           ValueRange{mapping.lookup(cacheDict.getResult())});

    // Insert a call to the wrapper function after the cache dictionary
    // retrieval in the forward function.
    builder.setInsertionPointAfter(cacheDict);
    llvm::SmallVector<Value> callArgs;
    callArgs.push_back(cacheDict.getResult());
    callArgs.append(forwardBody.getArguments().begin(),
                    forwardBody.getArguments().end());
    auto callOp =
        func::CallOp::create(builder, forwardFunc.getLoc(), wrapperName,
                             TypeRange{dictType}, callArgs);

    // Update dictionary lookup operations to use the dictionary that wrapper
    // function returns.
    auto dictResult = callOp.getResult(0);
    for (auto getKVOp : forwardBody.getOps<ttcore::GetKeyValueOp>()) {
      getKVOp->setOperand(0, dictResult);
    }

    // Erase cloned ops that are no longer used in the forward body.
    for (auto *op : llvm::reverse(opsToClone)) {
      if (op->use_empty()) {
        op->erase();
      }
    }

    // Create a declaration of the wrapper function in the main file so that
    // func.call op can resolve the symbol.
    builder.setInsertionPointToEnd(&mainFile.getBodyRegion().front());
    auto privateDecl = func::FuncOp::create(builder, forwardFunc.getLoc(),
                                            wrapperName, wrapperFuncType);
    privateDecl.setPrivate();

    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
