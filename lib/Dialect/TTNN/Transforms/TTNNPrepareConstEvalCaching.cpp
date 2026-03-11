// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPARECONSTEVALCACHING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
//
// TTNNPrepareConstEvalCaching Pass
//
// This pass prepares the const-eval results for caching by packing the results
// of all LoadCachedOps into a single global caching dictionary per forward
// function. For each forward function containing LoadCachedOps, it creates a
// global dictionary (e.g., `_cached_forward`), retrieves it at the top of the
// function body, and stores each LoadCachedOp's results under its callee name.
//
//===----------------------------------------------------------------------===//

namespace {

constexpr const char *kCachingDictAttr = "ttcore.caching_dict";
constexpr const char *kWrapperAttr = "consteval_wrapper";

// Collect all ops in the def-use chain of the given LoadCachedOps.
static llvm::SetVector<Operation *>
collectDefUseChain(ArrayRef<ttcore::LoadCachedOp> loadCachedOps, Block *block) {
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

class TTNNPrepareConstEvalCaching
    : public impl::TTNNPrepareConstEvalCachingBase<
          TTNNPrepareConstEvalCaching> {
public:
  using impl::TTNNPrepareConstEvalCachingBase<
      TTNNPrepareConstEvalCaching>::TTNNPrepareConstEvalCachingBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    llvm::SmallVector<func::FuncOp> funcOps(moduleOp.getOps<func::FuncOp>());
    for (auto funcOp : funcOps) {
      llvm::SmallVector<ttcore::LoadCachedOp> loadCachedOps;
      funcOp.walk(
          [&](ttcore::LoadCachedOp op) { loadCachedOps.push_back(op); });
      if (loadCachedOps.empty()) {
        continue;
      }

      auto dictType = ttcore::DictType::get(&getContext());
      std::string cacheName = kCachePrefix + funcOp.getName().str();

      // Create the global caching dictionary before the function.
      builder.setInsertionPoint(funcOp);
      builder.create<ttcore::GlobalOp>(funcOp.getLoc(),
                                       llvm::StringRef(cacheName), dictType,
                                       /*index=*/IntegerAttr());

      // Retrieve the caching dictionary at the top of the function body.
      Block &entryBlock = funcOp.getBody().front();
      builder.setInsertionPointToStart(&entryBlock);
      auto dict = builder.create<ttcore::GetGlobalOp>(funcOp.getLoc(), dictType,
                                                      cacheName);
      dict->setDiscardableAttr(kCachingDictAttr, builder.getUnitAttr());

      // Create the wrapper function that will contain the complete const-eval
      // logic for the forward function.
      if (failed(createWrapper(builder, funcOp, loadCachedOps,
                               dict.getResult()))) {
        return signalPassFailure();
      }

      // Insert a call to the wrapper function after the cache dictionary
      // retrieval in the forward function.
      builder.setInsertionPointAfter(dict);
      Block &forwardBody = funcOp.getBody().front();
      llvm::SmallVector<Value> callArgs;
      callArgs.push_back(dict.getResult());
      callArgs.append(forwardBody.getArguments().begin(),
                      forwardBody.getArguments().end());
      auto cacheDict = builder.create<func::CallOp>(
          funcOp.getLoc(), "consteval_" + funcOp.getName().str(),
          TypeRange{dictType}, callArgs);

      // Replace each LoadCachedOp with a dictionary lookup. The results of
      // the LoadCachedOp are stored under one key in the caching dictionary.
      for (auto loadCachedOp : loadCachedOps) {
        builder.setInsertionPointAfter(loadCachedOp);
        auto getKVOp = builder.create<ttcore::GetKeyValueOp>(
            loadCachedOp.getLoc(), loadCachedOp.getResultTypes(),
            cacheDict.getResult(0),
            builder.getStringAttr(loadCachedOp.getCallee()));
        for (unsigned i = 0; i < loadCachedOp->getNumResults(); ++i) {
          loadCachedOp->getResult(i).replaceAllUsesWith(getKVOp.getResult(i));
        }
      }

      // Erase the original LoadCachedOps and their now-unused def-chain ops
      // from the forward function body.
      llvm::SetVector<Operation *> defChainOps =
          collectDefUseChain(loadCachedOps, &forwardBody);
      for (auto loadCachedOp : loadCachedOps) {
        loadCachedOp->erase();
      }
      for (auto *op : llvm::reverse(defChainOps)) {
        if (op->use_empty()) {
          op->erase();
        }
      }
    }
  }

  LogicalResult
  createWrapper(OpBuilder &builder, func::FuncOp forwardFunc,
                llvm::SmallVector<ttcore::LoadCachedOp> &loadCachedOps,
                Value cacheDict) {

    // Create the wrapper function.
    std::string wrapperName = "consteval_" + forwardFunc.getName().str();
    auto dictType = ttcore::DictType::get(&getContext());
    llvm::SmallVector<Type> wrapperArgTypes;
    wrapperArgTypes.push_back(dictType);
    wrapperArgTypes.append(forwardFunc.getArgumentTypes().begin(),
                           forwardFunc.getArgumentTypes().end());
    auto wrapperFuncType = builder.getFunctionType(wrapperArgTypes, {dictType});

    builder.setInsertionPointAfter(forwardFunc);
    auto wrapperFunc = builder.create<func::FuncOp>(
        forwardFunc.getLoc(), wrapperName, wrapperFuncType);
    wrapperFunc.addEntryBlock();
    wrapperFunc->setDiscardableAttr(kWrapperAttr, builder.getUnitAttr());

    Block &forwardBody = forwardFunc.getBody().front();
    Block &wrapperBody = wrapperFunc.getBody().front();
    builder.setInsertionPointToStart(&wrapperBody);

    // Map arguments from the forward function to the wrapper's
    // arguments.
    IRMapping mapping;
    mapping.map(cacheDict, wrapperBody.getArgument(0));
    for (unsigned i = 0; i < forwardFunc.getNumArguments(); ++i) {
      wrapperFunc.setArgAttrs(i + 1, forwardFunc.getArgAttrDict(i));
      mapping.map(forwardBody.getArgument(i), wrapperBody.getArgument(i + 1));
    }

    // Collect all ops from the forward function to clone into the wrapper.
    llvm::SetVector<Operation *> opsToClone =
        collectDefUseChain(loadCachedOps, &forwardBody);
    opsToClone.set_union(loadCachedOps);

    for (auto *op : opsToClone) {
      Operation *clonedOp = builder.clone(*op, mapping);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), clonedOp->getResult(i));
      }
    }

    // For each LoadCachedOp, store its results under one key in the caching
    // dictionary. The key is the callee name of the LoadCachedOp.
    for (auto loadCachedOp : loadCachedOps) {
      auto *clonedLoadCachedOp = mapping.lookup(loadCachedOp);
      builder.setInsertionPointAfter(clonedLoadCachedOp);
      builder.create<ttcore::SetKeyValueOp>(
          clonedLoadCachedOp->getLoc(), mapping.lookup(cacheDict),
          builder.getStringAttr(loadCachedOp.getCallee()),
          clonedLoadCachedOp->getResults());
    }

    // Create the return operation in the wrapper function.
    builder.create<func::ReturnOp>(forwardFunc.getLoc(),
                                   ValueRange{mapping.lookup(cacheDict)});

    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
