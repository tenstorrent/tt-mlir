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
// This pass prepares const-eval results for caching by consolidating all
// LoadCachedOps within each forward function into a single global cache
// dictionary. For each forward function containing LoadCachedOps, it:
//
//   1. Creates a global cache dictionary (e.g., `_cached_forward`).
//   2. Inserts a retrieval of the dictionary at the top of the forward function
//   body.
//   3. Creates a separate consteval wrapper function and moves the complete
//   const-eval logic of that forward function into it.
//   4. Inserts a call to the consteval wrapper function after the dictionary
//   retrieval in the forward function body. The consteval wrapper function
//   receives the global cache dictionary and the forward function inputs as
//   arguments.
//
//===----------------------------------------------------------------------===//

namespace {

constexpr const char *kCachePrefix = "_cached_";
constexpr const char *kConstEvalWrapperNamePrefix = "consteval_";

// Collect the sorteddef-use chain of the given ops.
static llvm::SmallVector<Operation *>
collectSortedDefUseChain(llvm::SmallVector<Operation *> &ops, Block *block) {
  llvm::SetVector<Operation *> result;
  llvm::SmallVector<Operation *> worklist(ops.begin(), ops.end());

  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    if (op->getBlock() != block || !result.insert(op)) {
      continue;
    }
    for (auto operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        worklist.push_back(defOp);
      }
    }
  }

  llvm::SmallVector<Operation *> sortedResult(result.begin(), result.end());
  llvm::sort(sortedResult,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  return sortedResult;
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

      // Create the global cache dictionary before the function.
      builder.setInsertionPoint(funcOp);
      ttcore::GlobalOp::create(builder, funcOp.getLoc(),
                               llvm::StringRef(cacheName), dictType,
                               /*index=*/IntegerAttr());

      // Retrieve the cache dictionary at the top of the function body.
      Block &entryBlock = funcOp.getBody().front();
      builder.setInsertionPointToStart(&entryBlock);
      auto dict = ttcore::GetGlobalOp::create(builder, funcOp.getLoc(),
                                              dictType, cacheName);
      dict->setDiscardableAttr(kCacheDictAttr, builder.getUnitAttr());

      // Create the wrapper function that will encapsulate the complete
      // const-eval logic from the forward function.
      if (failed(createConstEvalWrapper(builder, funcOp, loadCachedOps,
                                        dict.getResult()))) {
        return signalPassFailure();
      }

      // Insert a call to the consteval wrapper function after the cache
      // dictionary retrieval in the forward function.
      builder.setInsertionPointAfter(dict);
      Block &forwardBody = funcOp.getBody().front();
      llvm::SmallVector<Value> callArgs;
      callArgs.push_back(dict.getResult());
      callArgs.append(forwardBody.getArguments().begin(),
                      forwardBody.getArguments().end());
      auto cacheDict = func::CallOp::create(builder, funcOp.getLoc(),
                                            kConstEvalWrapperNamePrefix +
                                                funcOp.getName().str(),
                                            TypeRange{dictType}, callArgs);

      // Replace each LoadCachedOp with a dictionary lookup. The results of
      // the LoadCachedOp are stored under one key in the cache dictionary.
      for (auto loadCachedOp : loadCachedOps) {
        builder.setInsertionPointAfter(loadCachedOp);
        auto getKVOp = ttcore::GetKeyValueOp::create(
            builder, loadCachedOp.getLoc(), loadCachedOp.getResultTypes(),
            cacheDict.getResult(0),
            builder.getStringAttr(loadCachedOp.getCallee()));
        for (unsigned i = 0; i < loadCachedOp->getNumResults(); ++i) {
          loadCachedOp->getResult(i).replaceAllUsesWith(getKVOp.getResult(i));
        }
      }

      // Erase the LoadCachedOps and the unused ops in their def-use chain
      // from the forward function body.
      llvm::SmallVector<Operation *> loadCachedPtrs(loadCachedOps.begin(),
                                                    loadCachedOps.end());
      llvm::SmallVector<Operation *> defChainOps =
          collectSortedDefUseChain(loadCachedPtrs, &forwardBody);
      for (auto *op : llvm::reverse(defChainOps)) {
        if (op->use_empty()) {
          op->erase();
        }
      }
    }
  }

  LogicalResult
  createConstEvalWrapper(OpBuilder &builder, func::FuncOp forwardFunc,
                         llvm::SmallVector<ttcore::LoadCachedOp> &loadCachedOps,
                         Value cacheDict) {
    // Create the consteval wrapper function.
    std::string wrapperName =
        kConstEvalWrapperNamePrefix + forwardFunc.getName().str();
    auto dictType = ttcore::DictType::get(&getContext());
    llvm::SmallVector<Type> wrapperArgTypes;
    wrapperArgTypes.push_back(dictType);
    wrapperArgTypes.append(forwardFunc.getArgumentTypes().begin(),
                           forwardFunc.getArgumentTypes().end());
    auto wrapperFuncType = builder.getFunctionType(wrapperArgTypes, {dictType});

    builder.setInsertionPointAfter(forwardFunc);
    auto wrapperFunc = func::FuncOp::create(builder, forwardFunc.getLoc(),
                                            wrapperName, wrapperFuncType);
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
    llvm::SmallVector<Operation *> loadCachedPtrs(loadCachedOps.begin(),
                                                  loadCachedOps.end());
    llvm::SmallVector<Operation *> opsToClone =
        collectSortedDefUseChain(loadCachedPtrs, &forwardBody);

    for (auto *op : opsToClone) {
      builder.clone(*op, mapping);
    }

    // For each LoadCachedOp, store its results under one key in the cache
    // dictionary. The key is the callee name of the LoadCachedOp.
    for (auto loadCachedOp : loadCachedOps) {
      auto *clonedLoadCachedOp = mapping.lookup(loadCachedOp);
      builder.setInsertionPointAfter(clonedLoadCachedOp);
      ttcore::SetKeyValueOp::create(
          builder, clonedLoadCachedOp->getLoc(), mapping.lookup(cacheDict),
          builder.getStringAttr(loadCachedOp.getCallee()),
          clonedLoadCachedOp->getResults());
    }

    // Create the return operation in the wrapper function.
    builder.setInsertionPointToEnd(&wrapperBody);
    func::ReturnOp::create(builder, forwardFunc.getLoc(),
                           ValueRange{mapping.lookup(cacheDict)});

    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
