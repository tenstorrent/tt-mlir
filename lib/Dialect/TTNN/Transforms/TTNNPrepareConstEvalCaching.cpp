// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

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
class TTNNPrepareConstEvalCaching
    : public impl::TTNNPrepareConstEvalCachingBase<
          TTNNPrepareConstEvalCaching> {
public:
  using impl::TTNNPrepareConstEvalCachingBase<
      TTNNPrepareConstEvalCaching>::TTNNPrepareConstEvalCachingBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      llvm::SmallVector<ttcore::LoadCachedOp> loadCachedOps;
      funcOp.walk(
          [&](ttcore::LoadCachedOp op) { loadCachedOps.push_back(op); });
      if (loadCachedOps.empty()) {
        continue;
      }

      auto dictType = ttcore::DictType::get(&getContext());
      std::string cacheName = "_cached_" + funcOp.getName().str();

      // Create the global caching dictionary before the function.
      builder.setInsertionPoint(funcOp);
      builder.create<ttcore::GlobalOp>(funcOp.getLoc(),
                                       llvm::StringRef(cacheName), dictType,
                                       /*index=*/IntegerAttr());

      // Retrieve the caching dictionary at the top of the function body.
      Block &entryBlock = funcOp.getBody().front();
      builder.setInsertionPointToStart(&entryBlock);
      auto dictVal = builder.create<ttcore::GetGlobalOp>(funcOp.getLoc(),
                                                         dictType, cacheName);
      dictVal->setDiscardableAttr("ttcore.caching_dict", builder.getUnitAttr());

      // For each LoadCachedOp, store its results under one key in the caching
      // dictionary. The key is the callee name of the LoadCachedOp.
      // Replace LoadCachedOp results with dictionary lookups.
      for (auto loadCachedOp : loadCachedOps) {
        builder.setInsertionPointAfter(loadCachedOp);
        auto setKVOp = builder.create<ttcore::SetKeyValueOp>(
            loadCachedOp.getLoc(), dictVal.getResult(),
            builder.getStringAttr(loadCachedOp.getCallee()),
            loadCachedOp.getResults());
        auto getKVOp = builder.create<ttcore::GetKeyValueOp>(
            loadCachedOp.getLoc(), loadCachedOp.getResultTypes(),
            dictVal.getResult(),
            builder.getStringAttr(loadCachedOp.getCallee()));
        for (unsigned i = 0; i < loadCachedOp->getNumResults(); ++i) {
          loadCachedOp->getResult(i).replaceAllUsesExcept(getKVOp.getResult(i),
                                                          setKVOp);
        }
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
