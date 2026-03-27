// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRINFERKVCACHEARGUMENTTYPES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRInferKVCacheArgumentTypes
    : public impl::TTIRInferKVCacheArgumentTypesBase<
          TTIRInferKVCacheArgumentTypes> {
public:
  using impl::TTIRInferKVCacheArgumentTypesBase<
      TTIRInferKVCacheArgumentTypes>::TTIRInferKVCacheArgumentTypesBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    moduleOp.walk([&](func::FuncOp funcOp) {
      llvm::DenseSet<BlockArgument> cacheArgs;

      funcOp.walk([&](CacheOpInterface cacheOp) {
        mlir::Value cacheValue = cacheOp.getCache();

        // Direct BlockArgument → CacheOp connection.
        if (auto blockArg = llvm::dyn_cast<BlockArgument>(cacheValue)) {
          if (blockArg.getOwner()->getParentOp() == funcOp) {
            cacheArgs.insert(blockArg);
          }
          return;
        }

        // In multichip workloads, KV cache args pass through mesh_shard
        // (full_to_shard) before reaching cache ops. Look through mesh_shard
        // to find the originating BlockArgument.
        if (auto *defOp = cacheValue.getDefiningOp()) {
          if (auto meshShardOp = mlir::dyn_cast<MeshShardOp>(defOp)) {
            if (auto blockArg =
                    llvm::dyn_cast<BlockArgument>(meshShardOp.getInput())) {
              if (blockArg.getOwner()->getParentOp() == funcOp) {
                cacheArgs.insert(blockArg);
              }
            }
          }
        }
      });

      for (auto blockArg : cacheArgs) {
        funcOp.setArgAttr(blockArg.getArgNumber(), ttcore::g_kvCacheAttrName,
                          mlir::UnitAttr::get(context));
      }
    });
  }
};

} // namespace mlir::tt::ttir
