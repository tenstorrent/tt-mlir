// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRINFERKVCACHEARGUMENTTYPES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Returns the cache operand for cache operations, or nullptr if not a cache op.
Value getCacheOperand(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<FillCacheOp>([](auto op) { return op.getCache(); })
      .Case<UpdateCacheOp>([](auto op) { return op.getCache(); })
      .Case<PagedFillCacheOp>([](auto op) { return op.getCache(); })
      .Case<PagedUpdateCacheOp>([](auto op) { return op.getCache(); })
      .Default([](Operation *) { return nullptr; });
}

// Traces the cache operand back through cache operations to find the original
// function argument.
BlockArgument traceCacheToBlockArg(Value cacheOperand) {
  Value current = cacheOperand;

  while (current) {
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(current)) {
      return blockArg;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return nullptr;
    }

    Value nextCache = getCacheOperand(defOp);
    if (nextCache) {
      current = nextCache;
    } else {
      return nullptr;
    }
  }

  return nullptr;
}

} // namespace

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

      funcOp.walk([&](Operation *op) {
        Value cacheOperand = getCacheOperand(op);
        if (!cacheOperand) {
          return;
        }

        BlockArgument blockArg = traceCacheToBlockArg(cacheOperand);
        if (blockArg && blockArg.getOwner()->getParentOp() == funcOp) {
          cacheArgs.insert(blockArg);
        }
      });

      for (auto blockArg : cacheArgs) {
        funcOp.setArgAttr(blockArg.getArgNumber(),
                          ttcore::ArgumentTypeAttr::name,
                          ttcore::ArgumentTypeAttr::get(
                              context, ttcore::ArgumentType::KVCache));
      }
    });
  }
};

} // namespace mlir::tt::ttir
