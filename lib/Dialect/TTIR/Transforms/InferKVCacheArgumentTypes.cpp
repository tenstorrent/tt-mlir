// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace {

// Returns the cache operand for cache operations, or nullptr if not a cache op.
static mlir::Value getCacheOperand(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
      .Case<mlir::tt::ttir::FillCacheOp>([](auto op) { return op.getCache(); })
      .Case<mlir::tt::ttir::UpdateCacheOp>(
          [](auto op) { return op.getCache(); })
      .Case<mlir::tt::ttir::PagedFillCacheOp>(
          [](auto op) { return op.getCache(); })
      .Case<mlir::tt::ttir::PagedUpdateCacheOp>(
          [](auto op) { return op.getCache(); })
      .Default([](mlir::Operation *) { return nullptr; });
}

// Traces the cache operand back through cache operations to find the original
// function argument. This only follows the cache operand chain, not other
// operands like the input data.
static mlir::BlockArgument traceCacheToBlockArg(mlir::Value cacheOperand) {
  mlir::Value current = cacheOperand;

  while (current) {
    // If we've reached a block argument, return it.
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(current)) {
      return blockArg;
    }

    // If the current value is defined by a cache operation, follow its cache
    // operand.
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return nullptr;
    }

    mlir::Value nextCache = getCacheOperand(defOp);
    if (nextCache) {
      current = nextCache;
    } else {
      // Not a cache operation, stop tracing.
      return nullptr;
    }
  }

  return nullptr;
}

} // namespace

namespace mlir::tt::ttir {

// This pass identifies function arguments used as cache inputs to FillCacheOp,
// UpdateCacheOp, PagedFillCacheOp, or PagedUpdateCacheOp and marks them with
// the ttcore.argument_type = #ttcore.argument_type<kv_cache> attribute.
class TTIRInferKVCacheArgumentTypes
    : public mlir::PassWrapper<TTIRInferKVCacheArgumentTypes,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRInferKVCacheArgumentTypes)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    // Walk through all nested func.func operations (they may be inside nested
    // modules like builtin.module @SyncTensorsGraph.xxx).
    moduleOp.walk([&](func::FuncOp funcOp) {
      llvm::DenseSet<BlockArgument> cacheArgs;

      // Walk all operations and collect cache arguments.
      funcOp.walk([&](Operation *op) {
        Value cacheOperand = getCacheOperand(op);
        if (!cacheOperand) {
          return;
        }

        // Trace the cache operand back to function arguments, following only
        // cache operands through chained cache operations.
        BlockArgument blockArg = traceCacheToBlockArg(cacheOperand);
        if (blockArg && blockArg.getOwner()->getParentOp() == funcOp) {
          cacheArgs.insert(blockArg);
        }
      });

      // Set kv_cache attribute on identified arguments.
      for (auto blockArg : cacheArgs) {
        funcOp.setArgAttr(blockArg.getArgNumber(),
                          ttcore::ArgumentTypeAttr::name,
                          ttcore::ArgumentTypeAttr::get(
                              context, ttcore::ArgumentType::KVCache));
      }
    });
  }

  llvm::StringRef getArgument() const override {
    return "ttir-infer-kv-cache-argument-types";
  }

  llvm::StringRef getDescription() const override {
    return "Infer KV cache argument types from cache operations.";
  }
};

std::unique_ptr<Pass> createTTIRInferKVCacheArgumentTypes() {
  return std::make_unique<TTIRInferKVCacheArgumentTypes>();
}

} // namespace mlir::tt::ttir
