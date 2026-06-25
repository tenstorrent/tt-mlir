// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRKVCACHEWRITEPATHNORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Returns the norm op if `v` is directly used as input to an RMSNormOp.
static RMSNormOp rmsNormConsumer(Value v) {
  for (OpOperand &use : v.getUses()) {
    auto rmsNorm = mlir::dyn_cast<RMSNormOp>(use.getOwner());
    if (rmsNorm && rmsNorm.getInput() == v) {
      return rmsNorm;
    }
  }
  return nullptr;
}

// Returns the unique RMSNormOp on the read path of readValue, or nullptr if
// absent or ambiguous.  The pattern is: readValue [→ reshape] → rms_norm.
static RMSNormOp findRmsNormOnReadPath(Value readValue) {
  RMSNormOp found = nullptr;
  auto matchUnique = [&](RMSNormOp candidate) -> bool {
    if (!candidate) {
      return true;
    }
    if (found && found != candidate) {
      return false;
    }
    found = candidate;
    return true;
  };

  for (OpOperand &use : readValue.getUses()) {
    Operation *op = use.getOwner();
    if (auto rmsNorm = mlir::dyn_cast<RMSNormOp>(op)) {
      if (!matchUnique(rmsNorm.getInput() == readValue ? rmsNorm : nullptr)) {
        return nullptr;
      }
      continue;
    }
    if (mlir::isa<ReshapeOp>(op)) {
      if (!matchUnique(rmsNormConsumer(op->getResult(0)))) {
        return nullptr;
      }
      continue;
    }
    // Prefill uses chained fill_cache ops: fill_cache(fill_cache(...), chunk).
    // When readValue is used as the cache operand (index 0) of a functional
    // cache op, follow the result to continue searching the chain.
    if (mlir::isa<FillCacheOp, UpdateCacheOp>(op) && op->getNumResults() > 0 &&
        use.getOperandNumber() == 0) {
      if (!matchUnique(findRmsNormOnReadPath(op->getResult(0)))) {
        return nullptr;
      }
    }
  }
  return found;
}

// Follows the cache operand of chained functional cache ops back to the root
// BlockArgument (e.g. fill_cache(fill_cache(arg, ...), ...) → arg).
static BlockArgument findRootCacheArg(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (auto op = mlir::dyn_cast<FillCacheOp>(defOp)) {
      v = op.getCache();
    } else if (auto op = mlir::dyn_cast<UpdateCacheOp>(defOp)) {
      v = op.getCache();
    } else {
      return nullptr;
    }
  }
  return mlir::dyn_cast<BlockArgument>(v);
}

// Hoists rms_norm from the KV cache read path to every cache write op that
// writes to the same cache argument.
class TTIRKVCacheWritePathNormPass
    : public impl::TTIRKVCacheWritePathNormBase<TTIRKVCacheWritePathNormPass> {
public:
  void runOnOperation() final {
    mlir::OpBuilder builder(&getContext());
    getOperation().walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return;
      }

      llvm::DenseMap<RMSNormOp, llvm::SmallVector<Operation *>> byNorm;

      auto collect = [&](auto cacheOp) {
        BlockArgument cacheArg = findRootCacheArg(cacheOp.getCache());
        if (!cacheArg) {
          return;
        }
        if (!ttcore::isKVCacheArgument(funcOp, cacheArg.getArgNumber())) {
          return;
        }
        // Functional ops (update_cache, fill_cache) return the updated tensor;
        // reads go through the result, not cacheArg directly.
        Value readValue = cacheOp->getNumResults() > 0 ? cacheOp->getResult(0)
                                                       : Value(cacheArg);
        if (RMSNormOp rmsNorm = findRmsNormOnReadPath(readValue)) {
          byNorm[rmsNorm].push_back(cacheOp.getOperation());
        }
      };
      funcOp.walk([&](FillCacheOp op) { collect(op); });
      funcOp.walk([&](UpdateCacheOp op) { collect(op); });
      funcOp.walk([&](PagedFillCacheOp op) { collect(op); });
      funcOp.walk([&](PagedUpdateCacheOp op) { collect(op); });

      for (auto &normEntry : byNorm) {
        RMSNormOp rmsNorm = normEntry.first;
        for (Operation *writeOp : normEntry.second) {
          llvm::TypeSwitch<Operation *, void>(writeOp)
              .Case<FillCacheOp, UpdateCacheOp, PagedFillCacheOp,
                    PagedUpdateCacheOp>([&](auto op) {
                Value fillValue = op.getInput();
                builder.setInsertionPoint(writeOp);
                Operation *prenorm = builder.clone(*rmsNorm);
                mlir::cast<RMSNormOp>(prenorm).getInputMutable().assign(
                    fillValue);
                prenorm->getResult(0).setType(fillValue.getType());
                op.getInputMutable().assign(prenorm->getResult(0));
              });
        }

        rmsNorm.getResult().replaceAllUsesWith(rmsNorm.getInput());
        rmsNorm->erase();
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttir
