// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRINFERSTATICCACHECUMULATIVELENGTHARGS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Walk back from a cache op's update_index operand through TM-like ops to
// collect any entry-block arguments that feed into it. The cumulative_length
// arg sits behind some combination of mesh_shard / broadcast / repeat /
// reshape / typecast and an add with an arange-derived offset.
static void collectArgsFeedingValue(Value start, func::FuncOp funcOp,
                                    llvm::DenseSet<BlockArgument> &out) {
  Block &entry = funcOp.getBody().front();
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(start);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) {
      continue;
    }

    if (auto blockArg = llvm::dyn_cast<BlockArgument>(v)) {
      if (blockArg.getOwner() == &entry) {
        out.insert(blockArg);
      }
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      continue;
    }

    // Look through TM-like single-operand ops.
    if (isa<MeshShardOp, BroadcastOp, RepeatOp, ReshapeOp, TypecastOp>(defOp)) {
      worklist.push_back(defOp->getOperand(0));
      continue;
    }

    // For add, the cumulative_length sits on one side; the other side is
    // typically an arange or constant offset. Push both and let recursion
    // terminate on non-arg leaves.
    if (auto addOp = llvm::dyn_cast<AddOp>(defOp)) {
      worklist.push_back(addOp.getLhs());
      worklist.push_back(addOp.getRhs());
      continue;
    }

    // Anything else (arange, full, constant, arbitrary compute) terminates
    // this branch.
  }
}

// Return the update-index operand of a cache op that takes one, or null if
// the op doesn't expose one. Only update-style cache ops carry a position.
static Value getCacheUpdateIndex(CacheOpInterface cacheOp) {
  if (auto op = llvm::dyn_cast<UpdateCacheOp>(cacheOp.getOperation())) {
    return op.getUpdateIndex();
  }
  if (auto op = llvm::dyn_cast<PagedUpdateCacheOp>(cacheOp.getOperation())) {
    return op.getUpdateIndex();
  }
  return {};
}

} // namespace

class TTIRInferStaticCacheCumulativeLengthArgs
    : public impl::TTIRInferStaticCacheCumulativeLengthArgsBase<
          TTIRInferStaticCacheCumulativeLengthArgs> {
public:
  using impl::TTIRInferStaticCacheCumulativeLengthArgsBase<
      TTIRInferStaticCacheCumulativeLengthArgs>::
      TTIRInferStaticCacheCumulativeLengthArgsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.getBody().empty()) {
        return;
      }

      llvm::DenseSet<BlockArgument> cumLenArgs;

      funcOp.walk([&](CacheOpInterface cacheOp) {
        Value updateIndex = getCacheUpdateIndex(cacheOp);
        if (!updateIndex) {
          return;
        }
        collectArgsFeedingValue(updateIndex, funcOp, cumLenArgs);
      });

      for (BlockArgument blockArg : cumLenArgs) {
        funcOp.setArgAttr(blockArg.getArgNumber(),
                          ttcore::g_cumulativeLengthAttrName,
                          mlir::UnitAttr::get(context));
      }
    });
  }
};

} // namespace mlir::tt::ttir
