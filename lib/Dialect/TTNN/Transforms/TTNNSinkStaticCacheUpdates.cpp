// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSINKSTATICCACHEUPDATES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static bool isDRAMToMemConfig(ToMemoryConfigOp op) {
  return isDRAMBufferType(op.getMemoryConfig().getBufferType().getValue());
}

static bool isL1ToMemConfig(ToMemoryConfigOp op) {
  return isL1BufferType(op.getMemoryConfig().getBufferType().getValue());
}

} // namespace

class TTNNSinkStaticCacheUpdates
    : public impl::TTNNSinkStaticCacheUpdatesBase<TTNNSinkStaticCacheUpdates> {
public:
  using impl::TTNNSinkStaticCacheUpdatesBase<
      TTNNSinkStaticCacheUpdates>::TTNNSinkStaticCacheUpdatesBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp))
        return;
      sinkCLUpdateCluster(funcOp);
    });
  }

private:
  void sinkCLUpdateCluster(func::FuncOp funcOp) {
    if (funcOp.getBody().empty())
      return;

    auto returnOp = llvm::dyn_cast<func::ReturnOp>(
        funcOp.getBody().front().getTerminator());
    if (!returnOp)
      return;

    // Collect candidate AddOps: reachable from return via DRAM to_memory_config.
    llvm::SmallPtrSet<Operation *, 4> candidateAdds;
    for (Value retVal : returnOp.getOperands()) {
      auto dramCopy = retVal.getDefiningOp<ToMemoryConfigOp>();
      if (!dramCopy || !isDRAMToMemConfig(dramCopy))
        continue;
      auto addOp = dramCopy.getInput().getDefiningOp<AddOp>();
      if (!addOp)
        continue;
      candidateAdds.insert(addOp.getOperation());
    }

    if (candidateAdds.empty())
      return;

    for (Operation *addOpPtr : candidateAdds) {
      auto addOp = llvm::cast<AddOp>(addOpPtr);
      // Verify the add result is exclusively consumed by DRAM to_memory_config.
      llvm::SmallVector<ToMemoryConfigOp> dramCopies;
      bool addResultSafe = true;
      for (auto *user : addOp.getResult().getUsers()) {
        auto toMem = llvm::dyn_cast<ToMemoryConfigOp>(user);
        if (!toMem || !isDRAMToMemConfig(toMem)) {
          addResultSafe = false;
          break;
        }
        dramCopies.push_back(toMem);
      }
      if (!addResultSafe || dramCopies.empty())
        continue;

      // Verify each DRAM copy is exclusively used by the return.
      bool dramCopiesSafe = true;
      for (auto dramCopy : dramCopies) {
        for (auto *user : dramCopy.getResult().getUsers()) {
          if (user != returnOp.getOperation()) {
            dramCopiesSafe = false;
            break;
          }
        }
        if (!dramCopiesSafe)
          break;
      }
      if (!dramCopiesSafe)
        continue;

      // Find the "delta to L1" op: one of add's inputs from a to_memory_config
      // that targets L1. This brings the constant delta tensor into L1 for
      // the add to consume.
      ToMemoryConfigOp deltaToL1;
      for (Value input : {addOp.getLhs(), addOp.getRhs()}) {
        auto toMem = input.getDefiningOp<ToMemoryConfigOp>();
        if (toMem && isL1ToMemConfig(toMem)) {
          deltaToL1 = toMem;
          break;
        }
      }

      // Build the cluster move order: [deltaToL1?], add, [dealloc?], dramCopies
      llvm::SmallVector<Operation *> clusterOrder;

      if (deltaToL1) {
        // Verify deltaToL1 result only goes to the add and at most one dealloc.
        DeallocateOp dealloc;
        bool deltaL1Safe = true;
        for (auto *user : deltaToL1.getResult().getUsers()) {
          if (user == addOp.getOperation())
            continue;
          auto d = llvm::dyn_cast<DeallocateOp>(user);
          if (!d || dealloc) {
            deltaL1Safe = false;
            break;
          }
          dealloc = d;
        }
        if (!deltaL1Safe)
          continue;

        clusterOrder.push_back(deltaToL1.getOperation());
        clusterOrder.push_back(addOp.getOperation());
        if (dealloc)
          clusterOrder.push_back(dealloc.getOperation());
      } else {
        clusterOrder.push_back(addOp.getOperation());
      }

      for (auto dramCopy : dramCopies)
        clusterOrder.push_back(dramCopy.getOperation());

      // Move each cluster op to just before the return, preserving dep order.
      for (auto *op : clusterOrder)
        op->moveBefore(returnOp);
    }
  }
};

} // namespace mlir::tt::ttnn
