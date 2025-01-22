// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include <llvm/Support/Casting.h>

namespace mlir::tt {
#define GEN_PASS_DEF_TTWRAPDEVICEMODULEPASS
#include "ttmlir/Dialect/TT/Transforms/Passes.h.inc"
class TTWrapDeviceModulePass
    : public impl::TTWrapDeviceModulePassBase<TTWrapDeviceModulePass> {
public:
  using impl::TTWrapDeviceModulePassBase<
      TTWrapDeviceModulePass>::TTWrapDeviceModulePassBase;
  void runOnOperation() override {
    ModuleOp rootModule = getOperation();

    // ensure we only lower top-level module, not any nested modules
    if (rootModule->getParentOp() != nullptr) {
      return;
    }
    // Check if module already contains a DeviceModuleOp, and do nothing if so.
    for (Operation &op : rootModule.getBodyRegion().front()) {
      if (isa<tt::DeviceModuleOp>(op)) {
        return;
      }
    }

    // Create new DeviceModuleOp
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&rootModule.getBodyRegion().front());
    auto deviceModule = builder.create<tt::DeviceModuleOp>(rootModule.getLoc());

    // Create nested ModuleOp inside DeviceModuleOp
    builder.setInsertionPointToStart(&deviceModule.getBodyRegion().front());
    auto nestedModule = builder.create<ModuleOp>(rootModule.getLoc());

    // Move all operations from top-level module to nested module
    Block *nestedBlock = &nestedModule.getBodyRegion().front();
    Block *originalBlock = &rootModule.getBodyRegion().front();
    // Since we'll be moving ops which could invalidate iterators,
    // collect ops to move first.
    SmallVector<Operation *> opsToMove;
    for (Operation &op : originalBlock->getOperations()) {
      // We ignore CPUModuleOp's so that we can hoist before calling this pass
      // if desired; and we mustn't move our new DeviceModuleOp either.
      if (!isa<tt::CPUModuleOp>(op) && !isa<tt::DeviceModuleOp>(op)) {
        opsToMove.push_back(&op);
      }
    }

    // Now move the collected ops.
    for (Operation *op : opsToMove) {
      op->moveBefore(nestedBlock, nestedBlock->begin());
    }
  }
};
} // namespace mlir::tt

namespace mlir::tt {
#define GEN_PASS_DEF_TTUNWRAPDEVICEMODULEPASS
#include "ttmlir/Dialect/TT/Transforms/Passes.h.inc"
class TTUnwrapDeviceModulePass
    : public impl::TTUnwrapDeviceModulePassBase<TTUnwrapDeviceModulePass> {
public:
  using impl::TTUnwrapDeviceModulePassBase<
      TTUnwrapDeviceModulePass>::TTUnwrapDeviceModulePassBase;
  void runOnOperation() override {
    ModuleOp rootModule = getOperation();

    // ensure we only lower top-level module, not any nested modules
    if (rootModule->getParentOp() != nullptr) {
      return;
    }
    // Find DeviceModuleOp, and erase CPUModuleOp if present.
    tt::DeviceModuleOp deviceModuleOp;
    for (Operation &op : rootModule.getBodyRegion().front()) {
      if (auto maybeDeviceModuleOp = dyn_cast<tt::DeviceModuleOp>(op)) {
        deviceModuleOp = maybeDeviceModuleOp;
      } else if (isa<tt::CPUModuleOp>(op)) {
        op.erase();
      }
    }

    // If we don't have a deviceModuleOp inside the top-level ModuleOp, this
    // Pass isn't meaningful.
    if (!deviceModuleOp) {
      return;
    }

    auto nestedModule = llvm::dyn_cast_or_null<ModuleOp>(
        deviceModuleOp.getBodyRegion().front().front());
    assert(
        nestedModule &&
        "device_module did not contain ModuleOp, which isn't a legal state!");

    // Move operations from nested ModuleOp to top-level ModuleOp
    Block *topLevelBlock = &rootModule.getBodyRegion().front();
    Block *nestedBlock = &nestedModule.getBodyRegion().front();

    // Insert point should be before the DeviceModuleOp
    topLevelBlock->getOperations().splice(deviceModuleOp->getIterator(),
                                          nestedBlock->getOperations());

    // Erase the emptied DeviceModuleOp.
    deviceModuleOp->erase();
  }
};
} // namespace mlir::tt
