// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tt::d2m::utils {

LogicalResult checkForIllegalSemaphoreOps(Block *block) {
  for (Operation &op : block->getOperations()) {
    if (isa<SemaphoreIncOp>(&op)) {
      return op.emitError()
             << "semaphore_inc is not supported in regions that will be "
                "replicated across multiple threads, as all threads would "
                "increment the semaphore, creating a race condition on the "
                "shared semaphore.";
    }
    if (isa<SemaphoreSetOp>(&op)) {
      return op.emitError()
             << "semaphore_set is not supported in regions that will be "
                "replicated across multiple threads, as all threads would "
                "set the semaphore, creating a race condition on the "
                "shared semaphore.";
    }
    if (auto waitOp = dyn_cast<SemaphoreWaitOp>(&op)) {
      if (waitOp.getResetValue()) {
        return waitOp.emitError()
               << "semaphore_wait with reset is not supported in regions that "
                  "will be replicated across multiple threads, as all threads "
                  "would execute the reset, creating a race condition on the "
                  "shared semaphore.";
      }
    }
    for (Region &region : op.getRegions()) {
      if (!region.empty()) {
        if (failed(checkForIllegalSemaphoreOps(&region.front()))) {
          return failure();
        }
      }
    }
  }
  return success();
}

static LogicalResult checkBackendDmCoreSupport(Operation *rootOp,
                                               ModuleOp moduleOp,
                                               llvm::StringRef backend) {
  auto systemDesc = moduleOp->getAttrOfType<ttcore::SystemDescAttr>(
      ttcore::SystemDescAttr::name);
  if (systemDesc && systemDesc.getChipDescs().front().getArch().getValue() ==
                        ttcore::Arch::Quasar) {
    moduleOp.emitError() << backend << " lowering does not support Quasar";
    return failure();
  }

  auto checkThread = [&](Operation *op, ThreadAttr thread) {
    if (thread.getThreadType() == ThreadType::Datamovement &&
        thread.getDmCoreIndex() >= 2) {
      op->emitError() << "DM core indices greater than 1 are not supported by "
                      << backend << " lowering yet";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  WalkResult unsupportedDmCore = rootOp->walk([&](Operation *nestedOp) {
    if (auto generic = dyn_cast<GenericOp>(nestedOp)) {
      for (Attribute threadAttr : generic.getThreads()) {
        if (checkThread(nestedOp, cast<ThreadAttr>(threadAttr))
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }

    if (auto func = dyn_cast<func::FuncOp>(nestedOp)) {
      auto threadAttr = func->getAttrOfType<ThreadAttr>(ThreadAttr::name);
      if (threadAttr) {
        return checkThread(nestedOp, threadAttr);
      }
    }
    return WalkResult::advance();
  });

  return unsupportedDmCore.wasInterrupted() ? failure() : success();
}

LogicalResult checkBackendDmCoreSupport(ModuleOp moduleOp,
                                        llvm::StringRef backend) {
  return checkBackendDmCoreSupport(moduleOp.getOperation(), moduleOp, backend);
}

LogicalResult checkBackendDmCoreSupport(func::FuncOp funcOp,
                                        llvm::StringRef backend) {
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    funcOp.emitError() << backend << " lowering expected function nested in a "
                       << "module";
    return failure();
  }
  return checkBackendDmCoreSupport(funcOp.getOperation(), moduleOp, backend);
}

} // namespace mlir::tt::d2m::utils
