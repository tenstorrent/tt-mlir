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
  // TODO(#unified-semaphores): this check is temporarily permissive.
  //
  // semaphore_inc / semaphore_set / semaphore_wait-with-reset in a unified
  // generic are UNSAFE today: the unified region is cloned across the compute
  // and datamovement threads (split-unified-thread-v2), and the datamovement
  // region is further cloned per NOC processor (schedule-dma), so a replicated
  // semaphore_inc/set runs once per thread -- a race on the shared semaphore.
  // The verifier used to reject these outright, which forced CCL kernels into
  // an explicit single-thread `datamovement` form. That form has now been
  // removed (all kernels are unified), so the ops must at least be *accepted*
  // here; this is intentionally unsafe until the robust handling lands.
  //
  // Robust fix (to revisit): pin every semaphore op to a SINGLE datamovement
  // thread (as ScheduleDMA already does for the device_synchronize barrier),
  // and give the other threads a local-barrier handshake against that
  // semaphore thread so the cross-thread ordering is preserved without
  // replicating the semaphore update. Until then, only kernels whose semaphore
  // updates happen to be safe under replication (e.g. device_synchronize, whose
  // barrier ScheduleDMA pins, and remote_store's internal increment, which is
  // CB-tied) are correct.
  (void)block;
  return success();
}

LogicalResult
checkBackendDatamovementProcessorSupport(ModuleOp moduleOp,
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
        thread.getProcessorIndex() >= 2) {
      op->emitError() << "datamovement processor indices greater than 1 are "
                         "not supported by "
                      << backend << " lowering yet";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  WalkResult unsupportedProcessor = moduleOp->walk([&](Operation *op) {
    if (auto generic = dyn_cast<GenericOp>(op)) {
      for (Attribute threadAttr : generic.getThreads()) {
        if (checkThread(op, cast<ThreadAttr>(threadAttr)).wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }

    if (auto func = dyn_cast<func::FuncOp>(op)) {
      auto threadAttr = func->getAttrOfType<ThreadAttr>(ThreadAttr::name);
      if (threadAttr) {
        return checkThread(op, threadAttr);
      }
    }
    return WalkResult::advance();
  });

  return unsupportedProcessor.wasInterrupted() ? failure() : success();
}

} // namespace mlir::tt::d2m::utils
