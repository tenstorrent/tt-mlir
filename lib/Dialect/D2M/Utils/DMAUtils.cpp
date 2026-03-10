// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

namespace mlir::tt::d2m::utils {

RemoteStoreOp findForwardableStore(RemoteLoadOp remoteLoad) {
  Value localBuffer = remoteLoad.getLocalBuffer();
  if (!localBuffer) {
    return nullptr;
  }

  int64_t userCount = 0;
  int64_t implicitLoadUserCount = 0;
  SmallVector<RemoteStoreOp> implicitStoreUsers;
  Type localBufferType = localBuffer.getType();

  for (Operation *user : localBuffer.getUsers()) {
    ++userCount;

    if (auto loadUser = mlir::dyn_cast<RemoteLoadOp>(user)) {
      if (loadUser.isImplicitForm() &&
          loadUser.getLocalBuffer() == localBuffer) {
        ++implicitLoadUserCount;
      }
      continue;
    }

    auto storeUser = mlir::dyn_cast<RemoteStoreOp>(user);
    if (!storeUser) {
      continue;
    }

    if (!storeUser.isImplicitForm() ||
        storeUser.getLocalBuffer() != localBuffer) {
      continue;
    }

    if (storeUser.getLocalBuffer().getType() != localBufferType) {
      continue;
    }

    implicitStoreUsers.push_back(storeUser);
  }

  bool isForwardablePair = implicitLoadUserCount == 1 &&
                           implicitStoreUsers.size() == 1 && userCount == 2;
  if (!isForwardablePair) {
    return nullptr;
  }
  return implicitStoreUsers.front();
}

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

} // namespace mlir::tt::d2m::utils
