// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "llvm/Support/ErrorHandling.h"

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

bool isSupportedDatamovementProcessor(int32_t processorIndex) {
  return processorIndex == 0 || processorIndex == 1;
}

ttcore::NocIndex
getNocForSupportedDatamovementProcessor(int32_t processorIndex) {
  TT_assertv(isSupportedDatamovementProcessor(processorIndex),
             "unsupported datamovement processor index");
  // Preserve the existing backend mapping: NoC0 uses RiscV1 and NoC1 uses
  // RiscV0.
  return processorIndex == 1 ? ttcore::NocIndex::Noc0 : ttcore::NocIndex::Noc1;
}

int32_t getDatamovementProcessorForNoc(ttcore::NocIndex nocIndex) {
  switch (nocIndex) {
  case ttcore::NocIndex::Noc0:
    return 1;
  case ttcore::NocIndex::Noc1:
    return 0;
  }
  llvm_unreachable("unsupported NoC index");
}

} // namespace mlir::tt::d2m::utils
