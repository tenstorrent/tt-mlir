// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_DMAUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_DMAUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/Block.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::d2m::utils {

// Recursively check that there are no illegal semaphore ops in a block and its
// nested regions. Semaphore_inc, semaphore_set, and semaphore_wait with reset
// are not supported in regions that will be replicated across multiple threads,
// as all threads would execute the operation, creating a race condition on the
// shared semaphore.
LogicalResult checkForIllegalSemaphoreOps(Block *block);

bool isSupportedDatamovementProcessor(int32_t processorIndex);
ttcore::NocIndex
getNocForSupportedDatamovementProcessor(int32_t processorIndex);
int32_t getDatamovementProcessorForNoc(ttcore::NocIndex nocIndex);

} // namespace mlir::tt::d2m::utils

#endif
