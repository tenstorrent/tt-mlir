// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_DMAUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_DMAUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::d2m::utils {

// Recursively check that there are no illegal semaphore ops in a block and its
// nested regions. Semaphore_inc, semaphore_set, and semaphore_wait with reset
// are not supported in regions that will be replicated across multiple threads,
// as all threads would execute the operation, creating a race condition on the
// shared semaphore.
LogicalResult checkForIllegalSemaphoreOps(Block *block);

// Backends currently support only the WH/BH 2-DM core model. Reject Quasar and
// any explicit DM core index beyond RiscV0/RiscV1 up front.
LogicalResult checkBackendDmCoreSupport(ModuleOp moduleOp,
                                        llvm::StringRef backend);
LogicalResult checkBackendDmCoreSupport(func::FuncOp funcOp,
                                        llvm::StringRef backend);

} // namespace mlir::tt::d2m::utils

#endif
