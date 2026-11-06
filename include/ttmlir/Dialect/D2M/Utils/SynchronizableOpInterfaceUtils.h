// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m::utils {
struct CBUsageInfo {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

/// Trace view aliases and scf.for iter args/results back to the buffer that
/// enters the outermost loop. This canonicalizes post-bufferization aliases
/// before CB producer/consumer analysis.
Value getSynchronizationRoot(Value value);

llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion);
} // namespace mlir::tt::d2m::utils

#endif // TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H
