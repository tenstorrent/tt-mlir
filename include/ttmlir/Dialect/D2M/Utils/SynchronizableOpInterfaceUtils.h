// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m::utils {
struct CBUsageInfo {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion);

/// Wraps a range of ops [start, end) in a SynchronizedRegionOp.
///
/// produce SSA results that are used outside of [start, end). Since
/// SynchronizedRegionOp has no results, any such external uses would become
/// invalid when the original ops are erased.
Operation *wrapInSynchronizedRegion(RewriterBase &rewriter,

/// Unwraps a SynchronizedRegionOp by hoisting its ops to the parent level.
LogicalResult
unwrapSynchronizedRegion(RewriterBase &rewriter,
                         d2m::SynchronizedRegionOp synchronizedOp);

} // namespace mlir::tt::d2m::utils
