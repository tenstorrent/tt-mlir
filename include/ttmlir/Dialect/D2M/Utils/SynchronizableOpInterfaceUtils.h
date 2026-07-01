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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tt::d2m::utils {
struct CBUsageInfo {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion);

/// Returns true if `op` is pure and all operations defining its operands are
/// also purely derived. Block arguments are considered pure roots.
bool isPurelyDerivedOp(Operation *op,
                       DenseMap<Operation *, bool> &purelyDerivedOps);

/// Wraps a range of ops [start, end) in a SynchronizedRegionOp.
///
/// PRECONDITION: No op in [start, end) that is not purely derived may
/// produce SSA results that are used outside of [start, end). Since
/// SynchronizedRegionOp has no results, any such external uses would become
/// invalid when the original ops are erased.
FailureOr<Operation *> wrapInSynchronizedRegion(
    RewriterBase &rewriter, Block::iterator start, Block::iterator end,
    const SmallVector<Value> &consumers, const SmallVector<Value> &producers,
    std::string *failureMessage = nullptr);

/// Unwraps a SynchronizedRegionOp by hoisting its ops to the parent level.
LogicalResult
unwrapSynchronizedRegion(RewriterBase &rewriter,
                         d2m::SynchronizedRegionOp synchronizedOp);

} // namespace mlir::tt::d2m::utils

#endif // TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H
