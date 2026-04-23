// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir::tt::d2m {
struct CBUsageInfo {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion);

Operation *wrapInSynchronizedRegion(PatternRewriter &rewriter,
                                    Block::iterator start, Block::iterator end,
                                    const SmallVector<Value> &consumers,
                                    const SmallVector<Value> &producers);

LogicalResult
removeSynchronizedRegions(IRRewriter &rewriter,
                          d2m::SynchronizedRegionOp synchronizedOp);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_SYNCHRONIZABLEOPINTERFACEUTILS_H
