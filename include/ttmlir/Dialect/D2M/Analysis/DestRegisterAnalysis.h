// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DESTREGISTERANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DESTREGISTERANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

/// Information about destination register usage for a GenericOp.
struct DstRegisterInfo {
  int dstMaxUsage = 0;
  // Store intermediate DST slice indices instead of a map of operations.
  // This avoids keeping stale Operation pointers around.
  llvm::SmallVector<int> dstSliceIndices;
};

/// Analysis for determining destination register tile usage in GenericOps.
struct DestRegisterAnalysis {
  DestRegisterAnalysis(Operation *op);

  llvm::SmallVector<DstRegisterInfo> dstRegisterInfoList;
  // Map from operation pointer to generic op counter.
  llvm::DenseMap<Operation *, int> opToGenericOpCounter;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DESTREGISTERANALYSIS_H
