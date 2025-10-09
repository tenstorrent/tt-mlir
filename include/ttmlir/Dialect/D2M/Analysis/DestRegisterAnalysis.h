// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DESTREGISTERANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DESTREGISTERANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {

/// Information about destination register usage for a GenericOp.
struct DstRegisterInfo {
  int dstMaxUsage = 0;
  llvm::DenseMap<Operation *, int> computeOpMap;
};

/// Analysis for determining destination register tile usage in GenericOps.
struct DestRegisterAnalysis {
  DestRegisterAnalysis(Operation *op);

  int getDstMaxUsage(GenericOp genericOp) const {
    auto match = genericOpMap.find(genericOp);
    assert(match != genericOpMap.end() && "Generic op not found.");
    return match->second.dstMaxUsage;
  }

  llvm::DenseMap<GenericOp, DstRegisterInfo> genericOpMap;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DESTREGISTERANALYSIS_H
