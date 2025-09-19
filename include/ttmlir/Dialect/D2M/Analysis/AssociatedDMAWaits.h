// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_ASSOCIATEDDMAWAITS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_ASSOCIATEDDMAWAITS_H

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir::tt::d2m {

struct AssociatedDMAWaits {
  AssociatedDMAWaits(Operation *op);

  SmallVector<DMAWaitOp> get(DMAOpInterface dmaOp) const {
    auto match = dmaWaitsMap.find(dmaOp);
    assert(match != dmaWaitsMap.end() && "Associated DMA wait not found.");
    return match->second;
  }

  llvm::DenseMap<DMAOpInterface, SmallVector<DMAWaitOp>> dmaWaitsMap;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_ASSOCIATEDDMAWAITS_H
