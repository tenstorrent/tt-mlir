// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DestRegisterAnalysis.h"

namespace mlir::tt::d2m {

int findDstCapacity(GenericOp genericOp) {
    int dstCapacity = 8;
    return dstCapacity;
    }

DestRegisterAnalysis::DestRegisterAnalysis(Operation *op) {
  // TODO: Implement analysis logic
  op->walk([&](GenericOp genericOp) {
    genericOpMap[genericOp] = findDstCapacity(genericOp);
  });
}

} // namespace mlir::tt::d2m

