// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/CBProducerConsumer.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
namespace mlir::tt::ttir {

CBProducerConsumer::CBProducerConsumer(Operation *op) {
  op->walk([&](Operation *op) {
    bool isProducer = mlir::isa<ttir::YieldOp>(op);
    bool isConsumer = mlir::isa<ttir::AwaitOp>(op);
    if (!isConsumer && !isProducer) {
      return;
    }
    auto enumVal = isConsumer ? Consumer : Producer;
    for (auto operand : op->getOperands()) {
      auto [iter, inserted] = threadCBOrientationMap.insert({operand, enumVal});
      if (!inserted && iter->second != enumVal) {
        iter->second = ProducerConsumer;
      }
    }
  });
}

} // namespace mlir::tt::ttir
