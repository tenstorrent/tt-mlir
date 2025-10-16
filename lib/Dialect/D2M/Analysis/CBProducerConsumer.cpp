// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
namespace mlir::tt::d2m {

CBProducerConsumer::CBProducerConsumer(Operation *op) {
  op->walk([&](Operation *op) {
    bool isProducer = mlir::isa<d2m::ReserveOp>(op);
    bool isConsumer = mlir::isa<d2m::PopOp>(op);
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

} // namespace mlir::tt::d2m
