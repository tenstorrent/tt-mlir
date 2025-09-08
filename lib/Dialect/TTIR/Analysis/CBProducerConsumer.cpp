// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/CBProducerConsumer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
namespace mlir::tt::ttir {

static void findThreadCBMappings(
    func::FuncOp funcOp,
    llvm::DenseMap<std::pair<mlir::StringRef, Value>, ttir::ThreadCBMapping>
        &threadCBMappingMap) {
  for (auto arg : funcOp.getArguments()) {
    if (auto memref = mlir::dyn_cast<MemRefType>(arg.getType())) {
      bool isProducer = false;
      bool isConsumer = false;
      for (auto &use : arg.getUses()) {
        if (auto awaitOp = mlir::dyn_cast<ttir::YieldOp>(use.getOwner())) {
          isProducer = true;
        }
        if (auto yieldOp = mlir::dyn_cast<ttir::AwaitOp>(use.getOwner())) {
          isConsumer = true;
        }
      }

      if (isProducer && isConsumer) {
        threadCBMappingMap.emplace_or_assign(
            std::make_pair(funcOp.getSymNameAttr(), arg),
            ttir::ThreadCBMapping::ProducerConsumer);
      } else if (isProducer) {
        threadCBMappingMap.emplace_or_assign(
            std::make_pair(funcOp.getSymNameAttr(), arg),
            ttir::ThreadCBMapping::Producer);
      } else if (isConsumer) {
        threadCBMappingMap.emplace_or_assign(
            std::make_pair(funcOp.getSymNameAttr(), arg),
            ttir::ThreadCBMapping::Consumer);
      }
    }
  }
}

CBProducerConsumer::CBProducerConsumer(Operation *op) {
  op->walk([&](func::FuncOp funcOp) {
    findThreadCBMappings(funcOp, threadCBMappingMap);
  });
}

} // namespace mlir::tt::ttir
