// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

// Include generated interface method definitions
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.cpp.inc"

namespace mlir::tt::d2m {

// TODO: change to just be generic op? 
// needs to be a region right now for use by split unified thread pass which creates duplicate regions during split
llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &region) {
    llvm::DenseMap<Value, CBUsageInfo> cbUsageInfo;
    region.walk([&](Operation *op) {
      // go thorugh all ops in generic region and check if they implement D2M_SynchronizableOpInterface
      // go through it's operands
      // if operand is a pipeline producer, get the defining op for this alloc:
      // add this to our list of allocs that need to become CBs, and increment it's producer count
      // if operand is a pipeline consumer, get the alloc
      // add this to our list of allocs that need to become CBs, and increment it's consumer count

      // now go through all these CB allocs and add cb layout attribute, and check exactly one producer and one consumer

      // and assert only one use as producer and one use as consumer
      // add this alloc to list of allocs that need to become a cb
      // and assert only one use as producer and one use as consumer
      // add this alloc to list of allocs that need to become a cb 
      if (SynchronizableOpInterface synchronized_op = dyn_cast<SynchronizableOpInterface>(op)) {
        for (auto& operand : op->getOpOperands()) {
          if (synchronized_op.isProducer(operand) && synchronized_op.isConsumer(operand)) {
            llvm_unreachable("A single op operand cannot be both a producer and consumer");
          }
          else if (synchronized_op.isProducer(operand)) {
            llvm::errs() << "operand x in op y is producer" << "\n";
            cbUsageInfo[operand.get()].producers.push_back(op);
          } 
          else if (synchronized_op.isConsumer(operand)) {
            llvm::errs() << "operand x in op y is consumer" << "\n";
            cbUsageInfo[operand.get()].consumers.push_back(op);
          } 
        }
      }
    
      return WalkResult::advance();
    });

    return cbUsageInfo;
  }

} // namespace mlir::tt::d2m