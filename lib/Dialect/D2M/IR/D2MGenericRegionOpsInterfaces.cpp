// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

// Include generated interface method definitions
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.cpp.inc"

namespace mlir::tt::d2m {

// TODO: change to just be generic op?
// needs to be a region right now for use by split unified thread pass which
// creates duplicate regions during split
// Note: must be used before converting to explicit CB form
llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion) {
  llvm::DenseMap<Value, CBUsageInfo> cbUsageInfo;
  genericRegion.walk([&](Operation *op) {
    // go thorugh all ops in generic region and check if they implement
    // D2M_SynchronizableOpInterface go through it's operands if operand is a
    // pipeline producer, get the defining op for this alloc: add this to our
    // list of allocs that need to become CBs, and increment it's producer count
    // if operand is a pipeline consumer, get the alloc
    // add this to our list of allocs that need to become CBs, and increment
    // it's consumer count

    // now go through all these CB allocs and add cb layout attribute, and check
    // exactly one producer and one consumer

    // and assert only one use as producer and one use as consumer
    // add this alloc to list of allocs that need to become a cb
    // and assert only one use as producer and one use as consumer
    // add this alloc to list of allocs that need to become a cb
    if (SynchronizableOpInterface synchronized_op =
            dyn_cast<SynchronizableOpInterface>(op)) {
      for (auto &operand : op->getOpOperands()) {
        if (synchronized_op.isProducer(operand) &&
            synchronized_op.isConsumer(operand)) {
          llvm_unreachable(
              "A single op operand cannot be both a producer and consumer");
        } else if (synchronized_op.isProducer(operand)) {
          // llvm::errs() << "operand x in op y is producer" << "\n";
          cbUsageInfo[operand.get()].producers.push_back(op);
        } else if (synchronized_op.isConsumer(operand)) {
          // llvm::errs() << "operand x in op y is consumer" << "\n";
          cbUsageInfo[operand.get()].consumers.push_back(op);
        }
      }
    }
    return WalkResult::advance();
  });

  // TODO: clean up
  // print cb usage info
  // for (auto &[cb, usageInfo] : cbUsageInfo) {
  //  llvm::errs() << "cb: " << cb << "\n";
  //  for (auto &producer : usageInfo.producers) {
  //    llvm::errs() << "producer: " << producer << "\n";
  //  }
  //  for (auto &consumer : usageInfo.consumers) {
  //    llvm::errs() << "consumer: " << consumer << "\n";
  //  }
  //}

  return cbUsageInfo;
}

LogicalResult insertCBLayoutAttr(RewriterBase &rewriter,
                                 memref::AllocOp allocOp,
                                 llvm::SmallVector<Operation *> &producers) {
  auto buffer = allocOp->getResult(0);
  auto bufferType = mlir::cast<MemRefType>(buffer.getType());
  auto shardShape = bufferType.getShape();
  // llvm::errs() << "\n";
  // for (int64_t shape : shardShape) {
  //   llvm::errs() << shape << " ";
  // }
  // llvm::errs() << "\n";
  Type elementType = bufferType.getElementType();
  // llvm::errs() << "element type: " << elementType << "\n";

  auto cbLayout = ttcore::CBLayoutAttr::get(
      bufferType.getContext(), shardShape,
      ttcore::getElementSizeBytes(elementType),
      2); // TODO: remove hardcoded value, actually change to just add
          // d2m.double_buffered attribute to alloc op

  auto newMemRefType = MemRefType::get(shardShape, elementType, cbLayout,
                                       bufferType.getMemorySpace());
  rewriter.setInsertionPoint(allocOp);
  auto newAllocOp =
      rewriter.replaceOpWithNewOp<memref::AllocOp>(allocOp, newMemRefType);
  Value newBuffer = newAllocOp->getResult(0);

  // check ops that have this buffer as a dps init, we need to change their
  // output type to also have cb layout
  for (auto &producer : producers) {
    // Check if producer has no results, if so, skip
    if (producer->getNumResults() == 0) {
      continue;
    }

    if (auto destinationStyleOp =
            dyn_cast<DestinationStyleOpInterface>(producer)) {
      for (auto &operand : destinationStyleOp->getOpOperands()) {
        if (destinationStyleOp.isDpsInit(&operand) &&
            operand.get() == newBuffer) {
          // update op result that matches dps init
          OpResult tiedResult = destinationStyleOp.getTiedOpResult(&operand);
          tiedResult.setType(newMemRefType);
        }
      }
    }
  }

  // Transfer address and alignment from the old alloc (assigned
  // by the planner in assignAllocAddresses).
  if (auto addrAttr = allocOp->getAttr("address")) {
    newAllocOp->setAttr("address", addrAttr);
  }
  if (auto alignAttr = allocOp.getAlignmentAttr()) {
    newAllocOp.setAlignmentAttr(alignAttr);
  }
  // TODO: can remove this above addr and alignement set once reordering
  // cb layout attr insertion to top of allocator methods
  // marking and allocating is ideally done after reblocking but currently
  // reblocking is before allocating

  return success();
}

LogicalResult markSynchronizedOpBuffers(IRRewriter &rewriter,
                                        d2m::GenericOp genericOp) {
  // print IR at this point
  // llvm::errs() << "IR at this point: " << "\n";
  // genericOp->print(llvm::errs());
  // llvm::errs() << "\n";

  auto cbUsageInfo = getCBUsageInfo(genericOp.getRegion(0));

  // print cb usage info
  // for (auto &[cb, usageInfo] : cbUsageInfo) {
  //  llvm::errs() << "cb: " << cb << "\n";
  //  for (auto &producer : usageInfo.producers) {
  //    llvm::errs() << "producer: " << producer << "\n";
  //  }
  //  for (auto &consumer : usageInfo.consumers) {
  //    llvm::errs() << "consumer: " << consumer << "\n";
  //  }
  //}

  // loop over memrefs in generic, separate out to separate pass or rewriter?
  // go through ops with synchronizable interface being used as input and
  // attach CB layout aatribute on their operands using interface to tell
  // which are Synchronized inputs and which are Synchronized outputs separate
  // out handling for allocs inside generic and outside
  for (auto &[cb, usageInfo] : cbUsageInfo) {
    // TODO: add check on num consumers and producers
    // TODO: this is is an inconsistency, most remote load/store will have to
    // correctly be bufferized to not return a result
    if (failed(insertCBLayoutAttr(
            rewriter, mlir::cast<memref::AllocOp>(cb.getDefiningOp()),
            usageInfo.producers))) {
      return failure();
    }
  }

  return success();
}

// Wraps a SynchronizableOpInterface in a SynchronizedRegionOp and returns the
// new ops. This is to be used if we want to lower a synchronized op to a non
// synchronized op/region before cb op insertion (e.g. linalg to affine). NOTE:
// this pass using this must ensure that the sychronized op gets lowered to non
// synchronized ops before we get to cb op insertion pass.
// TODO: better approach?
std::pair<Operation *, Operation *>
wrapInSynchronizedRegion(PatternRewriter &rewriter,
                         SynchronizableOpInterface synchronizedOp) {
  // print IR at this point
  // llvm::errs() << "IR at this point: " << "\n";
  // genericOp->print(llvm::errs());
  // llvm::errs() << "\n";

  // TODO: check already in a synchronized region
  SmallVector<Value> consumers;
  SmallVector<Value> producers;
  for (auto &operand : synchronizedOp->getOpOperands()) {
    if (synchronizedOp.isConsumer(operand)) {
      consumers.push_back(operand.get());
    }
    if (synchronizedOp.isProducer(operand)) {
      producers.push_back(operand.get());
    }
  }

  Operation *clonedSynchronizedOp;
  rewriter.setInsertionPoint(synchronizedOp);
  auto newSynchronizedRegionOp = rewriter.create<d2m::SynchronizedRegionOp>(
      synchronizedOp->getLoc(), synchronizedOp->getResultTypes(), consumers,
      producers,
      [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
          mlir::ValueRange bbArgs) {
        IRMapping mapping;

        for (size_t i = 0; i < consumers.size(); i++) {
          mapping.map(consumers[i], bbArgs[i]);
        }
        for (size_t i = 0; i < producers.size(); i++) {
          mapping.map(producers[i], bbArgs[i + consumers.size()]);
        }
        clonedSynchronizedOp = bbBuilder.clone(*synchronizedOp, mapping);
        bbBuilder.create<d2m::YieldOp>(bbLoc,
                                       clonedSynchronizedOp->getResults());
      });

  rewriter.replaceOp(synchronizedOp, newSynchronizedRegionOp.getOperation());

  return std::make_pair(clonedSynchronizedOp, newSynchronizedRegionOp);
}

} // namespace mlir::tt::d2m
