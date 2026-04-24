// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

namespace mlir::tt::d2m::utils {

// Note: This function must be used post-bufferization but before converting
// to explicit CB form.
llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion) {
  llvm::DenseMap<Value, CBUsageInfo> cbUsageInfo;
  genericRegion.walk([&](Operation *op) {
    // To find the CB usage info, we look for ops in the generic that implement
    // D2M_SynchronizableOpInterface and go through their operands to identify
    // which CBs are consumed (i.e. read from) and which are produced (i.e.
    // written to) by the op. We then store this information in a map keyed by
    // the CB Value.
    if (SynchronizableOpInterface synchronizedOp =
            dyn_cast<SynchronizableOpInterface>(op)) {
      for (auto &operand : op->getOpOperands()) {
        if (synchronizedOp.isProducer(operand) &&
            synchronizedOp.isConsumer(operand)) {
          llvm_unreachable(
              "A single op operand cannot be both a producer and consumer");
        } else if (synchronizedOp.isProducer(operand)) {
          cbUsageInfo[operand.get()].producers.push_back(op);
        } else if (synchronizedOp.isConsumer(operand)) {
          cbUsageInfo[operand.get()].consumers.push_back(op);
        }
      }
    }
    return WalkResult::advance();
  });

  return cbUsageInfo;
}

// Wraps a range of ops in a SynchronizedRegionOp and returns the latter.
// This is to be used to wrap a region of non-synchronized ops into an op that
// generically implements SynchronizableOpInterface (e.g. wrapping
// linalg.generic after it has been lowered into loops so we can use analysis
// relying on SynchronizableOpInterface, such as in SplitUnifiedThread pass to
// identify CBs and insert correct CB ops on producer/consumer side).
//
// PRECONDITION: No op in [start, end) may produce SSA results that are used
// outside of [start, end).
Operation *wrapInSynchronizedRegion(RewriterBase &rewriter,
                                    Block::iterator start, Block::iterator end,
                                    const SmallVector<Value> &consumers,
                                    const SmallVector<Value> &producers) {
  assert(start != end && "Cannot wrap an empty range in SynchronizedRegionOp");

  // Verify no results are used outside the wrapped range.
  llvm::DenseSet<Operation *> opsInRange;
  for (Operation &op : llvm::make_range(start, end)) {
    opsInRange.insert(&op);
  }

  for (Operation &op : llvm::make_range(start, end)) {
    for (Value result : op.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!opsInRange.contains(user)) {
          llvm::errs() << "Found use of op result outside the range of ops "
                          "being wrapped: \n"
                       << *user;
          llvm_unreachable(
              "Results of ops being wrapped in SynchronizedRegionOp cannot be "
              "used outside the range.");
        }
      }
    }
  }

  rewriter.setInsertionPoint(&*start);
  auto newSynchronizedRegionOp = rewriter.create<d2m::SynchronizedRegionOp>(
      start->getLoc(), TypeRange(), consumers, producers,
      [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
          mlir::ValueRange bbArgs) {
        IRMapping mapping;

        for (size_t i = 0; i < consumers.size(); i++) {
          mapping.map(consumers[i], bbArgs[i]);
        }
        for (size_t i = 0; i < producers.size(); i++) {
          mapping.map(producers[i], bbArgs[i + consumers.size()]);
        }

        for (Operation &op : llvm::make_range(start, end)) {
          bbBuilder.clone(op, mapping);
        }
      });

  SmallVector<Operation *> opsToErase;
  for (Operation &op : llvm::make_range(start, end)) {
    opsToErase.push_back(&op);
  }
  for (Operation *op : llvm::reverse(opsToErase)) {
    rewriter.eraseOp(op);
  }

  return newSynchronizedRegionOp;
}

// This is used to remove SynchronizedRegionOps and hoist their ops up parent
// level (opposite of wrapInSynchronizedRegion)
LogicalResult unwrapSynchronizedRegion(RewriterBase &rewriter,
                                       SynchronizedRegionOp synchronizedOp) {
  // Remove synchronized region ops, and hoist it's ops up parent level
  // and replace the block args with the synchronized region's operands
  rewriter.setInsertionPoint(synchronizedOp);
  IRMapping mapping;
  for (auto [blockArg, operand] :
       llvm::zip(synchronizedOp.getRegion().front().getArguments(),
                 synchronizedOp->getOperands())) {
    mapping.map(blockArg, operand);
  }

  for (Operation &op : synchronizedOp.getRegion().front().getOperations()) {
    if (!mlir::isa<d2m::YieldOp>(op)) {
      rewriter.clone(op, mapping);
    }
  }
  // remove the synchronized region op
  rewriter.eraseOp(synchronizedOp);
  return success();
}
} // namespace mlir::tt::d2m::utils
