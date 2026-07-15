// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseSet.h"

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
          llvm::report_fatal_error(
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

bool isPurelyDerivedOp(Operation *op,
                       DenseMap<Operation *, bool> &purelyDerivedOps) {
  // check if the result is already cached
  if (purelyDerivedOps.contains(op)) {
    return purelyDerivedOps[op];
  }

  // View-like memref ops only derive aliases/metadata. Keep the original op
  // outside the synchronized region for later users and clone it inside the
  // region for local users. Requiring their source allocs to be pure would
  // incorrectly force all downstream users into one synchronized region.
  if (isa<memref::CollapseShapeOp, memref::SubViewOp>(op)) {
    purelyDerivedOps[op] = true;
    return true;
  }

  // if not, compute the result
  bool isPurelyDerived = mlir::isPure(op);
  for (auto operand : op->getOperands()) {
    // Block arguments (e.g., loop induction variables) have no defining op
    // and are considered pure since they don't have side effects.
    Operation *definingOp = operand.getDefiningOp();
    if (definingOp != nullptr &&
        !isPurelyDerivedOp(definingOp, purelyDerivedOps)) {
      isPurelyDerived = false;
      break;
    }
  }

  // cache the result and return it
  purelyDerivedOps[op] = isPurelyDerived;
  return isPurelyDerived;
}

static bool isNestedInSet(Operation *op,
                          const llvm::DenseSet<Operation *> &ops) {
  for (; op; op = op->getParentOp()) {
    if (ops.contains(op)) {
      return true;
    }
  }
  return false;
}

static Operation *
getUserOutsideRange(Operation *op,
                    const llvm::DenseSet<Operation *> &opsInRange) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!isNestedInSet(user, opsInRange)) {
        return user;
      }
    }
  }
  return nullptr;
}

static void setOutsideUseFailureMessage(Operation *op, Operation *user,
                                        std::string *failureMessage) {
  if (!failureMessage) {
    return;
  }
  failureMessage->clear();
  llvm::raw_string_ostream os(*failureMessage);
  os << "found use of op result outside the range of ops being wrapped: "
     << *user << "\nproducer: " << *op;
}

static LogicalResult
verifyNoOutsideUses(Operation *op,
                    const llvm::DenseSet<Operation *> &opsInRange,
                    std::string *failureMessage) {
  if (Operation *user = getUserOutsideRange(op, opsInRange)) {
    setOutsideUseFailureMessage(op, user, failureMessage);
    return failure();
  }
  return success();
}

static bool
isDefinedByErasedOp(Value value,
                    const llvm::DenseSet<Operation *> &opsToEraseSet) {
  Operation *definingOp = value.getDefiningOp();
  return definingOp && isNestedInSet(definingOp, opsToEraseSet);
}

static bool
hasOperandDefinedByErasedOp(Operation *op,
                            const llvm::DenseSet<Operation *> &opsToEraseSet) {
  for (Value operand : op->getOperands()) {
    if (isDefinedByErasedOp(operand, opsToEraseSet)) {
      return true;
    }
  }
  return false;
}

static void populateSynchronizedRegion(OpBuilder &builder, ValueRange blockArgs,
                                       ValueRange consumers,
                                       ValueRange producers,
                                       Block::iterator start,
                                       Block::iterator end) {
  IRMapping mapping;
  for (size_t i = 0; i < consumers.size(); i++) {
    mapping.map(consumers[i], blockArgs[i]);
  }
  for (size_t i = 0; i < producers.size(); i++) {
    mapping.map(producers[i], blockArgs[i + consumers.size()]);
  }

  for (Operation &op : llvm::make_range(start, end)) {
    builder.clone(op, mapping);
  }
}

// Wraps a range of ops in a SynchronizedRegionOp and returns the latter.
// This is to be used to wrap a region of non-synchronized ops into an op that
// generically implements SynchronizableOpInterface (e.g. wrapping
// linalg.generic after it has been lowered into loops so we can use analysis
// relying on SynchronizableOpInterface, such as in SplitUnifiedThread pass to
// identify CBs and insert correct CB ops on producer/consumer side).
//
// PRECONDITION: No op in [start, end) that is not purely derived may
// produce SSA results that are used outside of [start, end).
FailureOr<Operation *> wrapInSynchronizedRegion(
    RewriterBase &rewriter, Block::iterator start, Block::iterator end,
    const SmallVector<Value> &consumers, const SmallVector<Value> &producers,
    std::string *failureMessage) {
  assert(start != end && "Cannot wrap an empty range in SynchronizedRegionOp");

  // Verify no results are used outside the wrapped range.
  llvm::DenseSet<Operation *> opsInRange;
  for (Operation &op : llvm::make_range(start, end)) {
    opsInRange.insert(&op);
  }

  DenseMap<Operation *, bool> purelyDerivedOps;
  llvm::DenseSet<Operation *> opsToEraseSet;
  for (Operation &op : llvm::make_range(start, end)) {
    if (!isPurelyDerivedOp(&op, purelyDerivedOps)) {
      opsToEraseSet.insert(&op);
      if (failed(verifyNoOutsideUses(&op, opsInRange, failureMessage))) {
        return failure();
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_range(start, end)) {
      if (opsToEraseSet.contains(&op) ||
          !isPurelyDerivedOp(&op, purelyDerivedOps)) {
        continue;
      }
      if (!hasOperandDefinedByErasedOp(&op, opsToEraseSet)) {
        continue;
      }
      if (failed(verifyNoOutsideUses(&op, opsInRange, failureMessage))) {
        return failure();
      }
      opsToEraseSet.insert(&op);
      changed = true;
    }
  }

  rewriter.setInsertionPoint(&*start);
  auto newSynchronizedRegionOp = rewriter.create<d2m::SynchronizedRegionOp>(
      start->getLoc(), TypeRange(), consumers, producers,
      [&](mlir::OpBuilder &bbBuilder, mlir::Location, mlir::ValueRange bbArgs) {
        populateSynchronizedRegion(bbBuilder, bbArgs, consumers, producers,
                                   start, end);
      });

  // Erase wrapped ops in reverse program order so nested uses in later
  // compute ops are removed before their view-like producers.
  SmallVector<Operation *> opsToEraseInBlockOrder;
  for (Operation &op : llvm::make_range(start, end)) {
    if (opsToEraseSet.contains(&op)) {
      opsToEraseInBlockOrder.push_back(&op);
    }
  }
  for (Operation *op : llvm::reverse(opsToEraseInBlockOrder)) {
    rewriter.eraseOp(op);
  }

  return newSynchronizedRegionOp.getOperation();
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
    rewriter.clone(op, mapping);
  }
  // remove the synchronized region op
  rewriter.eraseOp(synchronizedOp);
  return success();
}
} // namespace mlir::tt::d2m::utils
