// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m::utils {

Value getSynchronizationRoot(Value value) {
  llvm::DenseSet<Value> visited;
  while (value && visited.insert(value).second) {
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
      Block *block = blockArg.getOwner();
      auto forOp = mlir::dyn_cast_or_null<scf::ForOp>(block->getParentOp());
      unsigned argumentNumber = blockArg.getArgNumber();
      if (forOp && block == forOp.getBody() && argumentNumber > 0 &&
          argumentNumber <= forOp.getInitArgs().size()) {
        value = forOp.getInitArgs()[argumentNumber - 1];
        continue;
      }
      return value;
    }

    auto result = mlir::dyn_cast<OpResult>(value);
    Operation *definingOp = result ? result.getOwner() : nullptr;
    if (!definingOp) {
      return value;
    }
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(definingOp)) {
      value = forOp.getInitArgs()[result.getResultNumber()];
      continue;
    }
    if (auto viewOp = mlir::dyn_cast<ViewLikeOpInterface>(definingOp)) {
      value = viewOp.getViewSource();
      continue;
    }
    if (mlir::isa<memref::CastOp, memref::CollapseShapeOp,
                  memref::ExpandShapeOp>(definingOp)) {
      value = definingOp->getOperand(0);
      continue;
    }
    return value;
  }
  return value;
}

// Note: This function must be used post-bufferization but before converting
// to explicit CB form.
llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &genericRegion) {
  llvm::DenseMap<Value, CBUsageInfo> cbUsageInfo;
  auto appendUnique = [](SmallVectorImpl<Operation *> &ops, Operation *op) {
    if (!llvm::is_contained(ops, op)) {
      ops.push_back(op);
    }
  };
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
          appendUnique(
              cbUsageInfo[getSynchronizationRoot(operand.get())].producers, op);
        } else if (synchronizedOp.isConsumer(operand)) {
          appendUnique(
              cbUsageInfo[getSynchronizationRoot(operand.get())].consumers, op);
        }
      }
    }
    return WalkResult::advance();
  });

  return cbUsageInfo;
}

} // namespace mlir::tt::d2m::utils
