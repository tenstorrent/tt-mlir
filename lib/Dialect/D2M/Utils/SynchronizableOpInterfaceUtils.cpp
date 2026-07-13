// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

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

} // namespace mlir::tt::d2m::utils
