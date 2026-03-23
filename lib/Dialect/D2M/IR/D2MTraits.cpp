// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m {
namespace impl {

// Helper function to verify that an operation is in a compatible thread region
static mlir::LogicalResult
verifyGenericRegionOpThreadType(mlir::Operation *op,
                                llvm::ArrayRef<ThreadType> allowedThreadTypes) {
  mlir::Region *region =
      ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  if (!region) {
    // If not enclosed in a generic op then we forgo verification.
    return mlir::success();
  }
  GenericOp genericOp = mlir::cast<GenericOp>(region->getParentOp());
  ThreadType regionThreadType =
      genericOp.getRegionThreadType(region->getRegionNumber());
  for (ThreadType allowed : allowedThreadTypes) {
    if (regionThreadType == allowed) {
      return mlir::success();
    }
  }
  std::string expectedTypes;
  llvm::raw_string_ostream os(expectedTypes);
  for (size_t i = 0; i < allowedThreadTypes.size(); ++i) {
    os << stringifyEnum(allowedThreadTypes[i]);
    if (i + 1 < allowedThreadTypes.size()) {
      os << " or ";
    }
  }
  os.flush();
  return op->emitOpError("expected to be in a ") << expectedTypes << " region";
}

mlir::LogicalResult verifyGenericRegionComputeOp(mlir::Operation *op) {
  // Unified threads are treated the same as compute threads
  return verifyGenericRegionOpThreadType(
      op, {ThreadType::Compute, ThreadType::Unified});
}

mlir::LogicalResult verifyGenericRegionDatamovementOp(mlir::Operation *op) {
  return verifyGenericRegionOpThreadType(
      op, {ThreadType::Datamovement, ThreadType::Unified});
}

} // namespace impl

} // namespace mlir::tt::d2m
