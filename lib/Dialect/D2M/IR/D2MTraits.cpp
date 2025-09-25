// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m {
namespace impl {

// Helper function to verify that an operation is in the correct thread region
// type
static mlir::LogicalResult
verifyGenericRegionOpThreadType(mlir::Operation *op, ThreadType threadType) {
  mlir::Region *region =
      ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  if (!region) {
    // If not enclosed in a generic op then we forgo verification.
    return mlir::success();
  }
  GenericOp genericOp = mlir::cast<GenericOp>(region->getParentOp());
  if (genericOp.getRegionThreadType(region->getRegionNumber()) != threadType) {
    return op->emitOpError("expected to be in a ")
           << stringifyEnum(threadType) << " region";
  }
  return mlir::success();
}

mlir::LogicalResult verifyGenericRegionComputeOp(mlir::Operation *op) {
  return verifyGenericRegionOpThreadType(op, ThreadType::Compute);
}

mlir::LogicalResult verifyGenericRegionDatamovementOp(mlir::Operation *op) {
  return verifyGenericRegionOpThreadType(op, ThreadType::Datamovement);
}

} // namespace impl

} // namespace mlir::tt::d2m
