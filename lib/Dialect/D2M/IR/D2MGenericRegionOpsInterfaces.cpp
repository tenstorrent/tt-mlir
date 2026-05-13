// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::tt::d2m;

LogicalResult mlir::tt::d2m::detail::verifyGenericParent(Operation *op) {
  return (op->getParentOfType<GenericOp>() ||
          op->getParentOfType<func::FuncOp>())
             ? success()
             : op->emitOpError(
                   "D2M Generic Ops must be inside a generic region");
}

// Include generated interface method definitions
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.cpp.inc"
