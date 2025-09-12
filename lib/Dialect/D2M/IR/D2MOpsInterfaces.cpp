// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::tt::d2m;

mlir::LogicalResult
mlir::tt::d2m::detail::verifyGenericParent(mlir::Operation *op) {
  return (op->getParentOfType<mlir::tt::d2m::GenericOp>() ||
          op->getParentOfType<func::FuncOp>())
             ? success()
             : op->emitOpError(
                   "D2M Generic Ops must be inside a generic region");
}
