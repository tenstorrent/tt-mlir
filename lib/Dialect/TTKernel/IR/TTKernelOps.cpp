// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.cpp.inc"

namespace mlir::tt::ttkernel {

static bool insideDispatchOpRegion(mlir::Operation *op) {
  mlir::Operation *parentOp = op->getParentOp();

  if (!parentOp) {
    return false;
  }

  if (dyn_cast_or_null<ttmetal::DispatchOp>(parentOp)) {
    return true;
  }

  if (dyn_cast_or_null<func::FuncOp>(parentOp) &&
      dyn_cast_or_null<mlir::ModuleOp>(parentOp->getParentOp())) {
    return true;
  }
  return insideDispatchOpRegion(parentOp);
}

::mlir::LogicalResult BuiltinOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("KernelOp must be inside of a DispatchOp region");
  }
  return success();
}

::mlir::LogicalResult CBPushBackOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("CBPushBackOp must be inside of a DispatchOp region");
  }
  return success();
}

::mlir::LogicalResult CBPopFrontOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("CBPopFrontOp must be inside of a DispatchOp region");
  }
  return success();
}

::mlir::LogicalResult CBReserveBackOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("CBReserveBackOp must be inside of a DispatchOp region");
  }
  return success();
}

::mlir::LogicalResult CBWaitFrontOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("CBWaitFrontOp must be inside of a DispatchOp region");
  }
  return success();
}

static std::string verifyTilizeUntilizeCBs(CBType tilizedCB, CBType scalarCB) {
  if (tilizedCB.getPort() == scalarCB.getPort()) {
    return "Input circular buffer port and output circular buffer "
           "port must be different";
  }
  if (mlir::isa<tt::TileType>(scalarCB.getMemref().getElementType())) {
    return "Input to TilizeOp or Output to UntilizeOp must have scalar "
           "element type";
  }
  if (!mlir::isa<tt::TileType>(tilizedCB.getMemref().getElementType())) {
    return "Input to UntilizeOp or Output to TilizeOp must have tile "
           "element type";
  }
  return std::string();
}

::mlir::LogicalResult TilizeInitOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("TilizeInitOp must be inside of a DispatchOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult UntilizeInitOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("UntilizeInitOp must be inside of a DispatchOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult TilizeBlockOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("TilizeBlockOp must be inside of a DispatchOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult UntilizeBlockOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("UntilizeBlockOp must be inside of a DispatchOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult ReturnOp::verify() {
  if (!insideDispatchOpRegion(getOperation())) {
    return emitOpError("ReturnOp must be inside of a DispatchOp region");
  }
  return success();
}

} // namespace mlir::tt::ttkernel
