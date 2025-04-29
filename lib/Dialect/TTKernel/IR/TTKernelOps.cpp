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

static bool insideEnqueueProgramOpRegion(mlir::Operation *op) {
  mlir::Operation *parentOp = op->getParentOp();

  if (!parentOp) {
    return false;
  }

  if (dyn_cast_if_present<ttmetal::EnqueueProgramOp>(parentOp)) {
    return true;
  }

  if (dyn_cast_if_present<func::FuncOp>(parentOp) &&
      dyn_cast_if_present<mlir::ModuleOp>(parentOp->getParentOp())) {
    return true;
  }
  return insideEnqueueProgramOpRegion(parentOp);
}

::mlir::LogicalResult CBPushBackOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "CBPushBackOp must be inside of a EnqueueProgramOp region");
  }
  return success();
}

::mlir::LogicalResult CBPopFrontOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "CBPopFrontOp must be inside of a EnqueueProgramOp region");
  }
  return success();
}

::mlir::LogicalResult CBReserveBackOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "CBReserveBackOp must be inside of a EnqueueProgramOp region");
  }
  return success();
}

::mlir::LogicalResult CBWaitFrontOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "CBWaitFrontOp must be inside of a EnqueueProgramOp region");
  }
  return success();
}

static std::string verifyTilizeUntilizeCBs(CBType tilizedCB, CBType scalarCB) {
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
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "TilizeInitOp must be inside of a EnqueueProgramOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult UntilizeInitOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "UntilizeInitOp must be inside of a EnqueueProgramOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult TilizeBlockOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "TilizeBlockOp must be inside of a EnqueueProgramOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult UntilizeBlockOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "UntilizeBlockOp must be inside of a EnqueueProgramOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult CBReinterpretShapeOp::verify() {
  auto inCBType = getInput().getType();
  auto outCBType = getOutput().getType();
  if (inCBType.getPort() != outCBType.getPort()) {
    return emitOpError("input circular buffer port and output circular buffer "
                       "port must be the same");
  }

  if (inCBType.getAddress() != outCBType.getAddress()) {
    return emitOpError("input circular buffer address and output circular "
                       "buffer address must be the same");
  }

  if (inCBType.getMemref().getElementType() !=
      outCBType.getMemref().getElementType()) {
    return emitOpError("input circular buffer element type and output "
                       "circular buffer element type must be the same");
  }

  if (inCBType.getNumBuffers() != outCBType.getNumBuffers()) {
    return emitOpError("input circular buffer number of buffers and output "
                       "circular buffer number of buffers must be the same");
  }

  return success();
}

OpFoldResult GetCBOp::fold(FoldAdaptor) { return getCbIndexAttr(); }

} // namespace mlir::tt::ttkernel
