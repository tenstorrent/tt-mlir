// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
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
  if (mlir::isa<ttcore::TileType>(scalarCB.getMemref().getElementType())) {
    return "Input to TilizeOp or Output to UntilizeOp must have scalar "
           "element type";
  }
  if (!mlir::isa<ttcore::TileType>(tilizedCB.getMemref().getElementType())) {
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
  auto inputCBType = getCbIn().getType();
  if (!mlir::isa<ttcore::TileType>(inputCBType.getMemref().getElementType())) {
    return emitOpError("Input to UntilizeInitOp must have tile element type");
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

::mlir::LogicalResult ExperimentalTilizeBlockOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError("ExperimentalTilizeBlockOp must be inside of a "
                       "EnqueueProgramOp region");
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

::mlir::LogicalResult ExperimentalUntilizeBlockOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError("ExperimentalUntilizeBlockOp must be inside of a "
                       "EnqueueProgramOp region");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult TransposeInitOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError(
        "TransposeInitOp must be inside of a EnqueueProgramOp region");
  }

  // Both input and output should have tile element types for transpose.
  auto inputCBType = getCbIn().getType();
  auto outputCBType = getCbOut().getType();

  if (!mlir::isa<ttcore::TileType>(inputCBType.getMemref().getElementType())) {
    return emitOpError("Input to TransposeInitOp must have tile element type");
  }

  if (!mlir::isa<ttcore::TileType>(outputCBType.getMemref().getElementType())) {
    return emitOpError("Output to TransposeInitOp must have tile element type");
  }

  return success();
}

::mlir::LogicalResult TransposeTileOp::verify() {
  if (!insideEnqueueProgramOpRegion(getOperation())) {
    return emitOpError("TransposeWHTileOp must be inside of a "
                       "EnqueueProgramOp region");
  }

  // Only need to check the input CB since this is a single-tile operation
  // The output is implicit (DST register)
  auto inputCBType = getIcb().getType();

  if (!mlir::isa<ttcore::TileType>(inputCBType.getMemref().getElementType())) {
    return emitOpError(
        "Input to TransposeWHTileOp must have tile element type");
  }

  return success();
}

::mlir::LogicalResult CBReinterpretShapeOp::verify() {
  auto inCBType = getInput().getType();
  auto outCBType = getOutput().getType();

  if (inCBType.getMemref().getElementType() !=
      outCBType.getMemref().getElementType()) {
    return emitOpError("input circular buffer element type and output "
                       "circular buffer element type must be the same");
  }

  return success();
}

::mlir::LogicalResult DPrintOp::verify() {
  StringRef fmt = getFmt();
  size_t numFormatSpecifiers = fmt.count("{}");

  if (numFormatSpecifiers != getOperands().size()) {
    return emitOpError("number of format specifiers must match number of "
                       "operands");
  }
  return success();
}

} // namespace mlir::tt::ttkernel
