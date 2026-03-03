// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include <limits>

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
  if (mlir::isa<ttcore::TileType>(scalarCB.getElementType())) {
    return "Input to TilizeOp or Output to UntilizeOp must have scalar "
           "element type";
  }
  if (!mlir::isa<ttcore::TileType>(tilizedCB.getElementType())) {
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
  if (!mlir::isa<ttcore::TileType>(inputCBType.getElementType())) {
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

  if (!mlir::isa<ttcore::TileType>(inputCBType.getElementType())) {
    return emitOpError("Input to TransposeInitOp must have tile element type");
  }

  if (!mlir::isa<ttcore::TileType>(outputCBType.getElementType())) {
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

  if (!mlir::isa<ttcore::TileType>(inputCBType.getElementType())) {
    return emitOpError(
        "Input to TransposeWHTileOp must have tile element type");
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
::mlir::LogicalResult ResetNocTridBarrierCounterOp::verify() {
  Value noc = getNoc();
  if (noc) {
    auto nocValue = getConstantIntValue(noc);
    constexpr int32_t kNumNocs =
        TTKernelTridNocOpTrait<ResetNocTridBarrierCounterOp>::kNumNocs;
    if (nocValue && (*nocValue < 0 || *nocValue >= kNumNocs)) {
      return emitOpError() << "noc must be in [0, " << (kNumNocs - 1) << "].";
    }
  }
  return success();
}

::mlir::LogicalResult TensorAccessorArgsOp::verify() {
  // Validation rules:
  // 1. If prev_args is present, cta_base and crta_base should NOT be present.
  // 2. If prev_args is NOT present, both cta_base and crta_base MUST be present
  //    and must be constants (unless expr attrs are provided).

  if (getPrevArgs()) {
    // When chaining, we shouldn't have cta_base/crta_base
    if (getCtaBase() || getCrtaBase()) {
      return emitOpError(
          "cta_base and crta_base should not be provided when using prev_args");
    }
  } else {
    // When not chaining, both cta_base and crta_base are required.
    if (!getCtaBase() || !getCrtaBase()) {
      return emitOpError(
          "both cta_base and crta_base are required when prev_args is not "
          "provided");
    }

    // If no expr attribute, the base must be a constant.
    if (!getCtaExprAttr()) {
      if (!getCtaBase().getDefiningOp<arith::ConstantOp>()) {
        return emitOpError(
            "cta_base must be a constant when cta_expr is not provided");
      }
    }

    if (!getCrtaExprAttr()) {
      if (!getCrtaBase().getDefiningOp<arith::ConstantOp>()) {
        return emitOpError(
            "crta_base must be a constant when crta_expr is not provided");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TensorAccessorArgsOp custom assembly format
//===----------------------------------------------------------------------===//
// Format:
// - Without prev_args: TensorAccessorArgs(%cta, %crta) [cta_expr = "..."]
//                      [crta_expr = "..."] {attr-dict}
// - With prev_args:    TensorAccessorArgs(prev = %prev) [cta_expr = "..."]
//                      [crta_expr = "..."] {attr-dict}

void TensorAccessorArgsOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  if (getPrevArgs()) {
    p << "prev = " << getPrevArgs();
  } else {
    p << getCtaBase() << ", " << getCrtaBase();
  }
  p << ")";

  if (getCtaExprAttr()) {
    p << " cta_expr = " << getCtaExprAttr();
  }
  if (getCrtaExprAttr()) {
    p << " crta_expr = " << getCrtaExprAttr();
  }

  llvm::SmallVector<::llvm::StringRef, 3> elidedAttrs = {
      "cta_expr", "crta_expr", "operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

::mlir::ParseResult
TensorAccessorArgsOp::parse(::mlir::OpAsmParser &parser,
                            ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ctaBaseOperand;
  ::mlir::OpAsmParser::UnresolvedOperand crtaBaseOperand;
  ::mlir::OpAsmParser::UnresolvedOperand prevArgsOperand;
  bool hasPrevArgs = false;

  auto i32Type = parser.getBuilder().getI32Type();
  auto tensorAccessorArgsType =
      TensorAccessorArgsType::get(parser.getContext());

  if (parser.parseLParen()) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("prev"))) {
    if (parser.parseEqual() || parser.parseOperand(prevArgsOperand)) {
      return failure();
    }
    hasPrevArgs = true;
  } else {
    // Parse cta_base, crta_base.
    if (parser.parseOperand(ctaBaseOperand) || parser.parseComma() ||
        parser.parseOperand(crtaBaseOperand)) {
      return failure();
    }
  }

  if (parser.parseRParen()) {
    return failure();
  }

  StringAttr ctaExprAttr;
  if (succeeded(parser.parseOptionalKeyword("cta_expr"))) {
    if (parser.parseEqual() || parser.parseAttribute(ctaExprAttr)) {
      return failure();
    }
    result.addAttribute("cta_expr", ctaExprAttr);
  }

  StringAttr crtaExprAttr;
  if (succeeded(parser.parseOptionalKeyword("crta_expr"))) {
    if (parser.parseEqual() || parser.parseAttribute(crtaExprAttr)) {
      return failure();
    }
    result.addAttribute("crta_expr", crtaExprAttr);
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve operands and build operandSegmentSizes
  // Arguments order: cta_base (optional), crta_base (optional), prev_args
  // (optional). When prev_args is present, cta_base and crta_base are not
  // provided.
  int32_t ctaBaseCount = 0;
  int32_t crtaBaseCount = 0;
  int32_t prevArgsCount = 0;

  if (hasPrevArgs) {
    // Only prev_args operand
    if (parser.resolveOperand(prevArgsOperand, tensorAccessorArgsType,
                              result.operands)) {
      return failure();
    }
    prevArgsCount = 1;
  } else {
    // cta_base and crta_base operands
    if (parser.resolveOperand(ctaBaseOperand, i32Type, result.operands) ||
        parser.resolveOperand(crtaBaseOperand, i32Type, result.operands)) {
      return failure();
    }
    ctaBaseCount = 1;
    crtaBaseCount = 1;
  }

  // Add operandSegmentSizes attribute (required by AttrSizedOperandSegments)
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {ctaBaseCount, crtaBaseCount, prevArgsCount}));

  // Add result type
  result.addTypes(tensorAccessorArgsType);

  return success();
}

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

void MyLogicalXOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

void MyLogicalYOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

} // namespace mlir::tt::ttkernel
