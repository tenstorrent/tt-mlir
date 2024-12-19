// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TT.h"

using namespace mlir;

static void printTupleOpType(OpAsmPrinter &p, Operation *, TypeRange,
                             Type result) {
  p.printType(result);
}

static ParseResult parseTupleOpType(OpAsmParser &parser,
                                    SmallVectorImpl<Type> &operands,
                                    Type &result) {
  // Result type must be tuple type.
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(result))
    return failure();

  auto tupType = dyn_cast<TupleType>(result);
  if (!tupType)
    return parser.emitError(loc, "expected tuple type");

  // Assign operand types to tuple types
  llvm::append_range(operands, tupType.getTypes());
  return success();
}

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOps.cpp.inc"

namespace mlir::tt {

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  GetTupleElementOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto operandType = dyn_cast<TupleType>(adaptor.getOperand().getType());
  if (!operandType) {
    return failure();
  }
  if (adaptor.getIndex() >= static_cast<int64_t>(operandType.size())) {
    return emitOptionalError(location, "index ", adaptor.getIndex(),
                             " is out of bounds of operand with size ",
                             operandType.size());
  }

  inferredReturnTypes.push_back(operandType.getType(adaptor.getIndex()));
  return success();
}

LogicalResult inferTupleOp(MLIRContext *context, std::optional<Location>,
                           ValueRange val,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(TupleType::get(context, val.getTypes()));
  return success();
}

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  TupleOp::Adaptor adaptor(operands, attributes, properties, regions);

  inferredReturnTypes.push_back(
      TupleType::get(context, adaptor.getOperands().getTypes()));
  return success();
}

} // namespace mlir::tt
