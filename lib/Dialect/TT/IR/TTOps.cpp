// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/IR/BuiltinOps.h"

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
  if (parser.parseType(result)) {
    return failure();
  }

  auto tupType = dyn_cast<TupleType>(result);
  if (!tupType) {
    return parser.emitError(loc, "expected tuple type");
  }

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

void CPUModuleOp::build(OpBuilder &builder, OperationState &state) {
  state.addRegion()->emplaceBlock();
}

void DeviceModuleOp::build(OpBuilder &builder, OperationState &state) {
  state.addRegion()->emplaceBlock();
}

// Helper to do verification for both CPUModuleOp and DeviceModuleOp.
template <typename ModuleOpTy>
static LogicalResult verifyModuleWrapper(ModuleOpTy op) {
  Block &body = op.getBodyRegion().front();
  if (!llvm::hasSingleElement(body)) {
    return op.emitOpError("expected exactly one block");
  }

  size_t moduleCount = 0;
  for (Operation &innerOp : body) {
    if (isa<mlir::ModuleOp>(innerOp)) {
      moduleCount++;
    }
  }

  if (moduleCount != 1) {
    return op.emitOpError("expected exactly one ModuleOp but found ")
           << moduleCount;
  }

  return success();
}

LogicalResult DeviceModuleOp::verify() { return verifyModuleWrapper(*this); }

LogicalResult CPUModuleOp::verify() { return verifyModuleWrapper(*this); }

LogicalResult verifyTTLoadCachedOp(TT_LoadCachedOp op) {
  // Verify that the callee exists and has the right type
  SymbolRefAttr calleeAttr = op.getCalleeAttr();
  Operation *callee = SymbolTable::lookupNearestSymbolFrom(op, calleeAttr);
  if (!callee)
    return op.emitOpError() << "'" << calleeAttr.getValue()
                            << "' does not reference a valid function";

  auto fnType = dyn_cast<FunctionType>(
      callee->getAttrOfType<TypeAttr>("function_type").getValue());
  if (!fnType)
    return op.emitOpError("callee does not have function type");

  // Verify input arity
  if (fnType.getNumInputs() != op.getInputs().size())
    return op.emitOpError("incorrect number of arguments for callee");

  // Verify input types
  for (unsigned i = 0; i < fnType.getNumInputs(); ++i) {
    if (op.getInputs()[i].getType() != fnType.getInput(i))
      return op.emitOpError("argument type mismatch at index ") << i;
  }

  // Verify result types by checking that the function returns a tuple
  // and the tuple elements match our result types
  // This assumes you have a TT_TupleOp defined elsewhere

  // Look for a return op with tt.tuple
  bool foundTupleReturn = false;
  func::FuncOp funcOp = cast<func::FuncOp>(callee);
  for (Block &block : funcOp.getBody()) {
    if (auto returnOp =
            dyn_cast_or_null<func::ReturnOp>(block.getTerminator())) {
      if (returnOp.getNumOperands() == 1) {
        if (auto tupleOp = dyn_cast_or_null<TT_TupleOp>(
                returnOp.getOperand(0).getDefiningOp())) {
          foundTupleReturn = true;

          if (tupleOp.getNumResults() != op.getNumResults())
            return op.emitOpError(
                "callee returns tuple with wrong number of elements");

          for (unsigned i = 0; i < op.getNumResults(); ++i) {
            if (op.getResult(i).getType() != tupleOp.getOperand(i).getType())
              return op.emitOpError("result type mismatch at index ") << i;
          }
        }
      }
    }
  }

  if (!foundTupleReturn)
    return op.emitOpError("callee does not return a tuple");

  return success();
}

} // namespace mlir::tt
