// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

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
    return parser.emitError(loc, "Expected tuple type");
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
    return op.emitOpError("Expected exactly one block");
  }

  size_t moduleCount = 0;
  for (Operation &innerOp : body) {
    if (isa<mlir::ModuleOp>(innerOp)) {
      moduleCount++;
    }
  }

  if (moduleCount != 1) {
    return op.emitOpError("Expected exactly one ModuleOp but found ")
           << moduleCount;
  }

  return success();
}

LogicalResult DeviceModuleOp::verify() { return verifyModuleWrapper(*this); }

LogicalResult CPUModuleOp::verify() { return verifyModuleWrapper(*this); }

LogicalResult LoadCachedOp::verify() {
  // Verify that the callee exists
  FlatSymbolRefAttr calleeAttr = this->getCalleeAttr();
  func::FuncOp calleeFunc =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, calleeAttr);
  if (!calleeFunc) {
    return emitOpError() << "'" << calleeAttr.getValue()
                         << "' does not reference a function";
  }

  // Get the function where this op resides
  Operation *op = this->getOperation();
  func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc) {
    return emitOpError() << "must be nested inside a function";
  }
  FunctionType parentFnType = parentFunc.getFunctionType();

  // Verify that all indices in input_indices are valid
  llvm::ArrayRef<int32_t> inputIndices = this->getInputIndices();

  // Verify each input index is valid for the parent function's signature
  for (const int32_t index : inputIndices) {
    if (index < 0 ||
        static_cast<unsigned>(index) >= parentFnType.getNumInputs()) {
      return emitOpError("input index ")
             << index << " out of range; enclosing function has "
             << parentFnType.getNumInputs() << " parameters";
    }
  }

  FunctionType calleeFnType = calleeFunc.getFunctionType();
  // Verify result count.
  if (calleeFnType.getNumResults() != this->getNumResults()) {
    return emitOpError("Incorrect number of results for callee -- expected ")
           << calleeFnType.getNumResults()
           << " but got: " << this->getNumResults();
  }

  // Verify result types.
  for (unsigned i = 0; i < calleeFnType.getNumResults(); ++i) {
    if (this->getResult(i).getType() != calleeFnType.getResult(i)) {
      return emitOpError("Result type mismatch at index ") << i;
    }
  }

  return success();
}

} // namespace mlir::tt
