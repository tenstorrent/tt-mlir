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
  // Verify that the callee exists and has the right type.
  FlatSymbolRefAttr calleeAttr = this->getCalleeAttr();
  func::FuncOp funcOp =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, calleeAttr);
  if (!funcOp) {
    return emitOpError() << "'" << calleeAttr.getValue()
                         << "' does not reference a function";
  }

  FunctionType fnType = funcOp.getFunctionType();

  // Check if the function has a tuple input, in order to handle EmitC funcs
  // properly.
  bool hasTupleInput = fnType.getNumInputs() == 1 &&
                       mlir::isa<mlir::TupleType>(fnType.getInput(0));

  // Special handling for functions with tuple inputs
  if (hasTupleInput) {
    // For functions with tuple inputs, we allow passing individual tensors
    // that will be packed into a tuple by the TTNNToEmitC pass

    // Get the tuple elements
    auto tupleType = mlir::cast<mlir::TupleType>(fnType.getInput(0));

    // Verify that the number of operands matches the number of tuple elements
    if (tupleType.size() != this->getNumOperands()) {
      return emitOpError("Incorrect number of operands for callee with tuple "
                         "input -- expected ")
             << tupleType.size() << " but got: " << this->getNumOperands();
    }

    // Verify that each operand type matches the corresponding tuple element
    // type
    for (unsigned i = 0; i < tupleType.size(); ++i) {
      if (this->getOperand(i).getType() != tupleType.getType(i)) {
        return emitOpError("Operand type mismatch at index ") << i;
      }
    }
  } else {
    // Standard verification for non-const-eval functions or those without tuple
    // inputs
    if (fnType.getNumInputs() != this->getNumOperands()) {
      return emitOpError("Incorrect number of operands for callee -- expected ")
             << fnType.getNumInputs() << " but got: " << this->getNumOperands();
    }

    for (unsigned i = 0; i < fnType.getNumInputs(); ++i) {
      if (this->getOperand(i).getType() != fnType.getInput(i)) {
        return emitOpError("Operand type mismatch at index ") << i;
      }
    }
  }

  // Check if the function has a tuple result, in order to handle EmitC funcs
  // properly.
  bool hasTupleResult = fnType.getNumResults() == 1 &&
                        mlir::isa<mlir::TupleType>(fnType.getResult(0));

  // Special handling for functions with tuple results
  if (hasTupleResult) {
    // For functions with tuple results, we allow returning individual tensors
    // that will be packed into a tuple by the TTNNToEmitC pass

    // Get the tuple elements
    auto tupleType = mlir::cast<mlir::TupleType>(fnType.getResult(0));

    // Verify that the number of results matches the number of tuple elements
    if (tupleType.size() != this->getNumResults()) {
      return emitOpError("Incorrect number of results for callee with tuple "
                         "result -- expected ")
             << tupleType.size() << " but got: " << this->getNumResults();
    }

    // Verify that each result type matches the corresponding tuple element type
    for (unsigned i = 0; i < tupleType.size(); ++i) {
      if (this->getResult(i).getType() != tupleType.getType(i)) {
        return emitOpError("Result type mismatch at index ") << i;
      }
    }
  } else {
    // Standard verification for functions without tuple results
    // Verify result count.
    if (fnType.getNumResults() != this->getNumResults()) {
      return emitOpError("Incorrect number of results for callee -- expected ")
             << fnType.getNumResults() << " but got: " << this->getNumResults();
    }

    // Verify result types.
    for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
      if (this->getResult(i).getType() != fnType.getResult(i)) {
        return emitOpError("Result type mismatch at index ") << i;
      }
    }
  }

  return success();
}

} // namespace mlir::tt
