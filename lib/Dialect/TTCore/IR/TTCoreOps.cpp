// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

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
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.cpp.inc"

namespace mlir::tt::ttcore {

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

// Helper method to verify a list of tensors (inputs or outputs) for
// LoadCachedOp.
static LogicalResult verifyTensorList(LoadCachedOp *op, ValueRange opValues,
                                      TypeRange fnTypes, bool isInput) {
  // Verify count
  if (opValues.size() != fnTypes.size()) {
    return op->emitOpError("Incorrect number of ")
           << (isInput ? "operands" : "results") << " for callee"
           << " -- expected " << fnTypes.size()
           << " but got: " << opValues.size();
  }

  // Verify types
  for (unsigned i = 0; i < fnTypes.size(); ++i) {
    if (opValues[i].getType() != fnTypes[i]) {
      return op->emitOpError() << (isInput ? "Operand" : "Result")
                               << " type mismatch at index " << i;
    }
  }

  return success();
}

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

  // Check if we have tuple inputs/outputs.
  bool hasTupleInput = fnType.getNumInputs() == 1 &&
                       mlir::isa<mlir::TupleType>(fnType.getInput(0));
  bool hasTupleResult = fnType.getNumResults() == 1 &&
                        mlir::isa<mlir::TupleType>(fnType.getResult(0));

  if (LogicalResult result = verifyTensorList(
          this, this->getOperands(),
          hasTupleInput
              ? mlir::cast<mlir::TupleType>(fnType.getInput(0)).getTypes()
              : fnType.getInputs(),
          /*isInput=*/true);
      failed(result)) {
    return result;
  }

  if (LogicalResult result = verifyTensorList(
          this, this->getResults(),
          hasTupleResult
              ? mlir::cast<mlir::TupleType>(fnType.getResult(0)).getTypes()
              : fnType.getResults(),
          false);
      failed(result)) {
    return result;
  }

  return success();
}

} // namespace mlir::tt::ttcore
