// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tt::emitpy;

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsTypes.cpp.inc"

void EmitPyDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsTypes.cpp.inc"
      >();
}

::mlir::LogicalResult mlir::tt::emitpy::OpaqueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef value) {
  if (value.empty()) {
    return emitError() << "expected non-empty string in !emitpy.opaque type";
  }
  return success();
}

::mlir::LogicalResult mlir::tt::emitpy::ClassType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef name) {
  if (name.empty()) {
    return emitError() << "expected non-empty string in !emitpy.class type";
  }
  return success();
}

void DictType::print(::mlir::AsmPrinter &printer) const {
  auto keyType = getKeyType();
  auto valueType = getValueType();

  if (keyType || valueType) {
    printer << "<";
    auto opaqueType = dyn_cast<OpaqueType>(keyType);
    if (!opaqueType || opaqueType.getValue() != "Any") {
      printer.printType(keyType);
    } else {
      printer << "Any";
    }
    printer << ", ";
    opaqueType = dyn_cast<OpaqueType>(valueType);
    if (!opaqueType || opaqueType.getValue() != "Any") {
      printer.printType(valueType);
    } else {
      printer << "Any";
    }
    printer << ">";
  }
}

Type DictType::parse(::mlir::AsmParser &parser) {
  ::mlir::MLIRContext *context = parser.getContext();
  Type keyType;
  Type valueType;

  if (succeeded(parser.parseOptionalLess())) {
    if (succeeded(parser.parseOptionalKeyword("Any"))) {
      keyType = OpaqueType::get(context, "Any");
    } else {
      Type parsedKeyType;
      if (failed(parser.parseType(parsedKeyType))) {
        parser.emitError(parser.getNameLoc())
            << "expected specified key type or 'Any' keyword";
        return Type();
      }
      keyType = parsedKeyType;
    }

    if (failed(parser.parseComma())) {
      parser.emitError(parser.getNameLoc()) << "expected comma after key type";
      return Type();
    }

    if (succeeded(parser.parseOptionalKeyword("Any"))) {
      valueType = OpaqueType::get(context, "Any");
    } else {
      Type parsedValueType;
      if (failed(parser.parseType(parsedValueType))) {
        parser.emitError(parser.getNameLoc())
            << "expected specified value type or 'Any' keyword";
        return Type();
      }
      valueType = parsedValueType;
    }

    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getNameLoc()) << "expected '>'";
      return Type();
    }
  }

  return DictType::get(context, keyType, valueType);
}
