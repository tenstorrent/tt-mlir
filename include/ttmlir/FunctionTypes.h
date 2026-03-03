// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_FUNCTIONTYPES_H
#define TTMLIR_FUNCTIONTYPES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace ttmlir::utils {

/// Enum representing the different types of functions in the compilation
/// pipeline.
enum class FunctionType {
  /// Forward function for device execution - the main
  /// computational graph intended for device execution, often
  /// referred to as just "forward" function.
  ForwardDevice,

  /// Forward function for CPU execution - a function that has been hoisted
  /// to execute on the CPU instead of the device.
  ForwardCPU,

  /// Forward CPU declaration - a declaration (prototype) for a CPU-hoisted
  /// function in the device module.
  ForwardCPUDeclaration,

  /// Const-eval function - a function that performs constant evaluation
  /// at runtime.
  ConstEval,

  /// Trace main function - a function that represents the main entry point
  /// for tracing.
  TraceMain,

  /// Trace run and capture function - a function that runs and captures
  /// a trace for later execution.
  TraceRunAndCapture,

  /// Trace execute function - a function that executes a previously
  /// captured trace.
  TraceExecute,

  /// Kernel function - a low-level function representing a kernel
  /// operation.
  Kernel,

  /// Input generator function - generates input data for testing/execution.
  /// Relevant for EmitPy and EmitC targets.
  InputGenerator,

  /// Main function - the entry point for the generated program.
  /// Relevant for EmitPy and EmitC targets.
  Main,
};

namespace detail {
/// Attribute name for function type.
constexpr inline llvm::StringLiteral kFunctionTypeAttrName = "tt.function_type";

/// String value constants for function types. Internal use only.
constexpr inline llvm::StringLiteral kForwardDeviceValue = "forward_device";
constexpr inline llvm::StringLiteral kForwardCPUValue = "forward_cpu";
constexpr inline llvm::StringLiteral kForwardCPUDeclarationValue =
    "forward_cpu_declaration";
constexpr inline llvm::StringLiteral kConstEvalValue = "const_eval";
constexpr inline llvm::StringLiteral kTraceMainValue = "trace_main";
constexpr inline llvm::StringLiteral kTraceRunAndCaptureValue =
    "trace_run_and_capture";
constexpr inline llvm::StringLiteral kTraceExecuteValue = "trace_execute";
constexpr inline llvm::StringLiteral kKernelValue = "kernel";
constexpr inline llvm::StringLiteral kInputGeneratorValue = "input_generator";
constexpr inline llvm::StringLiteral kMainValue = "main";
} // namespace detail

/// Returns the string value for the given function type.
inline llvm::StringRef getFunctionTypeValue(FunctionType type) {
  switch (type) {
  case FunctionType::ForwardDevice:
    return detail::kForwardDeviceValue;
  case FunctionType::ForwardCPU:
    return detail::kForwardCPUValue;
  case FunctionType::ForwardCPUDeclaration:
    return detail::kForwardCPUDeclarationValue;
  case FunctionType::ConstEval:
    return detail::kConstEvalValue;
  case FunctionType::TraceMain:
    return detail::kTraceMainValue;
  case FunctionType::TraceRunAndCapture:
    return detail::kTraceRunAndCaptureValue;
  case FunctionType::TraceExecute:
    return detail::kTraceExecuteValue;
  case FunctionType::Kernel:
    return detail::kKernelValue;
  case FunctionType::InputGenerator:
    return detail::kInputGeneratorValue;
  case FunctionType::Main:
    return detail::kMainValue;
  }
  llvm_unreachable("Unknown FunctionType");
}

/// Returns the FunctionType for a given string value, or nullopt if not found.
inline std::optional<FunctionType>
parseFunctionTypeValue(llvm::StringRef value) {
  if (value == detail::kForwardDeviceValue) {
    return FunctionType::ForwardDevice;
  }
  if (value == detail::kForwardCPUValue) {
    return FunctionType::ForwardCPU;
  }
  if (value == detail::kForwardCPUDeclarationValue) {
    return FunctionType::ForwardCPUDeclaration;
  }
  if (value == detail::kConstEvalValue) {
    return FunctionType::ConstEval;
  }
  if (value == detail::kTraceMainValue) {
    return FunctionType::TraceMain;
  }
  if (value == detail::kTraceRunAndCaptureValue) {
    return FunctionType::TraceRunAndCapture;
  }
  if (value == detail::kTraceExecuteValue) {
    return FunctionType::TraceExecute;
  }
  if (value == detail::kKernelValue) {
    return FunctionType::Kernel;
  }
  if (value == detail::kInputGeneratorValue) {
    return FunctionType::InputGenerator;
  }
  if (value == detail::kMainValue) {
    return FunctionType::Main;
  }
  return std::nullopt;
}

/// Sets the function type attribute on the given function.
inline void setFunctionType(mlir::func::FuncOp funcOp, FunctionType type) {
  funcOp->setAttr(
      detail::kFunctionTypeAttrName,
      mlir::StringAttr::get(funcOp->getContext(), getFunctionTypeValue(type)));
}

/// Gets the function type from the given function, or nullopt if not set.
inline std::optional<FunctionType> getFunctionType(mlir::func::FuncOp funcOp) {
  auto attr =
      funcOp->getAttrOfType<mlir::StringAttr>(detail::kFunctionTypeAttrName);
  if (!attr) {
    return std::nullopt;
  }
  return parseFunctionTypeValue(attr.getValue());
}

/// Checks if the function has the given function type attribute.
inline bool hasFunctionType(mlir::func::FuncOp funcOp, FunctionType type) {
  auto currentType = getFunctionType(funcOp);
  return currentType.has_value() && *currentType == type;
}

/// Removes the function type attribute from the given function.
inline void clearFunctionType(mlir::func::FuncOp funcOp) {
  funcOp->removeAttr(detail::kFunctionTypeAttrName);
}

//===----------------------------------------------------------------------===//
// Convenience query functions
//===----------------------------------------------------------------------===//

/// Returns true if the function is marked as a forward device function.
inline bool isForwardDeviceFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::ForwardDevice);
}

/// Returns true if the function is marked as a forward CPU function.
inline bool isForwardCPUFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::ForwardCPU);
}

/// Returns true if the function is marked as a forward CPU declaration.
inline bool isForwardCPUDeclarationFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::ForwardCPUDeclaration);
}

/// Returns true if the function is marked as a const-eval function.
inline bool isConstEvalFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::ConstEval);
}

/// Returns true if the function is marked as a trace main function.
inline bool isTraceMainFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::TraceMain);
}

/// Returns true if the function is marked as a trace run and capture function.
inline bool isTraceRunAndCaptureFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::TraceRunAndCapture);
}

/// Returns true if the function is marked as a trace execute function.
inline bool isTraceExecuteFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::TraceExecute);
}

/// Returns true if the function is any trace function (TraceMain,
/// TraceRunAndCapture, or TraceExecute).
inline bool isTraceFunc(mlir::func::FuncOp funcOp) {
  return isTraceMainFunc(funcOp) || isTraceRunAndCaptureFunc(funcOp) ||
         isTraceExecuteFunc(funcOp);
}

/// Returns true if the function is marked as a kernel function.
inline bool isKernelFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::Kernel);
}

/// Returns true if the function is marked as an input generator function.
inline bool isInputGeneratorFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::InputGenerator);
}

/// Returns true if the function is marked as a main function.
inline bool isMainFunc(mlir::func::FuncOp funcOp) {
  return hasFunctionType(funcOp, FunctionType::Main);
}

//===----------------------------------------------------------------------===//
// Verification functions
//===----------------------------------------------------------------------===//

/// Verifies that all functions in the module have a function type attribute.
/// Emits an error for each function missing the attribute and returns failure
/// if any function is missing the attribute.
inline mlir::LogicalResult verifyFunctionTypes(mlir::ModuleOp moduleOp) {
  mlir::LogicalResult result = mlir::success();
  moduleOp->walk([&](mlir::func::FuncOp func) {
    if (!getFunctionType(func).has_value()) {
      func.emitError("function is missing required 'tt.function_type' "
                     "attribute during flatbuffer translation");
      result = mlir::failure();
    }
  });
  return result;
}

} // namespace ttmlir::utils

#endif // TTMLIR_FUNCTIONTYPES_H
