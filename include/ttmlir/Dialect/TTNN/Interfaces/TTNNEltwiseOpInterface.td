// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNELTWISEOPINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNELTWISEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTNN Elementwise Op Interfaces
//===----------------------------------------------------------------------===//

// Interface to identify elementwise operations in TTNN dialect.
// This provides a unified way to check if an operation is elementwise
// instead of using mlir::isa with multiple op types.

def TTNN_EltwiseOpInterface : OpInterface<"EltwiseOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface for operations that perform elementwise computations.
    This includes unary, binary, and ternary elementwise operations.
  }];
}

def TTNN_EltwiseUnaryOpInterface : OpInterface<"EltwiseUnaryOpInterface", [TTNN_EltwiseOpInterface]> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface for unary elementwise operations.
    These operations take a single input tensor and produce an output tensor
    where each element is computed independently from the corresponding input element.
  }];
}

def TTNN_EltwiseBinaryOpInterface : OpInterface<"EltwiseBinaryOpInterface", [TTNN_EltwiseOpInterface]> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface for binary elementwise operations.
    These operations take two input tensors and produce an output tensor
    where each element is computed independently from the corresponding input elements.
  }];
}

def TTNN_EltwiseTernaryOpInterface : OpInterface<"EltwiseTernaryOpInterface", [TTNN_EltwiseOpInterface]> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface for ternary elementwise operations.
    These operations take three input tensors and produce an output tensor
    where each element is computed independently from the corresponding input elements.
  }];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNELTWISEOPINTERFACE_TD
