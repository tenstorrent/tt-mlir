// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTNN traits definition.
//===----------------------------------------------------------------------===//

// Trait for ops that have memory config attribute.
def HasMemoryConfigTrait : NativeOpTrait<"HasMemoryConfigTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

// Trait for ops that have output data type attribute.
def HasOutputDTypeTrait: NativeOpTrait<"HasOutputDTypeTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

// Trait for ops that don't support implicit broadcast on operands.
def ExplicateOperandBroadcastsTrait : NativeOpTrait<"ExplicateOperandBroadcastsTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

// Trait for ops that don't support implicit broadcast on operands of type i32.
def ExplicateInt32OperandBroadcastsTrait : NativeOpTrait<"ExplicateInt32OperandBroadcastsTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD
