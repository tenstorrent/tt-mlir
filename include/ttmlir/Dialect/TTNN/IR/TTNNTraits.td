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
  let cppNamespace = "mlir::tt::ttnn";
}

// Trait for ops with variadic operands to specify the number of operands.
//
// Trait for ops with one operand.
def OneOperand : ParamNativeOpTrait<"NOperandTTNN", "1">
{
  let cppNamespace = "mlir::tt::ttnn";
}
// Trait for ops with two operands.
def TwoOperands : ParamNativeOpTrait<"NOperandTTNN", "2">
{
  let cppNamespace = "mlir::tt::ttnn";
}
// Trait for ops with three operands.
def ThreeOperands : ParamNativeOpTrait<"NOperandTTNN", "3">
{
  let cppNamespace = "mlir::tt::ttnn";
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD
