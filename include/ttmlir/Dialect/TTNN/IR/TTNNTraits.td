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
def HasDTypeTrait: NativeOpTrait<"HasDTypeTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

// Trait for ops that have layout attribute.
def HasLayoutTrait: NativeOpTrait<"HasLayoutTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

def HasOutputTensorSpecTraits : TraitList<[
  HasMemoryConfigTrait,
  HasDTypeTrait,
  HasLayoutTrait
]>;

// Trait for ops that don't support implicit broadcast on operands.
def ExplicateOperandBroadcastsTrait : NativeOpTrait<"ExplicateOperandBroadcastsTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

// bfp8_b is tile type so we need some way to check if op which returns
// it has correct layout information in encoding.
def CheckBFloat8BTrait : NativeOpTrait<"CheckBFloat8BTrait">
{
  let cppNamespace = "mlir::tt::ttnn";
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD
