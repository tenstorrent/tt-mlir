// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNN_TRAITS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNN_TRAITS_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTNN traits definition.
//===----------------------------------------------------------------------===//

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

// Trait to mark ops that can execute on host.
def CanExecuteOnHostTrait : NativeOpTrait<"CanExecuteOnHostTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNN_TRAITS_TD
