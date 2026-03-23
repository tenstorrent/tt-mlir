// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNN_TRAITS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNN_TRAITS_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTNN traits definition.
//===----------------------------------------------------------------------===//

// For ops that return tile-typed tensors, verify that the element type
// on the tensor matches the element type in the TTNNLayoutAttr encoding.
def CheckTileTypeTrait : NativeOpTrait<"CheckTileTypeTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

// Trait to mark ops that can execute on host.
def CanExecuteOnHostTrait : NativeOpTrait<"CanExecuteOnHostTrait">
{
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNN_TRAITS_TD
