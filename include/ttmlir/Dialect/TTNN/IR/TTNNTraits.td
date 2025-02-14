// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD

include "mlir/IR/OpBase.td"

// Trait for ops that have memory config attribute.
def HasMemoryConfigTrait : NativeOpTrait<"HasMemoryConfigTrait">
{
  let cppNamespace = "mlir::tt::ttnn";
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNTRAITS_TD
