// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalBase.td"

//===----------------------------------------------------------------------===//
// TTNN type definitions
//===----------------------------------------------------------------------===//

class TTMetal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTMetal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTMetal_GlobalSemaphore : TTMetal_Type<"GlobalSemaphore", "global_semaphore"> {
  let summary = "TTMetal global semaphore";
  let description = "Global semaphore type in TTMetal dialect";
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD
