// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTNN type definitions
//===----------------------------------------------------------------------===//

class TTNN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTNN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTNN_Device : TTNN_Type<"Device", "device", []> {
    let summary = "TTNN device";
    let description = "Opaque device type";
}

def TTNN_GlobalSemaphore : TTNN_Type<"GlobalSemaphore", "global_semaphore"> {
  let summary = "TTNN global semaphore";
  let description = "Global semaphore type in TTNN dialect";
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD
