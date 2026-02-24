// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"

//===----------------------------------------------------------------------===//
// TTIR type definitions
//===----------------------------------------------------------------------===//

class TTIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTIR_MemTx : TTIR_Type<"MemTx", "mem_tx"> {
    let summary = "TTIR memory transaction type.";
    let description = "Memory transaction returned by dma op, used to wait for completion.";
}

def TTIR_Semaphore : TTIR_Type<"Semaphore", "semaphore"> {
    let summary = "TTIR semaphore type.";
    let description = "Semaphore primitive type used in combination with semaphore ops to synchronize cores.";
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_TD
