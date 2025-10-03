// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSTYPES_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"

class D2M_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<D2M_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def D2M_MemTx : D2M_Type<"MemTx", "mem_tx"> {
    let summary = "D2M memory transaction type.";
    let description = "Memory transaction returned by dma op, used to wait for completion.";
}

def D2M_Semaphore : D2M_Type<"Semaphore", "semaphore"> {
    let summary = "D2M semaphore type.";
    let description = "Semaphore primitive type used with semaphore ops to synchronize cores.";
}

#endif  // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSTYPES_TD
