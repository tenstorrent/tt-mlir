// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_DEBUG_DEBUGDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_DEBUG_DEBUGDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Debug dialect definition.
//===----------------------------------------------------------------------===//

def Debug_Dialect : Dialect {
    let name = "debug";
    let summary = "Debug dialect allows inserting debugging information into the IR, that can be used to analyze the IR at different lowering stages or captured by runtime.";
    let description = [{
        The purpose of the debug dialect is to be able to insert debugging information into the IR, that can be used to analyze the IR at different lowering stages or captured by runtime.
        The debug dialect is a compiler API, it is and should be agnostic to the underlying IR it operates on.
    }];
    let cppNamespace = "::mlir::tt::debug";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let dependentDialects = [
      "::mlir::func::FuncDialect",
      "::mlir::tensor::TensorDialect",
    ];

    let hasConstantMaterializer = 1;

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base Debug operation definition.
//===----------------------------------------------------------------------===//

class Debug_Op<string mnemonic, list<Trait> traits = []> :
        Op<Debug_Dialect, mnemonic, traits>;

#endif
