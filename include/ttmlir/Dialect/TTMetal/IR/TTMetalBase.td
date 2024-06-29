// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMetal_TTMetalDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTMetal_TTMetalDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTMetal dialect definition.
//===----------------------------------------------------------------------===//

def TTMetal_Dialect : Dialect {
    let name = "ttmetal";
    let summary = "A TTMetal out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::tt::ttmetal";

    // let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [
      "::mlir::func::FuncDialect",
      "::mlir::tensor::TensorDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base TTMetal operation definition.
//===----------------------------------------------------------------------===//

class TTMetal_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTMetal_Dialect, mnemonic, traits>;

#endif
