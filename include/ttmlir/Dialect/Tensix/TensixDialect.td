// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TENSIX_TENSIXDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TENSIX_TENSIXDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tensix dialect definition.
//===----------------------------------------------------------------------===//

def Tensix_Dialect : Dialect {
    let name = "tensix";
    let summary = "A Tensix out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::tt::tensix";

    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base Tensix operation definition.
//===----------------------------------------------------------------------===//

class Tensix_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tensix_Dialect, mnemonic, traits>;

#endif
