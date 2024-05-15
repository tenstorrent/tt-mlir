// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTIR dialect definition.
//===----------------------------------------------------------------------===//

def TTIR_Dialect : Dialect {
    let name = "ttir";
    let summary = "A TTIR out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::tt::ttir";

    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::ml_program::MLProgramDialect",
      "::mlir::tensor::TensorDialect",
      "::mlir::linalg::LinalgDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base TTIR operation definition.
//===----------------------------------------------------------------------===//

class TTIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTIR_Dialect, mnemonic, traits>;

#endif
