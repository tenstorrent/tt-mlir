// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTDIALECT_TD
#define TTMLIR_TTMLIR_TTDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TT dialect definition.
//===----------------------------------------------------------------------===//

def TT_Dialect : Dialect {
    let name = "tt";
    let summary = "TT types and operations common to all TT dialects.";
    let description = [{
        This dialect defines types and operations common to all TT dialects.
    }];
    let cppNamespace = "::mlir::tt";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::ml_program::MLProgramDialect",
      "::mlir::tensor::TensorDialect",
      "::mlir::linalg::LinalgDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base TT operation definition.
//===----------------------------------------------------------------------===//

class TT_Op<string mnemonic, list<Trait> traits = []> :
        Op<TT_Dialect, mnemonic, traits>;

#endif
