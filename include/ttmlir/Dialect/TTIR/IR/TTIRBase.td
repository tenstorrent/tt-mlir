// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// TTIR dialect definition.
//===----------------------------------------------------------------------===//

def TTIR_Dialect : Dialect {
    let name = "ttir";
    let summary = "TTIR dialect provides high level semantics for dispatching work to TT HW.";
    let description = [{
        This dialect provides high level semantics for dispatching work to TT HW.
        It defines a set of declarative/high level operations that are used to
        describe the dispatch, but is largely agnostic to the set of operations
        or dialects that are actually supported by a consuming backend.
    }];
    let cppNamespace = "::mlir::tt::ttir";
    let useDefaultAttributePrinterParser = 1;
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::tensor::TensorDialect",
      "::mlir::linalg::LinalgDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];

    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base TTIR operation definition.
//===----------------------------------------------------------------------===//

class TTIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTIR_Dialect, mnemonic, traits>;

#endif
