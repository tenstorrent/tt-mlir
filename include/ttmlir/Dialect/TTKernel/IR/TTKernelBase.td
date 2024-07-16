// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTKernel dialect definition.
//===----------------------------------------------------------------------===//

def TTKernel_Dialect : Dialect {
  let name = "ttkernel";
  let summary = "A TTKernel out-of-tree MLIR dialect.";
  let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::tt::ttkernel";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base TTKernel operation definition.
//===---------------------------------------------------------- -- -- -- -- -- -- ==
      = //

      class TTKernel_Op<string mnemonic, list<Trait> traits = []>
      : Op<TTKernel_Dialect, mnemonic, traits>;

#endif
