// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.td"
include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundInterface.td"

//===----------------------------------------------------------------------===//
// TTNN dialect definition.
//===----------------------------------------------------------------------===//

def TTNN_Dialect : Dialect {
    let name = "ttnn";
    let summary = "A TTNN out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::tt::ttnn";

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
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::ttcore::TTCoreDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base TTNN operation definition.
//===----------------------------------------------------------------------===//

class TTNN_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTNN_Dialect, mnemonic, [Pure, TTNN_WorkaroundInterface] # traits>;

class TTNN_InplaceOp<string mnemonic, list<Trait> traits = []> :
        Op<TTNN_Dialect, mnemonic, [MemoryEffects<[MemWrite]>, TTNN_WorkaroundInterface] # traits>;

class TTNN_MemoryEffectOp<string mnemonic, list<Trait> traits = []> :
        Op<TTNN_Dialect, mnemonic, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>, TTNN_WorkaroundInterface] # traits>;

#endif
