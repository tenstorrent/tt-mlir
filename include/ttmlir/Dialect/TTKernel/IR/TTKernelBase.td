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
      "::mlir::memref::MemRefDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];
}

//===----------------------------------------------------------------------===//
// TTKernel Traits.
//===----------------------------------------------------------------------===//

class TTKernel_Trait<string name, list<Trait> traits = []> : NativeOpTrait<name, traits> {
  let cppNamespace = "::mlir::tt::ttkernel::OpTrait";
}

def TTKernel_FPUOpTrait : TTKernel_Trait<"TTKernelFPUOpTrait", []>;

def TTKernel_SFPUOpTrait : TTKernel_Trait<"TTKernelSFPUOpTrait", []>;

def TTKernel_InitOpTrait : TTKernel_Trait<"TTKernelInitOpTrait", []>;

//===----------------------------------------------------------------------===//
// Base TTKernel operation definitions.
//===----------------------------------------------------------------------===//

class TTKernel_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTKernel_Dialect, mnemonic, traits>;

class TTKernel_FPUOp<string mnemonic, list<Trait> traits = [TTKernel_FPUOpTrait]> :
        TTKernel_Op<mnemonic, traits>;

class TTKernel_SFPUOp<string mnemonic, list<Trait> traits = [TTKernel_SFPUOpTrait]> :
        TTKernel_Op<mnemonic, traits>;

class TTKernel_InitOp<string mnemonic, list<Trait> traits = [TTKernel_InitOpTrait]> :
        TTKernel_Op<mnemonic, traits>;


#endif
