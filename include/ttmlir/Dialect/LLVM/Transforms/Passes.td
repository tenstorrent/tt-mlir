// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_LLVM_LLVMPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_LLVM_LLVMPASSES_TD

include "mlir/Pass/PassBase.td"

def LLVMEmitHelperFuncs: Pass<"emit-llvm-helpers", "::mlir::ModuleOp">
{
  let summary = "Helper function for emitting standardized call-format for all of our functions lowered to LLVMDialect";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif
