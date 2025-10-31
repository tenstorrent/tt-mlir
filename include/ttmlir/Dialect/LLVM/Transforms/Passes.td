// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_LLVM_LLVMPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_LLVM_LLVMPASSES_TD

include "mlir/Pass/PassBase.td"

def LLVMEmitCallingConventionWrapperFuncs: Pass<"emit-calling-convention-wrappers", "::mlir::ModuleOp">
{
  let summary = "Emit unpacking functions to unwrap tensors into expected call signatures for our funcs lowered to LLVMDialect";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def NonContiguousMemrefCopyToLinalgPass : Pass<"::mlir::ModuleOp"> {
  let summary = "Lower memref.copy which operates on non-contiguous (strided, non-identity) tensors to linalg.generic";
}

#endif
