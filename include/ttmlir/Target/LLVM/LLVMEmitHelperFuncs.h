// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_LLVM_LLVMEMITHELPERFUNCS_H
#define TTMLIR_TARGET_LLVM_LLVMEMITHELPERFUNCS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::llvm_to_cpu {

// Verify that all ops in given module are in LLVM Dialect
LogicalResult verifyAllLLVM(mlir::ModuleOp module);

// Convert a linalg operation to a dylib
// This function signature is required in order to register the conversion in
// mlir translation framework even though we don't use last 2 args
LogicalResult translateLLVMToDyLib(
    Operation *op, llvm::raw_ostream &os,
    std::unordered_map<std::string, GoldenTensor> goldenMap = {});
} // namespace mlir::tt::llvm_to_cpu

#endif
