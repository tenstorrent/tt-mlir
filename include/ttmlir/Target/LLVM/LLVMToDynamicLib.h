// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_LINALG_LINALGTODYNAMICLIBRARY_H
#define TTMLIR_TARGET_LINALG_LINALGTODYNAMICLIBRARY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::llvm_to_cpu {
// Convert a linalg operation to a dylib
LogicalResult translateLLVMToDyLib(Operation *op, llvm::raw_ostream &os);
} // namespace mlir::tt::llvm_to_cpu

#endif
