// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_LLVM_LLVMTODYNAMICLIB_H
#define TTMLIR_TARGET_LLVM_LLVMTODYNAMICLIB_H

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
// Convert an LLVM operation to a dylib
LogicalResult translateLLVMToLib(Operation *op, llvm::raw_ostream &os,
                                 bool dynamicLib);

// Compile an LLVM operation to a RISC-V object, link it with the prebuilt X280
// firmware harness (x280_harness.o + fw.ld), and write the resulting flat
// binary to `os`. The firmware directory is resolved from the
// TTMLIR_X280_FIRMWARE_DIR environment variable.
LogicalResult translateLLVMToFirmware(Operation *op, llvm::raw_ostream &os);
} // namespace mlir::tt::llvm_to_cpu

#endif
