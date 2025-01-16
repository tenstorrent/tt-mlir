// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include <cassert>
#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h>
#include <mlir/Conversion/FuncToEmitC/FuncToEmitCPass.h>
#include <mlir/Conversion/SCFToEmitC/SCFToEmitC.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/Cpp/CppEmitter.h>

namespace mlir::tt::ttkernel {

static llvm::LogicalResult
translateModuleToCpp(Operation *op, llvm::raw_ostream &os,
                     const ttkernel::ThreadType &threadType) {

  ModuleOp module = dyn_cast<ModuleOp>(op);
  assert(module && "Expected ModuleOp as top level operation");
  return mlir::tt::emitKernelAsCpp(module, os, threadType);
}

LogicalResult translateTTKernelToCpp(Operation *op, llvm::raw_ostream &os,
                                     const ttkernel::ThreadType &threadType) {
  return translateModuleToCpp(op, os, threadType);
}

} // namespace mlir::tt::ttkernel
