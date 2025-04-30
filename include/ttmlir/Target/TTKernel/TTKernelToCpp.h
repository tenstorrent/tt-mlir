// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_TTKERNELTOCPP_H
#define TTMLIR_TARGET_TTKERNEL_TTKERNELTOCPP_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttkernel {

// Translates a top level TTKernel func op (already converted to EmitC) to C++
// and writes it to the given stream.
LogicalResult translateKernelFuncToCpp(func::FuncOp entry,
                                       llvm::raw_ostream &os);

// Translates a top level TTKernel func op (already converted to EmitC) to C++
// for the given symbol name and writes it to the given stream.
LogicalResult translateTopLevelKernelToCpp(ModuleOp moduleOp,
                                           llvm::raw_ostream &os,
                                           StringRef symbolName);

// Walks over all top level TTKernel func ops (already converted to EmitC) in
// the given module and writes them to the given stream.
LogicalResult translateTopLevelKernelsToCpp(ModuleOp moduleOp,
                                            llvm::raw_ostream &os);

} // namespace mlir::tt::ttkernel

#endif
