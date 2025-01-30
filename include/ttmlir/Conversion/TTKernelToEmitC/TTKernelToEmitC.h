// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H
#define TTMLIR_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::tt {
#define GEN_PASS_DECL_CONVERTTTKERNELTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// IR -> C++ text codegen:
//===----------------------------------------------------------------------===//

// Converts given region to EmitC dialect and translates it to C++ code.
LogicalResult emitOpRegionAsCpp(Region *region, std::string &regionCpp,
                                const ttkernel::ThreadType &threadType);

// Converts given region to EmitC dialect and writes it as C++ code to 'os'.
LogicalResult emitOpRegionAsCpp(Region *region, llvm::raw_ostream &os,
                                const ttkernel::ThreadType &threadType);

// Converts enqueue program op's regions to EmitC dialect and writes
// them as C++ code to 'cppStrings' (in the same order as
// 'enqueueProgramOp' regions).
LogicalResult
emitEnqueueProgramOpRegionsAsCpp(ttmetal::EnqueueProgramOp enqueueProgramOp,
                                 llvm::SmallVector<std::string> &cppStrings);

// Converts all FuncOps in 'op' as if by emitOpRegionAsCpp().
LogicalResult emitKernelAsCpp(mlir::ModuleOp op, llvm::raw_ostream &os,
                              const ttkernel::ThreadType &threadType);

} // namespace mlir::tt

#endif
