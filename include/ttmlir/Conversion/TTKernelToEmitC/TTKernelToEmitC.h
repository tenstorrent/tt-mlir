// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H
#define TTMLIR_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include <llvm/ADT/SmallVector.h>

namespace mlir::tt {
#define GEN_PASS_DECL_CONVERTTTKERNELTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

// Runs a conversion pass to EmitC dialect on a func op containing given
// region's body. Also, it adds boilerplate code such as includes and namespace
// declarations.
LogicalResult convertTTKernelRegionToEmitC(
    OpBuilder &builder, Region *region,
    const ttkernel::KernelConfigInterface &kernelConfig);

// Converts given region to EmitC dialect and translates it to C++ code.
LogicalResult
emitOpRegionAsCpp(Region *region, std::string &regionCpp,
                          const ttkernel::ThreadType &threadType);

LogicalResult
emitOpRegionAsCpp(Region *region, llvm::raw_ostream &os,
                          const ttkernel::ThreadType &threadType);

// Converts dispatch op's regions to C++ code.
LogicalResult
emitDispatchOpRegionsAsCpp(ttmetal::DispatchOp dispatchOp,
                           llvm::SmallVector<std::string> &cppStrings);


LogicalResult
emitKernelAsCpp( mlir::ModuleOp op, llvm::raw_ostream &os, const ttkernel::ThreadType &threadType);

} // namespace mlir::tt

#endif
