// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
using namespace mlir;

namespace mlir::tt::ttkernel {

// TODO: Should generalize this to read kernel type from Attribute?
void registerTensixKernelToCpp() {
  TranslateFromMLIRRegistration reg(
      "ttkernel-to-cpp-tensix", "translate tensix kernel to C++",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTTKernelToCpp(op, os, tt::ttkernel::ThreadType::Tensix);
      },
      [](DialectRegistry &registry) {
        registry
            .insert<mlir::scf::SCFDialect, mlir::tt::ttkernel::TTKernelDialect,
                    mlir::arith::ArithDialect, mlir::emitc::EmitCDialect,
                    mlir::func::FuncDialect, mlir::tt::TTDialect,
                    mlir::memref::MemRefDialect>();
      });
}

void registerNocKernelToCpp() {
  TranslateFromMLIRRegistration reg(
      "ttkernel-to-cpp-noc", "translate noc kernel to C++",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTTKernelToCpp(op, os, tt::ttkernel::ThreadType::Noc);
      },
      [](DialectRegistry &registry) {
        registry
            .insert<mlir::scf::SCFDialect, mlir::tt::ttkernel::TTKernelDialect,
                    mlir::arith::ArithDialect, mlir::emitc::EmitCDialect,
                    mlir::func::FuncDialect, mlir::tt::TTDialect,
                    mlir::memref::MemRefDialect>();
      });
}

} // namespace mlir::tt::ttkernel
