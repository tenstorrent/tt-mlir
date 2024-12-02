// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Target/LLVM/LLVMToDynamicLib.h"

using namespace mlir;

namespace mlir::tt::llvm_to_cpu {

void registerLLVMToDynamicLibrary() {
  TranslateFromMLIRRegistration reg(
      "llvm-to-dylib", "translate llvm dialect to dynamic library",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateLLVMToDyLib(op, os, {});
      },
      [](DialectRegistry &registry) {
        // clang-format off
        // registry.insert<mlir::tt::TTDialect,
        //                 mlir::tt::ttnn::TTNNDialect,
        //                 mlir::tt::ttkernel::TTKernelDialect,
        //                 mlir::func::FuncDialect,
        //                 mlir::emitc::EmitCDialect
        //                 >();
        // clang-format on
      });
}

} // namespace mlir::tt::llvm_to_cpu
