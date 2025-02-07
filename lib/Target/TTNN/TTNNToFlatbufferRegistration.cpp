// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

using namespace mlir;

namespace mlir::tt::ttnn {

void registerTTNNToFlatbuffer() {
  TranslateFromMLIRRegistration reg(
      "ttnn-to-flatbuffer", "translate ttnn to flatbuffer",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTTNNToFlatbuffer(op, os, {});
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<mlir::tt::TTDialect,
                        mlir::tt::ttnn::TTNNDialect,
                        mlir::tt::ttkernel::TTKernelDialect,
                        mlir::func::FuncDialect,
                        mlir::emitc::EmitCDialect,
                        mlir::LLVM::LLVMDialect
                        >();
        // clang-format on
        registerAllToLLVMIRTranslations(registry);
      });
}

} // namespace mlir::tt::ttnn
