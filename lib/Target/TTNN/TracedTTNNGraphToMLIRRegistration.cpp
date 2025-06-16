// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Target/TTNN/TracedTTNNGraphToMLIR.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <string>

using namespace mlir;

namespace mlir::tt::ttnn {

void registerTracedTTNNGraphToMLIR() {
  TranslateToMLIRRegistration reg(
      "traced-ttnn-to-mlir", "translate graph trace to MLIR",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        // std::cout << "manually loading dialects..." << std::endl;
        context->loadDialect<mlir::tt::TTDialect>();
        context->loadDialect<mlir::tt::ttnn::TTNNDialect>();
        context->loadDialect<mlir::func::FuncDialect>();
        return translateTracedTTNNGraphToMLIR(sourceMgr, context);
      },
      [](DialectRegistry &registry) {
        // std::cout << "registering dialects..." << std::endl;
        // clang-format off
        registry.insert<mlir::tt::TTDialect,
                        mlir::tt::ttnn::TTNNDialect,
                        mlir::func::FuncDialect
                        >();
        // clang-format on
      });
}

} // namespace mlir::tt::ttnn
