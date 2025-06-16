// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
#include <string>

using namespace mlir;

namespace mlir::tt::ttnn {

void registerTracedTTNNGraphToMLIR() {
  static llvm::cl::opt<std::string> tracedGraph("traced-graph",
                                                llvm::cl::desc("todo"));

  TranslateToMLIRRegistration reg(
      "ttnn-to-flatbuffer", "translate graph trace to MLIR",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        if (tracedGraph.empty()) {
          emitError(UnknownLoc::get(context))
              << "Error: traced-graph option must be provided.\n";
          return nullptr;
        }

        // Load the traced graph from the specified file
        std::string fileContent;
        auto bufferOrErr = llvm::MemoryBuffer::getFile(tracedGraph);
        if (std::error_code ec = bufferOrErr.getError()) {
          emitError(UnknownLoc::get(context))
              << "Error reading traced graph file: " << ec.message() << "\n";
          return nullptr;
        }
        // fileContent = bufferOrErr->get()->getBuffer().str();
        // sourceMgr.AddNewSourceBuffer(
        //     llvm::MemoryBuffer::getMemBuffer(fileContent, tracedGraph),
        //     llvm::SMLoc());

        sourceMgr.AddNewSourceBuffer(std::move(bufferOrErr.get()),
                                     llvm::SMLoc());

        // just pass filecontents???

        return translateTracedTTNNGraphToMLIR(sourceMgr, context);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<mlir::tt::TTDialect,
                        mlir::tt::ttnn::TTNNDialect
                        >();
        // clang-format on
      });
}

} // namespace mlir::tt::ttnn
