// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Target/Python/PythonEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

namespace mlir::tt::emitpy {

// Command-line option for file ID filtering.
static llvm::cl::opt<std::string>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    fileIdOption("emitpy-file-id",
                 llvm::cl::desc("Filter emitpy.file ops by ID. Only files "
                                "with matching ID will be emitted."),
                 llvm::cl::init(""));

void registerToPythonTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-python", "translate from mlir to python",
      [](Operation *op, raw_ostream &output) {
        std::string fileId = fileIdOption.getValue();
        return emitpy::translateToPython(op, output, fileId);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitpy::EmitPyDialect,
                        func::FuncDialect,
                        ttcore::TTCoreDialect>();
        // clang-format on
      });
}

} // namespace mlir::tt::emitpy
