// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Target/Python/PythonEmitter.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir::tt;

namespace mlir::tt::emitpy {

void registerToPythonTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-python", "translate from mlir to python",
      [](Operation *op, raw_ostream &output) {
        return emitpy::translateToPython(op, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitpy::EmitPyDialect,
                        func::FuncDialect>();
        // clang-format on
      });
}

} // namespace mlir::tt::emitpy
