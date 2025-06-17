// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

namespace mlir::tt::ttnn {
void registerTTNNToFlatbuffer();
void registerTracedTTNNGraphToMLIR();
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttmetal {
void registerTTMetalToFlatbuffer();
} // namespace mlir::tt::ttmetal

namespace mlir::tt::llvm_to_cpu {
void registerLLVMToDynamicLibrary();
} // namespace mlir::tt::llvm_to_cpu

namespace mlir::tt::ttkernel {
void registerTTKernelToCpp();
} // namespace mlir::tt::ttkernel

// Place to register all the custom translations
static void registerCustomTranslations() {
  static bool initOnce = []() {
    mlir::tt::ttnn::registerTTNNToFlatbuffer();
    mlir::tt::ttnn::registerTracedTTNNGraphToMLIR();
    mlir::tt::ttmetal::registerTTMetalToFlatbuffer();
    mlir::tt::llvm_to_cpu::registerLLVMToDynamicLibrary();
    mlir::tt::ttkernel::registerTTKernelToCpp();
    return true;
  }();
  (void)initOnce;
}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerCustomTranslations();

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
