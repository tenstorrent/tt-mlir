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

namespace mlir::tt::emitpy {
void registerToPythonTranslation();
} // namespace mlir::tt::emitpy

namespace mlir::tt::cuda {
void registerCudaToFlatbuffer();
} // namespace mlir::tt::cuda

// Place to register all the custom translations
static void registerCustomTranslations() {
  static bool initOnce = []() {
    mlir::tt::ttnn::registerTTNNToFlatbuffer();
    mlir::tt::ttmetal::registerTTMetalToFlatbuffer();
    mlir::tt::llvm_to_cpu::registerLLVMToDynamicLibrary();
    mlir::tt::ttkernel::registerTTKernelToCpp();
    mlir::tt::emitpy::registerToPythonTranslation();
    mlir::tt::cuda::registerCudaToFlatbuffer();
    return true;
  }();
  (void)initOnce;
}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerCustomTranslations();

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
