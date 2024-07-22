// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

#include "ttmlir/RegisterAll.h"

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TTMLIR", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            mlir::tt::registerAllDialects(*registry);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TTMLIRPasses", LLVM_VERSION_STRING,
          []() { mlir::tt::registerAllPasses(); }};
}
