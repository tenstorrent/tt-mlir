// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

namespace mlir::tt::ttnn {
void registerTTNNToFlatbuffer();
} // namespace mlir::tt::ttnn

// Place to register all the custom translations
static void registerCustomTranslations() {
  static bool initOnce = []() {
    mlir::tt::ttnn::registerTTNNToFlatbuffer();
    return true;
  }();
  (void)initOnce;
}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerCustomTranslations();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
