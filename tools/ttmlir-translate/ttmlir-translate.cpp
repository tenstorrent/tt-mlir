// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

// Place to register all the custom translations
static void registerCustomTranslations() {}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerCustomTranslations();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
