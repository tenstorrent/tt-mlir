// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/RegisterAll.h"

#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
