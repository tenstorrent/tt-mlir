// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ttmlir/TTDialect.h"
#include "ttmlir/TTPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::registerPasses();
  // TODO: Register tt passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::tt::TTDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttmlir optimizer driver\n", registry));
}
