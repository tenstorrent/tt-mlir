// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/TT/TTPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::tt::TTDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::ml_program::MLProgramDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                  mlir::tosa::TosaDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttmlir optimizer driver\n", registry));
}
