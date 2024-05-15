// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Passes.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"
#include "ttmlir/Dialect/Tensix/IR/Tensix.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::ttir::registerPasses();
  mlir::tt::ttmetal::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::tt::TTDialect, mlir::tt::ttir::TTIRDialect,
                  mlir::tt::ttmetal::TTMetalDialect,
                  mlir::tt::tensix::TensixDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::ml_program::MLProgramDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                  mlir::tosa::TosaDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttmlir optimizer driver\n", registry));
}
