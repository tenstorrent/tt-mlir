// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/RegisterAll.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Passes.h"

void mlir::tt::registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::tt::TTDialect, mlir::tt::ttir::TTIRDialect,
              mlir::tt::ttnn::TTNNDialect, mlir::tt::ttmetal::TTMetalDialect,
              mlir::tt::ttkernel::TTKernelDialect, mlir::arith::ArithDialect,
              mlir::func::FuncDialect, mlir::ml_program::MLProgramDialect,
              mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
              mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
              mlir::tosa::TosaDialect, mlir::vector::VectorDialect,
              mlir::emitc::EmitCDialect>();
}

void mlir::tt::registerAllPasses() {
  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerPasses();
  mlir::tt::ttmetal::registerPasses();

  mlir::PassPipelineRegistration<>(
      "ttir-to-ttmetal-backend-pipeline",
      "Pipeline lowering ttir to ttmetal backend.",
      mlir::tt::ttmetal::createTTIRToTTMetalBackendPipeline);

  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions>(
      "ttir-to-ttnn-backend-pipeline",
      "Pipeline lowering ttir to ttmetal backend.",
      mlir::tt::ttnn::createTTIRToTTNNBackendPipeline);
}
