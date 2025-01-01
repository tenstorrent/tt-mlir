// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/RegisterAll.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Pipelines/TTKernelPipelines.h"
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#if TTMLIR_ENABLE_STABLEHLO
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#endif

void mlir::tt::registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::tt::TTDialect, mlir::tt::ttir::TTIRDialect,
              mlir::tt::ttnn::TTNNDialect, mlir::tt::ttmetal::TTMetalDialect,
              mlir::tt::ttkernel::TTKernelDialect, mlir::func::FuncDialect,
              mlir::arith::ArithDialect, mlir::ml_program::MLProgramDialect,
              mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
              mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
              mlir::tosa::TosaDialect, mlir::vector::VectorDialect,
              mlir::emitc::EmitCDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect>();
#if TTMLIR_ENABLE_STABLEHLO
  mlir::stablehlo::registerAllDialects(registry);
  mlir::sdy::registerAllDialects(registry);
#endif
  // Registering BufferizableOpInterface for each dialect (including
  // intermediate dialects) is required to convert types to memrefs during
  // lowering.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
}

void mlir::tt::registerAllExtensions(mlir::DialectRegistry &registry) {
  // Both the inliner for TTIRDialect and FuncDialect must be registered
  // since we use a combination of TTIRDialect and FuncDialect in the IR.
  mlir::func::registerInlinerExtension(registry);
}

void mlir::tt::registerAllPasses() {
  // Register all dialect conversion passes.
  mlir::tt::registerTTMLIRConversionPasses();

  // Registering -remove-dead-values built-in mlir pass to optimize out the
  // unused OPs/operands after conversion.
  mlir::registerPass(mlir::createRemoveDeadValuesPass);

  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerTTNNOptimizer();
  mlir::tt::ttnn::registerPasses();
  mlir::tt::ttmetal::registerPasses();

  // Register pipelines.
  mlir::tt::ttir::registerTTIRPipelines();
  mlir::tt::ttnn::registerTTNNPipelines();
  mlir::tt::ttmetal::registerTTMetalPipelines();
  mlir::tt::ttkernel::registerTTKernelPipelines();
}
