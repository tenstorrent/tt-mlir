// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.hpp"

#include "tt-alchemist/tt_alchemist_c_api.hpp"

// MLIR includes
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/MLIRContext.h"
#ifdef TTMLIR_ENABLE_STABLEHLO
#include "shardy/dialect/sdy/ir/dialect.h"
#endif
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

namespace tt::alchemist {

// Singleton implementation
TTAlchemist &TTAlchemist::getInstance() {
  static TTAlchemist instance;
  return instance;
}

TTAlchemist::TTAlchemist() {
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  registry.insert<mlir::tt::ttcore::TTCoreDialect, mlir::tt::ttir::TTIRDialect,
                  mlir::tt::d2m::D2MDialect, mlir::tt::ttnn::TTNNDialect,
                  mlir::tt::emitpy::EmitPyDialect, mlir::func::FuncDialect,
                  mlir::emitc::EmitCDialect, mlir::LLVM::LLVMDialect,
                  mlir::quant::QuantDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::tosa::TosaDialect
#ifdef TTMLIR_ENABLE_STABLEHLO
                  ,
                  mlir::sdy::SdyDialect
#endif
                  >();
  context.appendDialectRegistry(registry);

  context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  context.loadDialect<mlir::tt::ttir::TTIRDialect>();
  context.loadDialect<mlir::tt::d2m::D2MDialect>();
  context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::emitc::EmitCDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::quant::QuantDialect>();
#ifdef TTMLIR_ENABLE_STABLEHLO
  context.loadDialect<mlir::sdy::SdyDialect>();
#endif

  // Register TTNN pipelines to make them available for lookup
  mlir::tt::ttnn::registerTTNNPipelines();
}

} // namespace tt::alchemist

// C-compatible API implementations
extern "C" {

// Get the singleton instance
void *tt_alchemist_TTAlchemist_getInstance() {
  return static_cast<void *>(&tt::alchemist::TTAlchemist::getInstance());
}

} // extern "C"
