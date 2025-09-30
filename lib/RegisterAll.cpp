// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/RegisterAll.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/SFPI/IR/SFPI.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Pipelines/TTKernelPipelines.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
// #include "ttmlir/Dialect/SFPI/Transforms/Passes.h"  // Commented out until we
// have passes
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#if TTMLIR_ENABLE_STABLEHLO
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/transforms/passes.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "stablehlo/dialect/Register.h"
#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#endif

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

void mlir::tt::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      mlir::tt::ttcore::TTCoreDialect, mlir::tt::ttir::TTIRDialect,
      mlir::tt::d2m::D2MDialect, mlir::tt::ttnn::TTNNDialect,
      mlir::tt::ttmetal::TTMetalDialect, mlir::tt::ttkernel::TTKernelDialect,
      mlir::tt::sfpi::SFPIDialect, mlir::func::FuncDialect,
      mlir::arith::ArithDialect, mlir::math::MathDialect,
      mlir::ml_program::MLProgramDialect, mlir::tensor::TensorDialect,
      mlir::linalg::LinalgDialect, mlir::affine::AffineDialect,
      mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
      mlir::tosa::TosaDialect, mlir::vector::VectorDialect,
      mlir::memref::MemRefDialect, mlir::emitc::EmitCDialect,
      mlir::bufferization::BufferizationDialect, mlir::LLVM::LLVMDialect,
      mlir::quant::QuantDialect, mlir::tt::emitpy::EmitPyDialect>();

#if TTMLIR_ENABLE_STABLEHLO
  mlir::stablehlo::registerAllDialects(registry);
  mlir::sdy::registerAllDialects(registry);
  mlir::mpmd::registerAllDialects(registry);
#endif
}

void mlir::tt::registerAllExtensions(mlir::DialectRegistry &registry) {
  // Both the inliner for TTIRDialect and FuncDialect must be registered
  // since we use a combination of TTIRDialect and FuncDialect in the IR.
  mlir::func::registerInlinerExtension(registry);
  LLVM::registerInlinerInterface(registry);
  // Registering BufferizableOpInterface for each dialect (including
  // intermediate dialects) is required to convert types to memrefs during
  // lowering.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerConvertVectorToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertOpenMPToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  registerAllToLLVMIRTranslations(registry);
}

void mlir::tt::registerAllPasses() {
  // Register all dialect conversion passes.
  mlir::tt::registerTTMLIRConversionPasses();

  // Registering -remove-dead-values built-in mlir pass to optimize out the
  // unused OPs/operands after conversion.
  mlir::registerPass(mlir::createRemoveDeadValuesPass);

  mlir::tt::ttcore::registerPasses();
  mlir::tt::ttcore::registerTTPopulateArgumentTypes();
  mlir::tt::ttir::registerPasses();
  mlir::tt::d2m::registerPasses();
  mlir::tt::ttnn::registerTTNNOptimizer();
  mlir::tt::ttnn::registerPasses();
  mlir::tt::ttmetal::registerPasses();
  mlir::tt::ttkernel::registerPasses();
  mlir::tt::llvm_util::registerPasses();
  mlir::tt::transforms::registerPasses();

#if TTMLIR_ENABLE_STABLEHLO
  mlir::tt::stablehlo::registerPasses();
#endif

  // Register pipelines.
  mlir::tt::ttir::registerTTIRPipelines();
  mlir::tt::ttnn::registerTTNNPipelines();
  mlir::tt::ttmetal::registerTTMetalPipelines();
  mlir::tt::ttkernel::registerTTKernelPipelines();

#if TTMLIR_ENABLE_STABLEHLO
  // Register all SDY passes and pipelines.
  mlir::sdy::registerAllSdyPassesAndPipelines();

  // Register automatic sharding pipeline.
  mlir::tt::stablehlo::registerStableHLOPipeline();
#endif
}

void mlir::tt::MLIRModuleLogger::attachContext(
    mlir::MLIRContext *ctx, std::vector<std::string> passNamesToCache = {}) {
  context = ctx;

  context->registerActionHandler(
      [this, passNamesToCache](llvm::function_ref<void()> transform,
                               const mlir::tracing::Action &action) {
        // Also might make sense to store the _FIRST_ module. Or the module
        // before it was sent through the pipeline.
        if (moduleCache.empty()) {
          // Add it to the current Cache.
          std::string passName = "PRE-PIPELINE", outString;
          llvm::raw_string_ostream os(outString);
          mlir::OpPrintingFlags flags;
          flags.enableDebugInfo();
          action.getContextIRUnits()[0].print(os, flags);
          os.flush();
          moduleCache.emplace_back(passName, outString);
        }

        // Might make more sense to hold the module after a transformation has
        // occured.
        transform(); // Run the transformation pass.

        // Now save the module if it should be Cached.
        if (mlir::isa<mlir::PassExecutionAction>(action)) {
          auto passAction = mlir::cast<mlir::PassExecutionAction>(action);
          // A Pass action has occured, need to store the previous module
          // before transform is completed.
          std::string passName = passAction.getPass().getName().str();

          if (passNamesToCache.empty() or
              std::find(passNamesToCache.begin(), passNamesToCache.end(),
                        passName) != passNamesToCache.end()) {
            std::string outString;
            llvm::raw_string_ostream os(outString);
            mlir::OpPrintingFlags flags;
            flags.enableDebugInfo();
            passAction.getOp()->print(os, flags);
            os.flush();

            this->moduleCache.emplace_back(passName, outString);
          }
        }
      });
}
