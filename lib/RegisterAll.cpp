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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
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
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
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

  // IR dumping will be set up when dialects are initialized
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
  linalg::registerSubsetOpInterfaceExternalModels(registry);
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
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
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
    mlir::MLIRContext *ctx, std::vector<std::string> passNamesToCache) {
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

mlir::tt::MLIRModuleLogger::Config
mlir::tt::MLIRModuleLogger::Config::fromEnvironment() {
  Config config;

  // Check if IR dumping is enabled
  const char *dumpEnabled = std::getenv("TTMLIR_DUMP_IR");
  if (dumpEnabled &&
      (std::string(dumpEnabled) == "1" || std::string(dumpEnabled) == "true")) {
    config.dumpEnabled = true;
  }

  // Get dump directory
  const char *dumpDir = std::getenv("TTMLIR_DUMP_IR_DIR");
  if (dumpDir) {
    config.dumpDir = std::string(dumpDir);
  } else {
    config.dumpDir = "./ir_dumps"; // Default directory
  }

  // Parse specific passes to dump
  const char *specificPasses = std::getenv("TTMLIR_DUMP_IR_PASSES");
  if (specificPasses) {
    std::string passes(specificPasses);
    std::stringstream ss(passes);
    std::string pass;
    while (std::getline(ss, pass, ',')) {
      if (!pass.empty()) {
        config.specificPasses.insert(pass);
      }
    }
  }

  // Check if dialect creation dumping is enabled
  const char *dumpDialects = std::getenv("TTMLIR_DUMP_IR_DIALECTS");
  if (dumpDialects && (std::string(dumpDialects) == "1" ||
                       std::string(dumpDialects) == "true")) {
    config.dumpDialectCreation = true;
  }

  // Check if debug info should be preserved
  const char *debugInfo = std::getenv("TTMLIR_DUMP_IR_DEBUG_INFO");
  if (debugInfo &&
      (std::string(debugInfo) == "0" || std::string(debugInfo) == "false")) {
    config.preserveDebugInfo = false;
  }

  return config;
}

void mlir::tt::MLIRModuleLogger::attachContextWithDumping(
    mlir::MLIRContext *ctx, const std::string &modelName, const std::string &pipelineName) {
  context = ctx;
  config = Config::fromEnvironment();
  
  // Use environment variable for model name if available, otherwise use parameter
  const char *envModelName = std::getenv("TTMLIR_DUMP_IR_MODEL_NAME");
  std::string finalModelName = envModelName ? std::string(envModelName) : modelName;
  
  // Use environment variable for pipeline name if available, otherwise use parameter
  const char *envPipelineName = std::getenv("TTMLIR_DUMP_IR_PIPELINE_NAME");
  std::string finalPipelineName = envPipelineName ? std::string(envPipelineName) : pipelineName;
  
  setModelName(finalModelName);
  setPipelineName(finalPipelineName);

  if (!config.dumpEnabled) {
    // Fall back to original behavior
    attachContext(ctx);
    return;
  }

  // Create dump directory if it doesn't exist
  std::filesystem::create_directories(config.dumpDir);

  context->registerActionHandler([this, modelName = this->modelName](llvm::function_ref<void()> transform,
                                        const mlir::tracing::Action &action) mutable {
    // Dump pre-pipeline IR
    if (moduleCache.empty()) {
      std::string passName = "PRE-PIPELINE", outString;
      llvm::raw_string_ostream os(outString);
      mlir::OpPrintingFlags flags;
      if (config.preserveDebugInfo) {
        flags.enableDebugInfo();
      }
      action.getContextIRUnits()[0].print(os, flags);
      os.flush();

      // Cache and dump to file
      moduleCache.emplace_back(passName, outString);
      dumpIRToFile(outString, getOutputFilename(passName));
    }

    // Run the transformation pass
    transform();

    // Dump post-pass IR
    if (mlir::isa<mlir::PassExecutionAction>(action)) {
      auto passAction = mlir::cast<mlir::PassExecutionAction>(action);
      std::string passName = passAction.getPass().getName().str();

      // Extract model name from the operation's location on first pass
      if (modelName == "unknown") {
        mlir::Operation *op = passAction.getOp();
        if (op) {
          std::string extractedName = extractModelNameFromLocation(op);
          if (extractedName != "unknown") {
            setModelName(extractedName);
          }
        }
      }

      // Check if we should dump this specific pass
      bool shouldDump = config.specificPasses.empty() ||
                        config.specificPasses.count(passName) > 0;

      if (shouldDump) {
        // Increment total pass count
        totalPassCount++;
        
        std::string outString;
        llvm::raw_string_ostream os(outString);
        mlir::OpPrintingFlags flags;
        if (config.preserveDebugInfo) {
          flags.enableDebugInfo();
        }
        passAction.getOp()->print(os, flags);
        os.flush();

        // Cache and dump to file
        moduleCache.emplace_back(passName, outString);
        dumpIRToFile(outString, getOutputFilename(passName));
      }
    }
  });
}

std::string
mlir::tt::MLIRModuleLogger::getOutputFilename(const std::string &passName,
                                              const std::string &stage) const {
  // Create a safe filename from the pass name
  std::string safeName = passName;
  std::replace(safeName.begin(), safeName.end(), '/', '_');
  std::replace(safeName.begin(), safeName.end(), '<', '_');
  std::replace(safeName.begin(), safeName.end(), '>', '_');
  std::replace(safeName.begin(), safeName.end(), ' ', '_');

  // Create safe model name
  std::string safeModelName = modelName;
  std::replace(safeModelName.begin(), safeModelName.end(), '/', '_');
  std::replace(safeModelName.begin(), safeModelName.end(), '<', '_');
  std::replace(safeModelName.begin(), safeModelName.end(), '>', '_');
  std::replace(safeModelName.begin(), safeModelName.end(), ' ', '_');
  std::replace(safeModelName.begin(), safeModelName.end(), '.', '_');

  // Create safe pipeline name
  std::string safePipelineName = pipelineName;
  std::replace(safePipelineName.begin(), safePipelineName.end(), '/', '_');
  std::replace(safePipelineName.begin(), safePipelineName.end(), '<', '_');
  std::replace(safePipelineName.begin(), safePipelineName.end(), '>', '_');
  std::replace(safePipelineName.begin(), safePipelineName.end(), ' ', '_');
  std::replace(safePipelineName.begin(), safePipelineName.end(), '.', '_');

  // Create filename with total pass count: <total_pass_count>_<pass_name>.mlir
  std::string filename = std::to_string(totalPassCount) + "_" + safeName;
  if (!stage.empty()) {
    filename += "_" + stage;
  }
  filename += ".mlir";

  // Create subdirectory structure: <model_name>/<pipeline_name>/
  std::string subdirPath = config.dumpDir + "/" + safeModelName + "/" + safePipelineName;
  
  return subdirPath + "/" + filename;
}

void mlir::tt::MLIRModuleLogger::dumpIRToFile(
    const std::string &irContent, const std::string &filename) const {
  // Create directory if it doesn't exist
  std::filesystem::path filePath(filename);
  std::filesystem::create_directories(filePath.parent_path());
  
  std::ofstream file(filename);
  if (file.is_open()) {
    file << irContent;
    file.close();
  }
}

void mlir::tt::MLIRModuleLogger::setModelName(const std::string &name) {
  modelName = name;
}

void mlir::tt::MLIRModuleLogger::setPipelineName(const std::string &name) {
  pipelineName = name;
}

std::string mlir::tt::MLIRModuleLogger::extractModelNameFromLocation(
    mlir::Operation *op) const {
  if (!op) {
    return "unknown";
  }

  mlir::Location loc = op->getLoc();
  
  // Try to extract filename from FileLineColLoc
  if (mlir::isa<mlir::FileLineColLoc>(loc)) {
    mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
    llvm::StringRef filename = fileLoc.getFilename();
    if (!filename.empty()) {
      // Extract just the filename without path and extension
      std::string filenameStr = filename.str();
      size_t lastSlash = filenameStr.find_last_of("/\\");
      if (lastSlash != std::string::npos) {
        filenameStr = filenameStr.substr(lastSlash + 1);
      }
      size_t lastDot = filenameStr.find_last_of(".");
      if (lastDot != std::string::npos) {
        filenameStr = filenameStr.substr(0, lastDot);
      }
      return filenameStr;
    }
  }
  
  // Try to extract from FusedLoc
  if (mlir::isa<mlir::FusedLoc>(loc)) {
    mlir::FusedLoc fusedLoc = mlir::cast<mlir::FusedLoc>(loc);
    for (mlir::Location subLoc : fusedLoc.getLocations()) {
      if (mlir::isa<mlir::FileLineColLoc>(subLoc)) {
        mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(subLoc);
        llvm::StringRef filename = fileLoc.getFilename();
        if (!filename.empty()) {
          // Extract just the filename without path and extension
          std::string filenameStr = filename.str();
          size_t lastSlash = filenameStr.find_last_of("/\\");
          if (lastSlash != std::string::npos) {
            filenameStr = filenameStr.substr(lastSlash + 1);
          }
          size_t lastDot = filenameStr.find_last_of(".");
          if (lastDot != std::string::npos) {
            filenameStr = filenameStr.substr(0, lastDot);
          }
          return filenameStr;
        }
      }
    }
  }
  
  return "unknown";
}

void mlir::tt::MLIRModuleLogger::dumpDialectCreation(
    const std::string &dialectName, mlir::MLIRContext *ctx) {
  Config config = Config::fromEnvironment();

  if (!config.dumpEnabled || !config.dumpDialectCreation) {
    return;
  }

  // Create dump directory if it doesn't exist
  std::filesystem::create_directories(config.dumpDir);

  std::string filename =
      config.dumpDir + "/dialect_" + dialectName + "_created.log";
  std::ofstream file(filename);
  if (file.is_open()) {
    file << "Dialect '" << dialectName << "' created in MLIRContext"
         << std::endl;
    file << "Available dialects after creation:" << std::endl;

    // List all loaded dialects
    for (const auto &loadedDialectName : ctx->getAvailableDialects()) {
      file << "  - " << loadedDialectName.str() << std::endl;
    }

    file.close();
  }
}

bool mlir::tt::MLIRModuleLogger::shouldEnableIRDumping() {
  Config config = Config::fromEnvironment();
  return config.dumpEnabled;
}
