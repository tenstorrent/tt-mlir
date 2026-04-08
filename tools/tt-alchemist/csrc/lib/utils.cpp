// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <iostream>

namespace tt::alchemist::utils {

namespace {

// Check if the module contains ops of the provided dialect.
//
bool hasOpsOfDialect(mlir::ModuleOp module, llvm::StringRef dialectNamespace) {
  return module
      ->walk([&](mlir::Operation *op) {
        if (op->getDialect() &&
            op->getDialect()->getNamespace() == dialectNamespace) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

// Extract the inner module of the given type from the provided module.
// Used for extracting the inner modules from DeviceModuleOp and CPUModuleOp.
//
template <typename T>
mlir::ModuleOp extractInnerModule(mlir::ModuleOp module) {
  auto ops = module.getBodyRegion().getOps<T>();
  if (ops.empty()) {
    return nullptr;
  }

  return *(*ops.begin())
              .getBodyRegion()
              .template getOps<mlir::ModuleOp>()
              .begin();
}

} // namespace

// Helper function to determine which pipeline to run based on the current state
// of the module.
//
llvm::Expected<std::string> getPipelineName(mlir::ModuleOp module,
                                            CodeGenerationTarget target) {
  // If CPU module exists, it might have already been lowered to LLVM dialect.
  //
  if (auto cpuModule =
          extractInnerModule<mlir::tt::ttcore::CPUModuleOp>(module)) {
    if (hasOpsOfDialect(cpuModule, "llvm")) {
      return llvm::make_error<llvm::StringError>(
          "CPU module is already lowered to LLVM dialect, which means that the "
          "output of the ttir-to-ttnn-runtime-pipeline have been fed into the "
          "tt-alchemist. Instead, use the outputs of the "
          "ttir-to-ttnn-common-pipeline, or the initial TTIR module.",
          llvm::inconvertibleErrorCode());
    }
  }

  // We should run the E2E pipeline if we have TTIR ops in the Device module.
  // Otherwise, we can run the target-specific pipeline which assumes TTNN ops
  // in the Device module.
  //
  mlir::ModuleOp moduleToCheck = module;

  if (auto deviceModule =
          extractInnerModule<mlir::tt::ttcore::DeviceModuleOp>(module)) {
    moduleToCheck = deviceModule;
  }

  bool shouldRunE2EPipeline = hasOpsOfDialect(moduleToCheck, "ttir");

  switch (target) {
  case CodeGenerationTarget::Cpp:
    return shouldRunE2EPipeline ? "ttir-to-emitc-pipeline"
                                : "ttnn-common-to-emitc-pipeline";
  case CodeGenerationTarget::Python:
    return shouldRunE2EPipeline ? "ttir-to-emitpy-pipeline"
                                : "ttnn-common-to-emitpy-pipeline";
  }
}

bool runPipeline(mlir::PassManager &pm, mlir::ModuleOp module,
                 const std::string &pipelineName,
                 const std::string &pipelineOptions) {
  const auto *pipeline = mlir::PassPipelineInfo::lookup(pipelineName);
  if (!pipeline) {
    std::cout << "Failed to find pipeline: " << pipelineName << std::endl;
    return false;
  }

  std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
      [](const llvm::Twine &msg) {
        std::cout << "Pipeline error: " << msg.str() << std::endl;
        return mlir::failure();
      };

  if (mlir::failed(pipeline->addToPipeline(pm, pipelineOptions, err_handler))) {
    std::cout << "Failed to add pipeline with options: " << pipelineOptions
              << std::endl;
    return false;
  }

  if (mlir::failed(pm.run(module))) {
    std::cout << "Failed to run pipeline: " << pipelineName << std::endl;
    return false;
  }
  return true;
}

bool runPipeline(mlir::PassManager &pm, mlir::ModuleOp module,
                 CodeGenerationTarget target,
                 const std::string &pipelineOptions) {
  auto pipelineNameOrError = getPipelineName(module, target);
  if (!pipelineNameOrError) {
    std::cout << "Failed to determine which pipeline to run: "
              << llvm::toString(pipelineNameOrError.takeError()) << std::endl;
    return false;
  }

  std::string pipelineName = *pipelineNameOrError;
  return runPipeline(pm, module, pipelineName, pipelineOptions);
}

void formatCode(const fs::path &filePath, CodeGenerationTarget target) {
  if (target == CodeGenerationTarget::Python) {
    std::string formatCommand = "black --quiet " + filePath.string() + " 2>&1";
    int formatResult = std::system(formatCommand.c_str());
    if (formatResult != 0) {
      std::cout
          << "Warning: Failed to format Python file with black (exit code: "
          << formatResult << ")" << std::endl;
      std::cout << "The generated Python code is still available at: "
                << filePath << std::endl;
    }
  } else if (target == CodeGenerationTarget::Cpp) {
    std::string formatCommand =
        "clang-format -i " + filePath.string() + " > /dev/null 2>&1";
    int formatResult = std::system(formatCommand.c_str());
    if (formatResult != 0) {
      std::cout
          << "Warning: Failed to format C++ file with clang-format (exit code: "
          << formatResult << ")" << std::endl;
      std::cout << "The generated C++ code is still available at: " << filePath
                << std::endl;
    }
  }
}

} // namespace tt::alchemist::utils
