// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <iostream>

namespace tt::alchemist::utils {

namespace {

bool isOnlyTTNN(mlir::ModuleOp module) {
  bool hasTTIR = false;
  bool hasTTNN = false;

  module->walk([&](mlir::Operation *op) {
    llvm::StringRef dialectName = op->getDialect()->getNamespace();
    if (dialectName == "ttir") {
      hasTTIR = true;
    } else if (dialectName == "ttnn") {
      hasTTNN = true;
    }
  });

  return hasTTNN && !hasTTIR;
}

} // namespace

std::string getPipelineName(mlir::ModuleOp module,
                            CodeGenerationTarget target) {
  bool ttnnInput = isOnlyTTNN(module);

  switch (target) {
  case CodeGenerationTarget::Cpp:
    return ttnnInput ? "ttnn-common-to-emitc-pipeline"
                     : "ttir-to-emitc-pipeline";
  case CodeGenerationTarget::Python:
    return ttnnInput ? "ttnn-common-to-emitpy-pipeline"
                     : "ttir-to-emitpy-pipeline";
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
  std::string pipelineName = getPipelineName(module, target);
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
