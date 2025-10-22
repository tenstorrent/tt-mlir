// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <iostream>

namespace tt::alchemist::utils {

std::string getPipelineName(mlir::ModuleOp module,
                            CodeGenerationTarget target) {
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

  bool isOnlyTTNN = hasTTNN && !hasTTIR;
  switch (target) {
  case CodeGenerationTarget::Cpp:
    if (isOnlyTTNN) {
      return "ttnn-backend-to-emitc-pipeline";
    } else {
      return "ttir-to-emitc-pipeline";
    }
  case CodeGenerationTarget::Python:
    if (isOnlyTTNN) {
      return "ttnn-backend-to-emitpy-pipeline";
    } else {
      return "ttir-to-emitpy-pipeline";
    }
  }
}

bool runPipeline(mlir::PassManager &pm, mlir::ModuleOp module,
                 const std::string &pipelineName,
                 const std::string &pipelineOptions) {
  const auto *pipeline = mlir::PassPipelineInfo::lookup(pipelineName);
  if (!pipeline) {
    std::cout << "Failed to find " << pipelineName << std::endl;
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
} // namespace tt::alchemist::utils
