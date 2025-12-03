// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.hpp"

#include "tt-alchemist/tt_alchemist_c_api.hpp"
#include "unit_test_generator.hpp"

// MLIR includes
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include <iostream>

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
                  mlir::func::FuncDialect, mlir::emitc::EmitCDialect,
                  mlir::LLVM::LLVMDialect, mlir::quant::QuantDialect,
                  mlir::sdy::SdyDialect>();
  context.appendDialectRegistry(registry);

  context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  context.loadDialect<mlir::tt::ttir::TTIRDialect>();
  context.loadDialect<mlir::tt::d2m::D2MDialect>();
  context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::emitc::EmitCDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::quant::QuantDialect>();
  context.loadDialect<mlir::sdy::SdyDialect>();

  // Register TTNN pipelines to make them available for lookup
  mlir::tt::ttnn::registerTTNNPipelines();
}

bool TTAlchemist::generateUnitTests(const std::string &input_file,
                                    const std::string &output_dir,
                                    const TestGenerationOptions &options) {
  // Parse the input MLIR file
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(input_file, &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR file: " << input_file << std::endl;
    return false;
  }

  // Create the test generator
  UnitTestGenerator generator(options);

  // Generate tests
  bool result = generator.generate(module.get(), output_dir);

  if (!result) {
    std::cerr << "Test generation failed: " << generator.getLastError() << std::endl;
  } else if (options.verbose) {
    std::cout << "Successfully generated unit tests in " << output_dir << std::endl;
  }

  return result;
}

bool TTAlchemist::generateUnitTestsFromString(const std::string &mlir_string,
                                              const std::string &output_dir,
                                              const TestGenerationOptions &options) {
  // Parse MLIR from string
  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_string, &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR string" << std::endl;
    return false;
  }

  // Create the test generator
  UnitTestGenerator generator(options);

  // Generate tests
  bool result = generator.generate(module.get(), output_dir);

  if (!result) {
    std::cerr << "Test generation failed: " << generator.getLastError() << std::endl;
  } else if (options.verbose) {
    std::cout << "Successfully generated unit tests in " << output_dir << std::endl;
  }

  return result;
}

} // namespace tt::alchemist

// C-compatible API implementations
extern "C" {

// Get the singleton instance
void *tt_alchemist_TTAlchemist_getInstance() {
  return static_cast<void *>(&tt::alchemist::TTAlchemist::getInstance());
}

} // extern "C"
