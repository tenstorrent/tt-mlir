// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/TTPrintIRInstrumentation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <string>

#include "llvm/Support/FileSystem.h"

using namespace mlir::tt;

// Helper function to create a simple MLIR module
mlir::OwningOpRef<mlir::ModuleOp> createTestModule(mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create a simple function
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   "test_func", funcType);
  funcOp.setPrivate();
  module.push_back(funcOp);

  return module;
}

// Helper function to create a unique temp directory
std::filesystem::path createUniqueTempDir() {
  llvm::SmallString<256> tempDir;
  if (llvm::sys::fs::createUniqueDirectory("test_ttmlir_ir_print", tempDir)) {
    ADD_FAILURE() << "Could not create temporary directory";
    return {};
  }
  return tempDir.str().str();
}

// Helper function to check if any .mlir files exist in directory
bool hasMlirFiles(const std::filesystem::path &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return false;
  }

  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".mlir") {
      return true;
    }
  }
  return false;
}

TEST(TTPrintIRInstrumentationTest, BasicCompilationWithFileOutput) {
  std::filesystem::path tempDir = createUniqueTempDir();
  std::filesystem::path expectedOutputDir =
      tempDir / "test_model" / "test_pipeline";

  // Setup MLIR context and module
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  auto module = createTestModule(context);
  mlir::PassManager pm(&context);

  TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions options;
  options.outputDir = tempDir.string();
  options.modelName = "test_model";
  options.pipelineName = "test_pipeline";

  // Add instrumentation and a pass to trigger it
  addTTPrintIRInstrumentation(pm, options);
  pm.addPass(mlir::createCanonicalizerPass());

  // Run compilation
  ASSERT_TRUE(succeeded(pm.run(*module)));

  // Verify output files were created
  EXPECT_TRUE(std::filesystem::exists(expectedOutputDir));
  EXPECT_TRUE(std::filesystem::is_directory(expectedOutputDir));
  EXPECT_TRUE(hasMlirFiles(expectedOutputDir));

  std::filesystem::remove_all(tempDir);
}
