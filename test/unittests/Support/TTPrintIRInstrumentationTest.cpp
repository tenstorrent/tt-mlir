// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/TTPrintIRInstrumentation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <string>

#include "llvm/Support/FileSystem.h"

using namespace mlir::tt;

// Type aliases for cleaner code
using Options = TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions;
using DumpLevel = TTPrintIRInstrumentation::DumpLevel;
using PipelineFunction = std::function<void(mlir::PassManager &)>;

namespace {
namespace test {

struct Constants {
  // Example output path:
  // test_ttmlir_ir_print-abc123/test_model/test_pipeline/0_Canonicalizer.mlir
  static constexpr const char *kTempDirPrefix = "test_ttmlir_ir_print";
  static constexpr const char *kTestModel = "test_model";
  static constexpr const char *kTestPipeline = "test_pipeline";
};

// File system utilities for test operations
class FileSystem {
public:
  static std::filesystem::path createUniqueTempDir() {
    llvm::SmallString<256> tempDir;
    if (llvm::sys::fs::createUniqueDirectory(Constants::kTempDirPrefix,
                                             tempDir)) {
      ADD_FAILURE() << "Could not create temporary directory";
      return {};
    }
    return tempDir.str().str();
  }

  static int countMlirFiles(const std::filesystem::path &dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
      return 0;
    }

    int count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".mlir") {
        count++;
      }
    }
    return count;
  }
};

// MLIR utilities for creating test constructs
class Utils {
public:
  static mlir::Location createTestLoc(mlir::OpBuilder &builder) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(__FILE__), __LINE__,
                                     0);
  }

  static mlir::OwningOpRef<mlir::ModuleOp>
  createTestModule(mlir::MLIRContext &context) {
    mlir::OpBuilder builder(&context);
    auto loc = createTestLoc(builder);

    auto module = mlir::ModuleOp::create(loc);

    // Create function directly inline
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp =
        builder.create<mlir::func::FuncOp>(loc, "test_func", funcType);
    funcOp.setPrivate();

    // Create function body inline
    auto &block = funcOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&block);

    auto const1 = builder.create<mlir::arith::ConstantIntOp>(loc, 42, 32);
    auto const2 = builder.create<mlir::arith::ConstantIntOp>(loc, 24, 32);
    (void)builder.create<mlir::arith::AddIOp>(loc, const1, const2);
    builder.create<mlir::func::ReturnOp>(loc);

    module.push_back(funcOp);
    return module;
  }
};

// Pipeline factory for creating different types of pass pipelines
namespace Pipelines {
inline PipelineFunction singlePassPipeline() {
  return [](mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
  };
}

inline PipelineFunction flatPipeline() {
  return [](mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
  };
}

// Nested pass manager - triggers pipeline hooks
inline PipelineFunction nestedFuncOpPipeline() {
  return [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::createCanonicalizerPass());
    funcPm.addPass(mlir::createCSEPass());
  };
}

// Mixed flat and nested passes
inline PipelineFunction FlatAndNested() {
  return [](mlir::PassManager &pm) {
    // Flat pass at top level
    pm.addPass(mlir::createCanonicalizerPass());

    // Nested pass manager for func::FuncOp
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::createCSEPass());

    // Another flat pass at top level
    pm.addPass(mlir::createCanonicalizerPass());
  };
}
} // namespace Pipelines

} // namespace test
} // namespace

// Test fixture class providing common setup/teardown
class InstrumentationTest : public ::testing::Test {
protected:
  void SetUp() override {
    tempDir = test::FileSystem::createUniqueTempDir();
    expectedOutputDir =
        tempDir / test::Constants::kTestModel / test::Constants::kTestPipeline;

    context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    module = test::Utils::createTestModule(*context);
    pm = std::make_unique<mlir::PassManager>(context.get());
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  // Force cleanup of PassManager to trigger instrumentation destructor
  void finalize() {
    pm.reset();
    module.release();
  }

  // Helper methods for cleaner test code
  int countOutputFiles() const {
    return test::FileSystem::countMlirFiles(expectedOutputDir);
  }

  bool hasOutput() const { return std::filesystem::exists(expectedOutputDir); }

public:
  std::filesystem::path tempDir;
  std::filesystem::path expectedOutputDir;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<mlir::PassManager> pm;
};

namespace {
namespace test {

// Helper to create default instrumentation options
Options createOptions(InstrumentationTest &fixture) {
  Options options;
  options.outputDir = fixture.tempDir.string();
  options.modelName = Constants::kTestModel;
  options.pipelineName = Constants::kTestPipeline;
  return options;
}

// Simple helper that just runs the instrumentation
void runWith(InstrumentationTest &fixture, Options options,
             PipelineFunction pipelineFn) {
  addTTPrintIRInstrumentation(*fixture.pm, options);
  pipelineFn(*fixture.pm);
  ASSERT_TRUE(succeeded(fixture.pm->run(*fixture.module)));
}

} // namespace test
} // namespace

// Once level tests - should dump only once at the very end (top-level only)
TEST_F(InstrumentationTest, Once_SinglePass) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Once;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  finalize(); // Trigger instrumentation destructor

  // Should dump once at the very end
  EXPECT_EQ(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Once_FlatMultiplePasses) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Once;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  finalize(); // Trigger instrumentation destructor

  // Should dump once at the very end (final state after all passes)
  EXPECT_EQ(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Once_Nested) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Once;
  test::runWith(*this, options, test::Pipelines::nestedFuncOpPipeline());

  finalize(); // Trigger instrumentation destructor

  // Should dump only top-level, not nested pipelines
  EXPECT_EQ(countOutputFiles(), 1);
}

// Pipeline level tests - these should dump at pipeline boundaries only
TEST_F(InstrumentationTest, Pipeline_SinglePass) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pipeline;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  finalize(); // Trigger instrumentation destructor

  // Should dump once at end of top-level pipeline (depth 0)
  EXPECT_EQ(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Pipeline_FlatMultiplePasses) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pipeline;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  finalize(); // Trigger instrumentation destructor

  // Should dump once at end of top-level pipeline (depth 0), not per pass
  EXPECT_EQ(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Pipeline_Nested) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pipeline;
  test::runWith(*this, options, test::Pipelines::nestedFuncOpPipeline());

  finalize(); // Trigger instrumentation destructor

  // Should dump twice: depth 0 (module) + depth 1 (nested func pipeline)
  EXPECT_EQ(countOutputFiles(), 2);
}

TEST_F(InstrumentationTest, Pipeline_MixedFlatAndNested) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pipeline;
  test::runWith(*this, options, test::Pipelines::FlatAndNested());

  finalize(); // Trigger instrumentation destructor

  // Should dump twice: depth 0 (module) + depth 1 (nested func pipeline)
  EXPECT_EQ(countOutputFiles(), 2);
}

TEST_F(InstrumentationTest, Pass) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  EXPECT_EQ(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Pass_Complex) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  EXPECT_GT(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Transformation) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Transformation;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  EXPECT_EQ(countOutputFiles(), 1);
}

TEST_F(InstrumentationTest, Transformation_Complex) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Transformation;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  EXPECT_GT(countOutputFiles(), 1);
}

// Test nested pass managers to trigger pipeline hooks
TEST_F(InstrumentationTest, NestedPipeline_FuncOp) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  test::runWith(*this, options, test::Pipelines::nestedFuncOpPipeline());

  EXPECT_GT(countOutputFiles(), 0);
}

// Test mixed flat + nested passes
TEST_F(InstrumentationTest, MixedFlatAndNested) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  test::runWith(*this, options, test::Pipelines::FlatAndNested());

  EXPECT_GT(countOutputFiles(), 0);
}
