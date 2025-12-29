// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/TTPrintIRInstrumentation.h"

#include "llvm/Support/FileSystem.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include "gtest/gtest.h"

using namespace mlir::tt;

using Options = TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions;
using DumpLevel = TTPrintIRInstrumentation::DumpLevel;
using PipelineFunction = std::function<void(mlir::PassManager &)>;

namespace {
namespace test {

struct Constants {
  // e.g.
  // test_ttmlir_ir_print-abc123/test_model/test_pipeline/0_Canonicalizer.mlir
  static constexpr const char *kTempDirPrefix = "test_ttmlir_ir_print";
  static constexpr const char *kTestModel = "test_model";
  static constexpr const char *kTestPipeline = "test_pipeline";
};

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

    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type, i32Type}, i32Type);
    auto funcOp =
        mlir::func::FuncOp::create(builder, loc, "test_func", funcType);
    funcOp.setPrivate();

    auto &block = funcOp.getBody().emplaceBlock();
    for (auto argType : funcType.getInputs()) {
      block.addArgument(argType, loc);
    }
    builder.setInsertionPointToStart(&block);

    auto arg0 = block.getArgument(0);
    auto arg1 = block.getArgument(1);

    auto addResult = mlir::arith::AddIOp::create(builder, loc, arg0, arg1);
    mlir::func::ReturnOp::create(builder, loc, addResult.getResult());

    module.push_back(funcOp);
    return module;
  }
};

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

inline PipelineFunction nestedFuncOpPipeline() {
  return [](mlir::PassManager &pm) {
    mlir::OpPassManager &funcPm = pm.nest<mlir::func::FuncOp>();
    funcPm.addPass(mlir::createCanonicalizerPass());
    funcPm.addPass(mlir::createCSEPass());
  };
}

inline PipelineFunction idempotentPipeline() {
  return [](mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCanonicalizerPass());
  };
}
} // namespace Pipelines

} // namespace test
} // namespace

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

  int countOutputFiles() const {
    return test::FileSystem::countMlirFiles(expectedOutputDir);
  }

  std::vector<std::string> getOutputFilePaths() const {
    std::vector<std::string> filepaths;
    if (!std::filesystem::exists(tempDir)) {
      return filepaths;
    }
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(tempDir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".mlir") {
        // Get relative path from tempDir
        std::filesystem::path relativePath =
            std::filesystem::relative(entry.path(), tempDir);
        filepaths.push_back(relativePath.string());
      }
    }
    return filepaths;
  }

public:
  void finalize() {
    pm.reset();
    module.release();
  }

  std::filesystem::path tempDir;
  std::filesystem::path expectedOutputDir;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<mlir::PassManager> pm;
};

namespace {
namespace test {

Options createOptions(InstrumentationTest &fixture) {
  Options options;
  options.outputDir = fixture.tempDir.string();
  options.modelName = Constants::kTestModel;
  options.pipelineName = Constants::kTestPipeline;
  options.onlyDumpOnChanges = false;
  return options;
}

void runWith(InstrumentationTest &fixture, Options options,
             PipelineFunction pipelineFn) {
  addTTPrintIRInstrumentation(*fixture.pm, options);
  pipelineFn(*fixture.pm);
  ASSERT_TRUE(succeeded(fixture.pm->run(*fixture.module)));
  fixture.finalize();
}

} // namespace test
} // namespace

TEST_F(InstrumentationTest, Once) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Once;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  EXPECT_EQ(countOutputFiles(), 1);
  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(
      std::find(filepaths.begin(), filepaths.end(),
                "test_model/test_pipeline/1_after_test_pipeline.mlir") !=
      filepaths.end());
}

TEST_F(InstrumentationTest, PipelineFlat) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pipeline;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  EXPECT_EQ(countOutputFiles(), 1);
  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(
      std::find(filepaths.begin(), filepaths.end(),
                "test_model/test_pipeline/1_after_test_pipeline.mlir") !=
      filepaths.end());
}

TEST_F(InstrumentationTest, PipelineNested) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pipeline;
  test::runWith(*this, options, test::Pipelines::nestedFuncOpPipeline());

  EXPECT_EQ(countOutputFiles(), 2);
  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(
      std::find(filepaths.begin(), filepaths.end(),
                "test_model/test_pipeline/"
                "1_mlir__detail__OpToOpPassAdaptor_func_func_pipeline.mlir") !=
      filepaths.end());
  EXPECT_TRUE(
      std::find(filepaths.begin(), filepaths.end(),
                "test_model/test_pipeline/2_after_test_pipeline.mlir") !=
      filepaths.end());
}

TEST_F(InstrumentationTest, InitialWithOnce) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Once;
  options.dumpInitial = true;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  EXPECT_EQ(countOutputFiles(), 2);
  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(std::find(filepaths.begin(), filepaths.end(),
                        "test_model/test_pipeline/0_initial.mlir") !=
              filepaths.end());
  EXPECT_TRUE(
      std::find(filepaths.begin(), filepaths.end(),
                "test_model/test_pipeline/1_after_test_pipeline.mlir") !=
      filepaths.end());
}

class TestableTTPrintIRInstrumentation : public TTPrintIRInstrumentation {
public:
  using TTPrintIRInstrumentation::extractModelNameFromLocation;
  using TTPrintIRInstrumentation::TTPrintIRInstrumentation;
};

TEST_F(InstrumentationTest, ChangeDetectionEnabled) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  options.onlyDumpOnChanges = true;

  test::runWith(*this, options, test::Pipelines::idempotentPipeline());

  int fileCount = countOutputFiles();
  EXPECT_LE(fileCount, 2);
  EXPECT_GE(fileCount, 1);
}

TEST_F(InstrumentationTest, ChangeDetectionDisabled) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;

  test::runWith(*this, options, test::Pipelines::idempotentPipeline());

  EXPECT_EQ(countOutputFiles(), 3);
}

TEST_F(InstrumentationTest, ExtractModelNameFromLocation) {
  auto module = test::Utils::createTestModule(*context);

  TestableTTPrintIRInstrumentation instr(
      TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions{});
  EXPECT_EQ(instr.extractModelNameFromLocation(nullptr), "unknown");

  mlir::Operation *firstOp = &module->getBody()->front();
  std::string extracted = instr.extractModelNameFromLocation(firstOp);

  EXPECT_EQ(extracted, "TTPrintIRInstrumentationTest");
}

TEST_F(InstrumentationTest, ModelNameExplicitlyProvided) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  options.modelName = "custom_model";
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(std::find(filepaths.begin(), filepaths.end(),
                        "custom_model/test_pipeline/1_Canonicalizer.mlir") !=
              filepaths.end());
}

TEST_F(InstrumentationTest, ModelNameExtractedFromIR) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  options.modelName = "";
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(
      std::find(
          filepaths.begin(), filepaths.end(),
          "TTPrintIRInstrumentationTest/test_pipeline/1_Canonicalizer.mlir") !=
      filepaths.end());
}

TEST_F(InstrumentationTest, Pass) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Pass;
  test::runWith(*this, options, test::Pipelines::singlePassPipeline());

  EXPECT_EQ(countOutputFiles(), 1);
  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(std::find(filepaths.begin(), filepaths.end(),
                        "test_model/test_pipeline/1_Canonicalizer.mlir") !=
              filepaths.end());
}

TEST_F(InstrumentationTest, Transformation) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Transformation;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  EXPECT_GE(countOutputFiles(), 5);
  auto filepaths = getOutputFilePaths();
  bool hasPassExecution = std::any_of(
      filepaths.begin(), filepaths.end(), [](const std::string &path) {
        return path.find("_after") != std::string::npos;
      });
  bool hasPatternIteration = std::any_of(
      filepaths.begin(), filepaths.end(), [](const std::string &path) {
        return path.find("GreedyPatternRewriteIteration_iter") !=
               std::string::npos;
      });
  EXPECT_TRUE(hasPassExecution);
  EXPECT_TRUE(hasPatternIteration);
}

TEST_F(InstrumentationTest, InitialLevel) {
  auto options = test::createOptions(*this);
  options.level = DumpLevel::Initial;
  test::runWith(*this, options, test::Pipelines::flatPipeline());

  EXPECT_EQ(countOutputFiles(), 1);
  auto filepaths = getOutputFilePaths();
  EXPECT_TRUE(std::find(filepaths.begin(), filepaths.end(),
                        "test_model/test_pipeline/0_initial.mlir") !=
              filepaths.end());
}
