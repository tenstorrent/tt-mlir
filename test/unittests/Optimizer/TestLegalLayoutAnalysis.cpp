// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir::tt::ttnn;

// Test fixture for testing the LegalLayoutAnalysis
class LegalLayoutAnalysisTest
    : public testing::TestWithParam<std::tuple<
          std::vector<int64_t>, std::vector<int64_t>, bool, int64_t>> {
protected:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;

  void SetUp() override {
    // Register necessary dialects
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();

    // Create a simple module with a function
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    // Create a function
    auto funcType = builder.getFunctionType({}, {});
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                              "test_func", funcType);

    // Create a basic block in the function
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
  }

  // Get parameters from test
  std::vector<int64_t> getTensorShape() const {
    return std::get<0>(GetParam());
  }

  std::vector<int64_t> getMaxGrid() const { return std::get<1>(GetParam()); }

  bool getRowMajorEnabled() const { return std::get<2>(GetParam()); }

  int64_t getMaxShardedConfigs() const { return std::get<3>(GetParam()); }

  // Helper method to create a tensor type with given dimensions
  mlir::RankedTensorType createTensorType(llvm::ArrayRef<int64_t> shape,
                                          mlir::Type elementType) {
    auto gridAttr = mlir::tt::ttcore::GridAttr::get(&context);
    TTNNLayoutAttr layoutAttr = TTNNLayoutAttr::get(
        &context, shape, elementType, BufferType::DRAM, gridAttr,
        TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
    return mlir::RankedTensorType::get(shape, elementType, layoutAttr);
  }

  // Create a set of scalar types for testing
  llvm::DenseSet<mlir::Type> createScalarTypeSet() {
    llvm::DenseSet<mlir::Type> types;
    types.insert(builder.getF32Type());
    types.insert(builder.getBF16Type());
    return types;
  }

  // Create test operations using parameterized values
  void createTestOps() {
    // Create tensor operations for testing
    auto f32Type = builder.getF32Type();

    // Create function with a test tensor type of the parameterized shape
    auto tensorType = createTensorType(getTensorShape(), f32Type);

    auto device = builder.create<mlir::tt::ttnn::GetDeviceOp>(
        builder.getUnknownLoc(), builder.getType<mlir::tt::ttnn::DeviceType>(),
        mlir::tt::ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1),
        mlir::tt::ttnn::MeshOffsetAttr::get(builder.getContext(), 0, 0));

    // Create a memory configuration
    auto memConfig = mlir::tt::ttnn::MemoryConfigAttr::get(
        &context,
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
            &context, mlir::tt::ttnn::TensorMemoryLayout::Interleaved),
        mlir::tt::ttnn::BufferTypeAttr::get(&context,
                                            mlir::tt::ttnn::BufferType::DRAM),
        std::nullopt);

    // Create an empty tensor
    auto empty = builder.create<mlir::tt::ttnn::EmptyOp>(
        builder.getUnknownLoc(), tensorType, device,
        mlir::tt::ttnn::ShapeAttr::get(&context, getTensorShape()),
        mlir::tt::ttcore::DataTypeAttr::get(
            &context, mlir::tt::ttcore::DataType::Float32),
        mlir::tt::ttnn::LayoutAttr::get(&context, Layout::Tile), memConfig);

    // Use that tensor in a ReluOp so we have a relevant op with a tensor result
    auto relu = builder.create<mlir::tt::ttnn::ReluOp>(builder.getUnknownLoc(),
                                                       empty.getResult());

    // Add return op
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         relu.getResult());
  }
};

// Test that LegalLayoutAnalysis correctly filters layouts based on RowMajor
// and maxShardedConfigs settings
TEST_P(LegalLayoutAnalysisTest, LegalLayoutAnalysisVariants) {
  SCOPED_TRACE("Params: shape=" + testing::PrintToString(getTensorShape()) +
               ", grid=" + testing::PrintToString(getMaxGrid()) +
               ", rowMajor=" + std::to_string(getRowMajorEnabled()) +
               ", maxSharded=" + std::to_string(getMaxShardedConfigs()));
  createTestOps();

  // Step 1: Run LegalTensorLayoutAnalysis
  auto scalarTypes = createScalarTypeSet();
  auto gridAttr = mlir::tt::ttcore::GridAttr::get(&context, getMaxGrid());
  LegalTensorLayoutAnalysisInput allLayoutsInput(gridAttr, &scalarTypes,
                                                 /*rowMajorAllowed=*/true);
  LegalTensorLayoutAnalysis allLayoutsAnalysis(module.get());
  allLayoutsAnalysis.init(allLayoutsInput);
  auto allLayoutsResult = allLayoutsAnalysis.getResult();

  // Verify that all possible layouts are generated
  ASSERT_FALSE(allLayoutsResult.empty());

  auto rowMajorEnabled = getRowMajorEnabled();
  auto maxShardedConfigs = getMaxShardedConfigs();

  // Step 2: Walk function ops and their sub-ops
  module->walk([&](mlir::func::FuncOp funcOp) {
    funcOp->walk([&](mlir::Operation *op) {
      if (!LegalOpLayoutAnalysis::isValidAnalysisTarget(op)) {
        return;
      }

      auto resultType =
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      auto it = allLayoutsResult.find(resultType);
      // Verify that layouts exist for this tensor type
      ASSERT_NE(it, allLayoutsResult.end());
      auto layoutsForTensor = it->second;

      // Verify that layouts are not empty
      EXPECT_FALSE(layoutsForTensor.empty());
      // Step 3: Run LegalLayoutAnalysis for this tensor type
      LegalOpLayoutAnalysisInput legalLayoutsInput(
          &layoutsForTensor, maxShardedConfigs,
          /*outputLayoutOverrides=*/nullptr, rowMajorEnabled);
      LegalOpLayoutAnalysis legalLayoutsAnalysis(op);
      legalLayoutsAnalysis.init(legalLayoutsInput);
      auto legalLayoutsResult = legalLayoutsAnalysis.getResult();

      // Verify that legal layouts are not empty
      EXPECT_FALSE(legalLayoutsResult.empty());

      // Validate legal layouts
      int shardedLayoutCount = 0;
      for (const auto &opConfig : legalLayoutsResult) {
        const auto &layout = opConfig.outputLayout;

        // Check RowMajor-specific constraints
        EXPECT_TRUE(rowMajorEnabled || layout.isTiled());

        // Count sharded layouts
        if (layout.hasShardedTensorMemoryLayout()) {
          ++shardedLayoutCount;
        }
      }

      // Verify that the number of sharded layouts does not exceed
      // maxShardedConfigs
      EXPECT_LE(shardedLayoutCount, maxShardedConfigs);
    });
  });
}

// Instantiate the test with different tensor shapes and grid sizes
INSTANTIATE_TEST_SUITE_P(
    ShapeAndGridVariations, LegalLayoutAnalysisTest,
    testing::Combine(
        // Different tensor shapes to test
        testing::Values(std::vector<int64_t>{256, 512, 32},
                        std::vector<int64_t>{128, 256, 64},
                        std::vector<int64_t>{512, 1024, 128},
                        std::vector<int64_t>{16, 16}),
        // Different max grid values to test
        testing::Values(std::vector<int64_t>{4, 4}, std::vector<int64_t>{8, 8},
                        std::vector<int64_t>{6, 6}, std::vector<int64_t>{2, 2}),
        // RowMajor enabled/disabled
        testing::Values(true, false),
        // Different max sharded configs
        testing::Values(0, 1, 8, 16, 32, 64, 128,
                        std::numeric_limits<int64_t>::max())));
