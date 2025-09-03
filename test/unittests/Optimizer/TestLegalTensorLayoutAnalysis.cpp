// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <optional>

using namespace mlir::tt::ttnn;

// Test fixture for testing the LegalTensorLayoutAnalysis
class LegalTensorLayoutAnalysisTest
    : public testing::TestWithParam<
          std::tuple<std::vector<int64_t>, std::vector<int64_t>>> {
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

    // Create an empty op with all required parameters
    builder.create<mlir::tt::ttnn::EmptyOp>(
        builder.getUnknownLoc(), tensorType, device,
        mlir::tt::ttnn::ShapeAttr::get(&context, getTensorShape()),
        mlir::tt::ttcore::DataTypeAttr::get(
            &context, mlir::tt::ttcore::DataType::Float32),
        mlir::tt::ttnn::LayoutAttr::get(&context, Layout::Tile), memConfig);

    // Add return op
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  }
};

// Test that layouts are correctly generated for all tensor types with different
// parameters.
//
// This test must be run serially for different inputs. Layout validation
// includes calls into TensorSpec APIs that require the creation of the cluster
// descriptor file. If multiple processes/threads attempt that in parallel, the
// test will fail
TEST_P(LegalTensorLayoutAnalysisTest, GenerateAndCategorizeLayouts) {
  createTestOps();

  // Create the analysis input
  auto scalarTypes = createScalarTypeSet();
  auto gridAttr = mlir::tt::ttcore::GridAttr::get(&context, getMaxGrid());
  LegalTensorLayoutAnalysisInput input(gridAttr, &scalarTypes, true);

  // Run the analysis
  LegalTensorLayoutAnalysis analysis(module.get());
  analysis.init(input);
  auto result = analysis.getResult();

  // Verify that results are not empty
  EXPECT_FALSE(result.empty());

  // Find the test tensor type in the results
  mlir::RankedTensorType testTensorType =
      createTensorType(getTensorShape(), builder.getF32Type());
  auto testTensorIt = result.find(testTensorType);
  EXPECT_NE(testTensorIt, result.end());

  // Check that layouts were generated for each scalar type
  for (auto scalarType : scalarTypes) {
    // Find the tensor type in the results
    auto entry = testTensorIt->second.find(scalarType);
    EXPECT_NE(entry, testTensorIt->second.end());

    // Check layouts for both memory layout types
    for (size_t dataLayoutIdx = 0;
         dataLayoutIdx < static_cast<size_t>(TensorPageLayout::kNumValues);
         ++dataLayoutIdx) {

      // Check for interleaved layouts
      const auto &interleavedLayouts =
          entry->second[dataLayoutIdx][static_cast<size_t>(
              TensorMemoryLayoutIndex::Interleaved)];

      // Check for sharded layouts
      const auto &shardedLayouts =
          entry->second[dataLayoutIdx]
                       [static_cast<size_t>(TensorMemoryLayoutIndex::Sharded)];

      // Verify that we have layouts for at least one memory layout type
      EXPECT_FALSE(interleavedLayouts.empty() && shardedLayouts.empty());
    }
  }
}

// Instantiate the test with different tensor shapes and grid sizes
INSTANTIATE_TEST_SUITE_P(
    ShapeAndGridVariations, LegalTensorLayoutAnalysisTest,
    testing::Combine(
        // Different tensor shapes to test
        testing::Values(std::vector<int64_t>{256, 512, 32},
                        std::vector<int64_t>{128, 256, 64},
                        std::vector<int64_t>{512, 1024, 128}),
        // Different max grid values to test
        testing::Values(std::vector<int64_t>{4, 4}, std::vector<int64_t>{8, 8},
                        std::vector<int64_t>{6, 6})));
