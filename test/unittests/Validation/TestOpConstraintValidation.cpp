// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;
using namespace mlir::tt;

class OpConstraintValidationTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override {
    // Initialize context and module
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .openDevice();
  }

  void TearDown() override {
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .closeInstance();
  }

  TTNNLayoutAttr createTiledLayout(const llvm::ArrayRef<int64_t> &tensorShape,
                                   BufferType bufferType,
                                   TensorMemoryLayout tensorMemoryLayout,
                                   const llvm::ArrayRef<int64_t> &gridShape = {
                                       1, 1}) {
    auto elementType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    return TTNNLayoutAttr::get(
        &context, tensorShape, elementType, bufferType,
        mlir::tt::ttcore::GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }

  // Helper to create layout with custom element type
  TTNNLayoutAttr createTiledLayoutWithElementType(
      const llvm::ArrayRef<int64_t> &tensorShape, mlir::Type elementType,
      BufferType bufferType, TensorMemoryLayout tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridShape = {1, 1}) {
    auto tileType = mlir::tt::ttcore::TileType::get(elementType);
    return TTNNLayoutAttr::get(
        &context, tensorShape, tileType, bufferType,
        mlir::tt::ttcore::GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }

  // Helper to create a simple AddOp for testing
  AddOp createMockAddOp() {
    llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 32};
    auto layout = createTiledLayout(inputShape, BufferType::L1,
                                    TensorMemoryLayout::Interleaved);

    // Create tensor type with layout
    auto tensorType =
        mlir::RankedTensorType::get(inputShape, builder.getBF16Type(), layout);

    // Create two input tensors using OnesOp (simpler than EmptyOp)
    auto input1 = builder.create<OnesOp>(
        builder.getUnknownLoc(), tensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    auto input2 = builder.create<OnesOp>(
        builder.getUnknownLoc(), tensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    // Create AddOp
    return builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                 input1.getResult(), input2.getResult());
  }

  // Helper to create OpConfig for testing
  OpConfig createTestConfig() {
    auto outputLayout = createTiledLayout({1, 1, 32, 32}, BufferType::L1,
                                          TensorMemoryLayout::Interleaved);
    return OpConfig(outputLayout, OpConfig::OpSpecificAttrs{});
  }
};

// Test validateOperation with real AddOp and proper layouts.
TEST_F(OpConstraintValidationTest, ValidateOperationRealAddOp) {
  auto addOp = createMockAddOp();
  auto layouts = ttnn::utils::extractInputLayouts(addOp);
  OpConfig config = createTestConfig();
  float tensorL1UsageCap = 1.0f;

  auto result = op_constraint_validation::validateOperation(
      addOp, layouts, config, tensorL1UsageCap);

  // This should either succeed or fail gracefully (not crash)
  // The exact result depends on OpModel implementation
  if (result.isSuccess()) {
    EXPECT_GE(result.configIndex, 0u);
    EXPECT_TRUE(result.actualOutputLayout);
  } else {
    // Validation failed - check that error message is populated
    EXPECT_FALSE(result.errorMessage.empty());
    // Check that status is one of the error types
    EXPECT_TRUE(result.isError());
  }
}

// Test validateWithMultipleAttributes with real AddOp.
TEST_F(OpConstraintValidationTest, ValidateWithMultipleAttributesRealAddOp) {
  auto addOp = createMockAddOp();
  auto layouts = ttnn::utils::extractInputLayouts(addOp);

  // Create 10 empty attributes
  std::vector<OpConfig> configs(10);
  float tensorL1UsageCap = 1.0f;

  // Test with null reference configs (should succeed if validation passes)
  auto results = op_constraint_validation::validateWithMultipleAttributes(
      addOp, layouts, configs, /*referenceConfigs=*/{}, tensorL1UsageCap);

  EXPECT_EQ(results.size(), 10);
  // Each result should have a valid status
  for (const auto &result : results) {
    // Result can be success or any error type - just check it's valid
    if (result.isSuccess()) {
      EXPECT_GE(result.configIndex, 0u);
      EXPECT_TRUE(result.actualOutputLayout);
    } else {
      EXPECT_FALSE(result.errorMessage.empty());
    }
  }
}

// Test validateOperation with UpdateCacheOp expecting uint32 type for
// update_index.
TEST_F(OpConstraintValidationTest, UpdateCacheOpWithInvalidUpdateIndexType) {
  // Create cache tensor (4D tensor with BF16 type)
  llvm::SmallVector<int64_t> cacheShape = {1, 1, 64, 32};
  auto cacheLayout = createTiledLayout(cacheShape, BufferType::L1,
                                       TensorMemoryLayout::Interleaved);
  auto cacheTensorType = mlir::RankedTensorType::get(
      cacheShape, builder.getBF16Type(), cacheLayout);
  auto cacheOp = builder.create<OnesOp>(
      builder.getUnknownLoc(), cacheTensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, cacheShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Create input tensor (4D tensor with dim 2 = 1)
  llvm::SmallVector<int64_t> inputShape = {1, 1, 1, 32};
  auto inputLayout = createTiledLayout(inputShape, BufferType::L1,
                                       TensorMemoryLayout::Interleaved);
  auto inputTensorType = mlir::RankedTensorType::get(
      inputShape, builder.getBF16Type(), inputLayout);
  auto inputOp = builder.create<OnesOp>(
      builder.getUnknownLoc(), inputTensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Create update_index tensor with WRONG type (BF16 instead of uint32)
  // This should cause validation to fail
  llvm::SmallVector<int64_t> updateIndexShape = {1, 1, 32, 32};
  auto updateIndexLayout = createTiledLayout(updateIndexShape, BufferType::L1,
                                             TensorMemoryLayout::Interleaved);
  auto updateIndexTensorType = mlir::RankedTensorType::get(
      updateIndexShape, builder.getBF16Type(), updateIndexLayout);
  auto updateIndexOp = builder.create<OnesOp>(
      builder.getUnknownLoc(), updateIndexTensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, updateIndexShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Create UpdateCacheOp (inplace operation, no result type)
  auto updateCacheOp = builder.create<ttnn::UpdateCacheOp>(
      builder.getUnknownLoc(), cacheOp.getResult(), inputOp.getResult(),
      updateIndexOp.getResult(), /*batch_offset=*/0);

  // Extract layouts and create config
  auto layouts = ttnn::utils::extractInputLayouts(updateCacheOp);
  OpConfig config = createTestConfig();
  float tensorL1UsageCap = 1.0f;

  // Validate the operation
  auto result = op_constraint_validation::validateOperation(
      updateCacheOp, layouts, config, tensorL1UsageCap);

  // Should fail because update_index has wrong type (BF16 instead of uint32)
  EXPECT_TRUE(result.isError());
  EXPECT_FALSE(result.errorMessage.empty());

  // Now create a CORRECT UpdateCacheOp with uint32 type for update_index
  // Create uint32 type for update_index
  auto uint32Type = mlir::IntegerType::get(
      &context, 32, mlir::IntegerType::SignednessSemantics::Unsigned);
  auto uint32UpdateIndexLayout = createTiledLayoutWithElementType(
      updateIndexShape, uint32Type, BufferType::L1,
      TensorMemoryLayout::Interleaved);
  auto uint32UpdateIndexTensorType = mlir::RankedTensorType::get(
      updateIndexShape, uint32Type, uint32UpdateIndexLayout);
  auto uint32UpdateIndexOp = builder.create<OnesOp>(
      builder.getUnknownLoc(), uint32UpdateIndexTensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, updateIndexShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Create UpdateCacheOp with correct uint32 type
  auto validUpdateCacheOp = builder.create<ttnn::UpdateCacheOp>(
      builder.getUnknownLoc(), cacheOp.getResult(), inputOp.getResult(),
      uint32UpdateIndexOp.getResult(), /*batch_offset=*/0);

  // Extract layouts and validate
  auto validLayouts = ttnn::utils::extractInputLayouts(validUpdateCacheOp);
  auto validResult = op_constraint_validation::validateOperation(
      validUpdateCacheOp, validLayouts, config, tensorL1UsageCap);

  // Should succeed with uint32 type
  EXPECT_TRUE(validResult.isSuccess());
}

// Test ValidationStatus::NotImplemented
// ScatterOp returns ArchitecturalMismatch which maps to NotImplemented
TEST_F(OpConstraintValidationTest, ValidationStatusNotImplemented) {
  llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 32};
  auto layout = createTiledLayout(inputShape, BufferType::L1,
                                  TensorMemoryLayout::Interleaved);
  auto tensorType =
      mlir::RankedTensorType::get(inputShape, builder.getBF16Type(), layout);

  auto input = builder.create<OnesOp>(
      builder.getUnknownLoc(), tensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  auto indices = builder.create<OnesOp>(
      builder.getUnknownLoc(), tensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  auto scatterOp = builder.create<ScatterOp>(
      builder.getUnknownLoc(), tensorType, input.getResult(),
      indices.getResult(), input.getResult(),
      /*dim=*/0, /*memory_config=*/nullptr);

  auto layouts = ttnn::utils::extractInputLayouts(scatterOp);
  OpConfig config = createTestConfig();
  float tensorL1UsageCap = 1.0f;

  auto result = op_constraint_validation::validateOperation(
      scatterOp, layouts, config, tensorL1UsageCap);

  // Should return NotImplemented
  EXPECT_TRUE(result.isNotImplemented());
  EXPECT_EQ(result.status,
            op_constraint_validation::ValidationStatus::NotImplemented);
  EXPECT_TRUE(result.isError());
  EXPECT_FALSE(result.isSuccess());
  EXPECT_FALSE(result.errorMessage.empty());
}

// Test ValidationStatus::MetalBackendError
// ToLayoutOp with incompatible layout configurations triggers backend error
TEST_F(OpConstraintValidationTest, ValidationStatusMetalBackendError) {
  // Create helper for row major layout (non-tiled)
  auto createRowMajorHSLayout = [&](const llvm::ArrayRef<int64_t> &tensorShape,
                                    BufferType bufferType,
                                    TensorMemoryLayout tensorMemoryLayout) {
    // Row major uses scalar element type instead of tiled
    return TTNNLayoutAttr::get(
        &context, tensorShape, builder.getBF16Type(), bufferType,
        mlir::tt::ttcore::GridAttr::get(
            &context, {64, 1},
            mlir::tt::ttnn::optimizer_utils::
                createSingleDeviceVirtualToPhysicalAffineMap(
                    &context, tensorMemoryLayout)),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  };

  llvm::SmallVector<int64_t> tensorShape = {64, 1024};

  // Input: DRAM Tiled layout
  auto inputLayout = createTiledLayout(tensorShape, BufferType::DRAM,
                                       TensorMemoryLayout::Interleaved);
  auto inputTensorType = mlir::RankedTensorType::get(
      tensorShape, builder.getBF16Type(), inputLayout);

  auto input = builder.create<OnesOp>(
      builder.getUnknownLoc(), inputTensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, tensorShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Output: L1 RowMajor HeightSharded layout (incompatible with DRAM Tiled)
  auto outputLayout = createRowMajorHSLayout(tensorShape, BufferType::L1,
                                             TensorMemoryLayout::HeightSharded);
  auto outputTensorType = mlir::RankedTensorType::get(
      tensorShape, builder.getBF16Type(), outputLayout);

  // Create ToLayoutOp with incompatible input/output layouts
  auto toLayoutOp = builder.create<ToLayoutOp>(
      builder.getUnknownLoc(), outputTensorType, input.getResult(),
      LayoutAttr::get(&context, Layout::RowMajor),
      // ttcore::DataTypeAttr::get(&context, ttcore::DataType::BFloat16),
      /*dtype=*/nullptr,
      /*memory_config=*/nullptr);

  auto layouts = ttnn::utils::extractInputLayouts(toLayoutOp);
  OpConfig config(outputLayout, OpConfig::OpSpecificAttrs{});
  float tensorL1UsageCap = 1.0f;

  // Expected error message contains:
  // tt-metal/ttnn/core/tensor/layout/tensor_layout.cpp:111:
  // (physical_shard_shape.height() % tile_shape[0] == 0 &&
  // physical_shard_shape.width() % tile_shape[1] == 0)
  // info: Physical shard shape (1, 1024) must be tile {32, 32} sized!
  auto result = op_constraint_validation::validateOperation(
      toLayoutOp, layouts, config, tensorL1UsageCap);

  // Should return MetalBackendError due to incompatible layouts
  EXPECT_EQ(result.status,
            op_constraint_validation::ValidationStatus::MetalBackendError);
  EXPECT_TRUE(result.isError());
  EXPECT_FALSE(result.isSuccess());
  EXPECT_FALSE(result.isNotImplemented());
  EXPECT_FALSE(result.errorMessage.empty());
}

// Test ValidationStatus::OutOfMemoryError
// Use restrictive L1 usage cap to trigger out of memory error
TEST_F(OpConstraintValidationTest, ValidationStatusOutOfMemoryError) {
  llvm::SmallVector<int64_t> largeShape = {1, 1, 512, 512};
  auto layout = createTiledLayout(largeShape, BufferType::L1,
                                  TensorMemoryLayout::Interleaved);
  auto tensorType =
      mlir::RankedTensorType::get(largeShape, builder.getBF16Type(), layout);

  auto input1 = builder.create<OnesOp>(
      builder.getUnknownLoc(), tensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, largeShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  auto input2 = builder.create<OnesOp>(
      builder.getUnknownLoc(), tensorType,
      /*device=*/nullptr, ShapeAttr::get(&context, largeShape),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                     input1.getResult(), input2.getResult());

  auto layouts = ttnn::utils::extractInputLayouts(addOp);
  OpConfig config(layout, OpConfig::OpSpecificAttrs{});

  // Set very restrictive L1 usage cap (0.1% of L1)
  float tensorL1UsageCap = 0.001f;

  auto result = op_constraint_validation::validateOperation(
      addOp, layouts, config, tensorL1UsageCap);

  // Should return OutOfMemoryError
  EXPECT_EQ(result.status,
            op_constraint_validation::ValidationStatus::OutOfMemoryError);
  EXPECT_TRUE(result.isError());
  EXPECT_FALSE(result.isSuccess());
  EXPECT_FALSE(result.isNotImplemented());
  EXPECT_FALSE(result.errorMessage.empty());
}

// Test ValidationStatus::UnmatchedReferenceConfig
// Use validateWithMultipleAttributes with non-matching reference configs
TEST_F(OpConstraintValidationTest, ValidationStatusUnmatchedReferenceConfig) {
  auto addOp = createMockAddOp();
  auto layouts = ttnn::utils::extractInputLayouts(addOp);

  // Test config with L1 Interleaved layout
  std::vector<OpConfig> testConfigs;
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto testLayout =
      createTiledLayout(shape, BufferType::L1, TensorMemoryLayout::Interleaved);
  testConfigs.emplace_back(testLayout, OpConfig::OpSpecificAttrs{});

  // Reference config with DRAM layout (won't match L1 output)
  std::vector<OpConfig> referenceConfigs;
  auto refLayout = createTiledLayout(shape, BufferType::DRAM,
                                     TensorMemoryLayout::Interleaved);
  referenceConfigs.emplace_back(refLayout, OpConfig::OpSpecificAttrs{});

  float tensorL1UsageCap = 1.0f;

  auto results = op_constraint_validation::validateWithMultipleAttributes(
      addOp, layouts, testConfigs, referenceConfigs, tensorL1UsageCap);

  // Should have one result
  ASSERT_EQ(results.size(), 1);

  const auto &result = results[0];
  // Should return UnmatchedReferenceConfig when output layout doesn't match
  // reference
  EXPECT_EQ(
      result.status,
      op_constraint_validation::ValidationStatus::UnmatchedReferenceConfig);
  EXPECT_TRUE(result.isError());
  EXPECT_FALSE(result.isSuccess());
  EXPECT_FALSE(result.errorMessage.empty());
}
