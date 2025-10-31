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
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

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
