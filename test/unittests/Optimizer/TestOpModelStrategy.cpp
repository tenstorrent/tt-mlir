// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/EmbeddingRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/NormalizationRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TransformerRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;
using namespace mlir::tt::ttnn::op_constraint_validation;
using namespace mlir::tt;

class OpModelStrategyTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .openDevice();

    setL1UsageCap(1.0f);
  }

  void TearDown() override {
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .closeInstance();
  }

  void setL1UsageCap(float cap) {
    module->getOperation()->setAttr(utils::g_TensorL1UsageCapAttrName,
                                    builder.getF32FloatAttr(cap));
  }

  TTNNLayoutAttr createTiledLayout(const llvm::ArrayRef<int64_t> &tensorShape,
                                   BufferType bufferType,
                                   TensorMemoryLayout tensorMemoryLayout,
                                   const llvm::ArrayRef<int64_t> &gridShape = {
                                       1, 1}) {
    auto elementType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
    return TTNNLayoutAttr::Builder(&context, tensorShape, elementType)
        .setBufferType(bufferType)
        .setMemoryLayout(tensorMemoryLayout)
        .setGridShape(gridShape)
        .buildWithCanonicalCorePlacement(deviceAttr);
  }

  TTNNLayoutAttr
  createDRAMInterleavedLayout(const llvm::ArrayRef<int64_t> &tensorShape) {
    return createTiledLayout(tensorShape, BufferType::DRAM,
                             TensorMemoryLayout::Interleaved);
  }

  TTNNLayoutAttr
  createL1InterleavedLayout(const llvm::ArrayRef<int64_t> &tensorShape) {
    return createTiledLayout(tensorShape, BufferType::L1,
                             TensorMemoryLayout::Interleaved);
  }

  TTNNLayoutAttr
  createL1ShardedLayout(const llvm::ArrayRef<int64_t> &tensorShape,
                        const llvm::ArrayRef<int64_t> &gridShape = {8, 4}) {
    return createTiledLayout(tensorShape, BufferType::L1,
                             TensorMemoryLayout::HeightSharded, gridShape);
  }

  // Create a simple AddOp for testing.
  AddOp createMockAddOp(const llvm::ArrayRef<int64_t> &inputShape = {1, 1, 32,
                                                                     32}) {
    auto layout = createL1InterleavedLayout(inputShape);
    auto tensorType =
        mlir::RankedTensorType::get(inputShape, builder.getBF16Type(), layout);

    auto input1 = builder.create<OnesOp>(builder.getUnknownLoc(), tensorType,
                                         /*device=*/nullptr,
                                         ShapeAttr::get(&context, inputShape));

    auto input2 = builder.create<OnesOp>(builder.getUnknownLoc(), tensorType,
                                         /*device=*/nullptr,
                                         ShapeAttr::get(&context, inputShape));

    return builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                 input1.getResult(), input2.getResult());
  }

  // Create a ReshapeOp for testing.
  ReshapeOp createMockReshapeOp(
      const llvm::ArrayRef<int64_t> &inputShape = {1, 1, 32, 32},
      const llvm::ArrayRef<int64_t> &outputShape = {1, 32, 32}) {
    auto inputLayout = createL1InterleavedLayout(inputShape);
    auto inputTensorType = mlir::RankedTensorType::get(
        inputShape, builder.getBF16Type(), inputLayout);

    auto outputLayout = createDRAMInterleavedLayout(outputShape);
    auto outputTensorType = mlir::RankedTensorType::get(
        outputShape, builder.getBF16Type(), outputLayout);

    auto input = builder.create<OnesOp>(
        builder.getUnknownLoc(), inputTensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape));

    llvm::SmallVector<int32_t> outputShapeI32(outputShape.begin(),
                                              outputShape.end());
    return builder.create<ReshapeOp>(builder.getUnknownLoc(), outputTensorType,
                                     input.getResult(),
                                     builder.getI32ArrayAttr(outputShapeI32));
  }

  // Create a MatmulOp for testing.
  MatmulOp
  createMockMatmulOp(const llvm::ArrayRef<int64_t> &lhsShape = {1, 1, 32, 64},
                     const llvm::ArrayRef<int64_t> &rhsShape = {1, 1, 64, 32}) {
    auto lhsLayout = createL1InterleavedLayout(lhsShape);
    auto lhsTensorType =
        mlir::RankedTensorType::get(lhsShape, builder.getBF16Type(), lhsLayout);

    auto rhsLayout = createL1InterleavedLayout(rhsShape);
    auto rhsTensorType =
        mlir::RankedTensorType::get(rhsShape, builder.getBF16Type(), rhsLayout);

    llvm::SmallVector<int64_t> outputShape = {lhsShape[0], lhsShape[1],
                                              lhsShape[2], rhsShape[3]};
    auto outputLayout = createL1InterleavedLayout(outputShape);
    auto outputTensorType = mlir::RankedTensorType::get(
        outputShape, builder.getBF16Type(), outputLayout);

    auto lhs = builder.create<OnesOp>(builder.getUnknownLoc(), lhsTensorType,
                                      /*device=*/nullptr,
                                      ShapeAttr::get(&context, lhsShape));

    auto rhs = builder.create<OnesOp>(builder.getUnknownLoc(), rhsTensorType,
                                      /*device=*/nullptr,
                                      ShapeAttr::get(&context, rhsShape));

    return builder.create<MatmulOp>(builder.getUnknownLoc(), outputTensorType,
                                    lhs.getResult(), rhs.getResult(),
                                    /*transpose_a=*/false,
                                    /*transpose_b=*/false,
                                    /*matmul_program_config=*/nullptr,
                                    /*activation=*/nullptr);
  }

  // Create legal configs for an elementwise op (DRAM + L1-interleaved).
  std::vector<OpConfig> createElementwiseLegalConfigs(
      const llvm::ArrayRef<int64_t> &shape = {1, 1, 32, 32}) {
    std::vector<OpConfig> configs;
    configs.emplace_back(createDRAMInterleavedLayout(shape));
    configs.emplace_back(createL1InterleavedLayout(shape));
    configs.emplace_back(createL1ShardedLayout(shape, {1, 1}));
    return configs;
  }
};

class OpRuleBookTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    mlir::tt::ttcore::registerDevice(module.get());
  }

  TTNNLayoutAttr createTiledLayout(
      const llvm::ArrayRef<int64_t> &tensorShape, BufferType bufferType,
      TensorMemoryLayout tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridShape = {1, 1}) {
    auto elementType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
    return TTNNLayoutAttr::Builder(&context, tensorShape, elementType)
        .setBufferType(bufferType)
        .setMemoryLayout(tensorMemoryLayout)
        .setGridShape(gridShape)
        .buildWithCanonicalCorePlacement(deviceAttr);
  }

  TTNNLayoutAttr
  createTiledLayoutWithCoreRange(const llvm::ArrayRef<int64_t> &tensorShape,
                                 TensorMemoryLayout tensorMemoryLayout,
                                 const llvm::ArrayRef<int64_t> &gridShape,
                                 uint64_t startX, uint64_t startY,
                                 uint64_t endX, uint64_t endY) {
    auto elementType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    auto start = CoreCoordAttr::get(&context, startX, startY);
    auto end = CoreCoordAttr::get(&context, endX, endY);
    auto ranges = CoreRangeSetAttr::get(
        &context, {CoreRangeAttr::get(&context, start, end)});
    return TTNNLayoutAttr::Builder(&context, tensorShape, elementType)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(tensorMemoryLayout)
        .setGridShape(gridShape)
        .setCoreRangeSet(ranges)
        .build();
  }

  TTNNLayoutAttr
  createDRAMInterleavedLayout(const llvm::ArrayRef<int64_t> &tensorShape) {
    return createTiledLayout(tensorShape, BufferType::DRAM,
                             TensorMemoryLayout::Interleaved);
  }

  OpConfig createMcast2DHint(TTNNLayoutAttr output, uint64_t in0BlockW,
                             uint64_t perCoreM, uint64_t perCoreN = 1) {
    auto grid = CoreCoordAttr::get(&context, 2, 2);
    auto config = MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
        &context, grid, in0BlockW,
        /*outSubblockH=*/1, /*outSubblockW=*/1,
        /*outBlockH=*/perCoreM, /*outBlockW=*/perCoreN, perCoreM, perCoreN,
        /*transposeMcast=*/false,
        /*fusedActivation=*/UnaryWithParamAttr(), /*fuseBatch=*/true);
    return OpConfig(output, MatmulAttrs{config, std::nullopt});
  }

  OpConfig createMcast1DHint(TTNNLayoutAttr output, uint64_t in0BlockW,
                             uint64_t perCoreM, bool mcastIn0,
                             uint64_t perCoreN = 1) {
    auto grid = CoreCoordAttr::get(&context, 1, 8);
    auto hopCores = CoreRangeSetAttr::get(&context, {});
    auto config = MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
        &context, grid, in0BlockW,
        /*outSubblockH=*/1, /*outSubblockW=*/1,
        /*outBlockH=*/perCoreM, /*outBlockW=*/perCoreN, perCoreM, perCoreN,
        /*fuseBatch=*/true,
        /*fusedActivation=*/UnaryWithParamAttr(), mcastIn0,
        /*gatherIn0=*/false, hopCores,
        /*numGlobalCbReceivers=*/0, /*untilizeOut=*/false);
    return OpConfig(output, MatmulAttrs{config, std::nullopt});
  }
};

//===----------------------------------------------------------------------===//
// getOutputHints tests
//===----------------------------------------------------------------------===//

TEST_F(OpModelStrategyTest, DefaultOpNullOnlyInPrimaryHints) {
  auto addOp = createMockAddOp();
  auto legalConfigs = createElementwiseLegalConfigs();

  OutputHints hints = getOutputHints(addOp, legalConfigs);

  // Primary hints should contain only the NULL hint.
  EXPECT_EQ(hints.hints.size(), 1u);
  EXPECT_FALSE(hints.hints[0].outputLayout);
}

TEST_F(OpModelStrategyTest, DefaultOpShardedInFallbackHints) {
  auto addOp = createMockAddOp();
  auto legalConfigs = createElementwiseLegalConfigs();

  OutputHints hints = getOutputHints(addOp, legalConfigs);

  // Fallback hints should contain only sharded configs.
  EXPECT_FALSE(hints.fallbackHints.empty());
  for (const auto &hint : hints.fallbackHints) {
    ASSERT_TRUE(hint.outputLayout);
    auto memLayout = hint.outputLayout.getMemLayout();
    ASSERT_TRUE(memLayout);
    EXPECT_TRUE(isShardedMemoryLayout(memLayout.getValue()));
  }

  // Primary hints should have no sharded configs.
  for (const auto &hint : hints.hints) {
    if (hint.outputLayout && hint.outputLayout.getMemLayout()) {
      EXPECT_FALSE(
          isShardedMemoryLayout(hint.outputLayout.getMemLayout().getValue()));
    }
  }
}

TEST_F(OpModelStrategyTest, MatmulOpFiltersL1Interleaved) {
  auto matmulOp = createMockMatmulOp();
  auto legalConfigs = createElementwiseLegalConfigs();

  OutputHints hints = getOutputHints(matmulOp, legalConfigs);

  // L1-interleaved configs are filtered out for matmul (no program config
  // generated -> HiFi4 fallback). Remaining: DRAM + L1-sharded variants.
  EXPECT_LT(hints.hints.size(), legalConfigs.size());

  // No hint should be L1-interleaved.
  for (const auto &hint : hints.hints) {
    if (hint.outputLayout &&
        hint.outputLayout.getBufferType() == BufferType::L1 &&
        hint.outputLayout.getMemLayout() &&
        hint.outputLayout.getMemLayout().getValue() ==
            TensorMemoryLayout::Interleaved) {
      FAIL() << "Matmul hints should not contain L1-interleaved configs";
    }
  }
}

TEST_F(OpModelStrategyTest, MatmulPreflightUsesProgramConfigFromOperation) {
  auto matmulOp = createMockMatmulOp();
  auto grid = CoreCoordAttr::get(&context, 8, 8);
  auto nativeConfig = MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      &context, grid, /*in0BlockW=*/1,
      /*outSubblockH=*/1, /*outSubblockW=*/1,
      /*outBlockH=*/1, /*outBlockW=*/1,
      /*perCoreM=*/1, /*perCoreN=*/1,
      /*transposeMcast=*/false,
      /*fusedActivation=*/UnaryWithParamAttr(), /*fuseBatch=*/true);
  matmulOp.setMatmulProgramConfigAttr(nativeConfig);

  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto shardedInput = createL1ShardedLayout(shape, {1, 1});
  auto interleaved = createDRAMInterleavedLayout(shape);
  OpConfig candidate(interleaved);

  EXPECT_TRUE(getMatmulPreflightError({shardedInput, interleaved}, candidate)
                  .value()
                  .find("requires an explicit program config") !=
              std::string::npos);
  EXPECT_FALSE(getMatmulPreflightError(
      {shardedInput, interleaved}, candidate, matmulOp.getOperation()));
}

TEST_F(OpModelStrategyTest, ReshapeOpSkipsL1Sharding) {
  auto reshapeOp = createMockReshapeOp();
  auto legalConfigs = createElementwiseLegalConfigs();

  OutputHints hints = getOutputHints(reshapeOp, legalConfigs);

  // All hints should be non-sharded (DRAM or L1-interleaved).
  for (const auto &hint : hints.hints) {
    if (hint.outputLayout && hint.outputLayout.getMemLayout()) {
      EXPECT_FALSE(
          isShardedMemoryLayout(hint.outputLayout.getMemLayout().getValue()));
    }
  }
}

TEST_F(OpModelStrategyTest, UnknownOpUsesDefaultStrategy) {
  // Use AddOp as a stand-in for "default" path testing.
  auto addOp = createMockAddOp();
  auto legalConfigs = createElementwiseLegalConfigs();

  OutputHints hints = getOutputHints(addOp, legalConfigs);

  // First hint should be NULL.
  EXPECT_FALSE(hints.hints[0].outputLayout);
}

TEST_F(OpRuleBookTest, RmsNormShardedInputAcceptsCompatibleOutputHints) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto input = createTiledLayout(shape, BufferType::L1,
                                 TensorMemoryLayout::WidthSharded, {1, 8});
  auto matchingOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  RmsNormRuleBook rules;

  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      OpConfig(TTNNLayoutAttr()), {input}));
  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      OpConfig(matchingOutput), {input}));
}

TEST_F(OpRuleBookTest, RmsNormShardedInputRejectsIncompatibleOutputs) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto input = createTiledLayout(shape, BufferType::L1,
                                 TensorMemoryLayout::WidthSharded, {1, 8});
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto heightSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto blockSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});
  auto dramWidthSharded = createTiledLayout(
      shape, BufferType::DRAM, TensorMemoryLayout::WidthSharded, {1, 8});
  RmsNormRuleBook rules;

  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      OpConfig(interleaved), {input}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      OpConfig(heightSharded), {input}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      OpConfig(blockSharded), {input}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      OpConfig(dramWidthSharded), {input}));
}

TEST_F(OpRuleBookTest, RmsNormRequiresIdenticalPhysicalShardSpec) {
  llvm::SmallVector<int64_t> shape = {1, 1024, 16, 128};
  auto input = createTiledLayout(shape, BufferType::L1,
                                 TensorMemoryLayout::BlockSharded, {8, 4});
  auto matchingOutput =
      createTiledLayout(shape, BufferType::L1,
                        TensorMemoryLayout::BlockSharded, {8, 4});
  auto mismatchedOutput =
      createTiledLayout(shape, BufferType::L1,
                        TensorMemoryLayout::BlockSharded, {7, 4});
  RmsNormRuleBook rules;

  ASSERT_EQ(input.getShardShape().size(), 2u);
  ASSERT_EQ(mismatchedOutput.getShardShape().size(), 2u);
  EXPECT_EQ(input.getShardShape()[0], 128);
  EXPECT_EQ(input.getShardShape()[1], 1);
  EXPECT_EQ(mismatchedOutput.getShardShape()[0], 147);
  EXPECT_EQ(mismatchedOutput.getShardShape()[1], 1);
  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      OpConfig(matchingOutput), {input}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      OpConfig(mismatchedOutput), {input}));
}

TEST_F(OpRuleBookTest,
       RmsNormFusedResidualPrunesMismatchedInputCombinations) {
  builder.setInsertionPointToStart(&module->getBodyRegion().front());
  llvm::SmallVector<int64_t> shape = {1, 1024, 16, 128};
  llvm::SmallVector<int64_t> weightShape = {128};
  auto inputLayout = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {8, 4});
  auto residualLayout = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {7, 4});
  auto weightLayout = createDRAMInterleavedLayout(weightShape);

  auto inputType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), inputLayout);
  auto residualType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), residualLayout);
  auto weightType = mlir::RankedTensorType::get(
      weightShape, builder.getBF16Type(), weightLayout);
  auto input = builder.create<OnesOp>(
      builder.getUnknownLoc(), inputType, /*device=*/nullptr,
      ShapeAttr::get(&context, shape));
  auto residual = builder.create<OnesOp>(
      builder.getUnknownLoc(), residualType, /*device=*/nullptr,
      ShapeAttr::get(&context, shape));
  auto weight = builder.create<OnesOp>(
      builder.getUnknownLoc(), weightType, /*device=*/nullptr,
      ShapeAttr::get(&context, weightShape));
  auto rmsNorm = builder.create<RMSNormOp>(
      builder.getUnknownLoc(), inputType, input.getResult(),
      weight.getResult(), /*bias=*/mlir::Value(), residual.getResult(),
      builder.getF32FloatAttr(1.0e-6f),
      /*compute_config=*/nullptr);

  RmsNormRuleBook rules;
  EXPECT_TRUE(rules.isValidInputCombination(
      rmsNorm, {inputLayout, weightLayout, inputLayout}));
  EXPECT_FALSE(rules.isValidInputCombination(
      rmsNorm, {inputLayout, weightLayout, residualLayout}));
}

TEST_F(OpRuleBookTest,
       RmsNormPreflightRejectsMismatchedFusedResidualBeforeOpModel) {
  builder.setInsertionPointToStart(&module->getBodyRegion().front());
  llvm::SmallVector<int64_t> shape = {1, 1024, 16, 128};
  llvm::SmallVector<int64_t> weightShape = {128};
  auto inputLayout = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {8, 4});
  auto residualLayout = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {7, 4});
  auto weightLayout = createDRAMInterleavedLayout(weightShape);

  auto inputType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), inputLayout);
  auto residualType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), residualLayout);
  auto weightType = mlir::RankedTensorType::get(
      weightShape, builder.getBF16Type(), weightLayout);
  auto input = builder.create<OnesOp>(
      builder.getUnknownLoc(), inputType, /*device=*/nullptr,
      ShapeAttr::get(&context, shape));
  auto residual = builder.create<OnesOp>(
      builder.getUnknownLoc(), residualType, /*device=*/nullptr,
      ShapeAttr::get(&context, shape));
  auto weight = builder.create<OnesOp>(
      builder.getUnknownLoc(), weightType, /*device=*/nullptr,
      ShapeAttr::get(&context, weightShape));
  auto rmsNorm = builder.create<RMSNormOp>(
      builder.getUnknownLoc(), inputType, input.getResult(),
      weight.getResult(), /*bias=*/mlir::Value(), residual.getResult(),
      builder.getF32FloatAttr(1.0e-6f),
      /*compute_config=*/nullptr);

  ValidationResult result = validateOperation(
      rmsNorm, {inputLayout, weightLayout, residualLayout},
      OpConfig(inputLayout), /*additionalL1Usage=*/0);
  EXPECT_EQ(result.status, ValidationStatus::MetalBackendError);
  EXPECT_EQ(result.errorMessage,
            "rms_norm sharded input and residual require identical physical "
            "layouts");
}

TEST_F(OpRuleBookTest,
       RmsNormPreflightRejectsMismatchedPhysicalShardSpec) {
  builder.setInsertionPointToStart(&module->getBodyRegion().front());
  llvm::SmallVector<int64_t> shape = {1, 1024, 16, 128};
  llvm::SmallVector<int64_t> weightShape = {128};
  auto inputLayout = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {8, 4});
  auto outputLayout = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {7, 4});
  auto weightLayout = createDRAMInterleavedLayout(weightShape);

  auto inputType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), inputLayout);
  auto outputType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), outputLayout);
  auto weightType = mlir::RankedTensorType::get(
      weightShape, builder.getBF16Type(), weightLayout);
  auto input = builder.create<OnesOp>(
      builder.getUnknownLoc(), inputType, /*device=*/nullptr,
      ShapeAttr::get(&context, shape));
  auto weight = builder.create<OnesOp>(
      builder.getUnknownLoc(), weightType, /*device=*/nullptr,
      ShapeAttr::get(&context, weightShape));
  auto rmsNorm = builder.create<RMSNormOp>(
      builder.getUnknownLoc(), outputType, input.getResult(),
      weight.getResult(), /*bias=*/mlir::Value(),
      /*residual_input_tensor=*/mlir::Value(),
      builder.getF32FloatAttr(1.0e-6f),
      /*compute_config=*/nullptr);

  ValidationResult result = validateOperation(
      rmsNorm, {inputLayout, weightLayout}, OpConfig(outputLayout),
      /*additionalL1Usage=*/0);
  EXPECT_EQ(result.status, ValidationStatus::MetalBackendError);
  EXPECT_EQ(result.errorMessage,
            "rms_norm sharded input and output require identical physical "
            "layouts");
}

TEST_F(OpRuleBookTest, SDPAAndPagedFillRequireInterleavedInputs) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto dramInterleaved = createDRAMInterleavedLayout(shape);
  auto l1Interleaved = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::Interleaved);
  auto widthSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  auto heightSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto blockSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});

  SDPAInterleavedRuleBook sdpaRules;
  for (unsigned operandIdx = 0; operandIdx < 3; ++operandIdx) {
    auto filter = sdpaRules.getInputLayoutFilter(operandIdx);
    ASSERT_TRUE(filter);
    EXPECT_TRUE(filter(dramInterleaved));
    EXPECT_TRUE(filter(l1Interleaved));
    EXPECT_FALSE(filter(widthSharded));
    EXPECT_FALSE(filter(heightSharded));
    EXPECT_FALSE(filter(blockSharded));
  }

  // The shared base remains unfiltered for NLPConcatHeadsDecodeOp.
  SDPARuleBook nlpRules;
  EXPECT_FALSE(nlpRules.getInputLayoutFilter(0));

  PagedFillCacheRuleBook pagedFillRules;
  auto inputFilter = pagedFillRules.getInputLayoutFilter(1);
  ASSERT_TRUE(inputFilter);
  EXPECT_TRUE(inputFilter(dramInterleaved));
  EXPECT_TRUE(inputFilter(l1Interleaved));
  EXPECT_FALSE(inputFilter(widthSharded));
  EXPECT_FALSE(inputFilter(heightSharded));
  EXPECT_FALSE(inputFilter(blockSharded));
}

TEST_F(OpRuleBookTest, MatmulMcast2DFiltersIncompatibleShardedInputs) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto blockInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});
  auto blockOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});
  auto heightInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto heightOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto widthInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  auto blockShard = blockInput.getShardShape();
  auto heightShard = heightInput.getShardShape();
  ASSERT_EQ(blockShard.size(), 2u);
  ASSERT_EQ(heightShard.size(), 2u);
  MatmulRuleBook rules;

  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, /*in0BlockW=*/1, blockShard[0]),
      {blockInput}));
  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, heightShard[1], heightShard[0]),
      {heightInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, /*in0BlockW=*/1, blockShard[0]),
      {widthInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, /*in0BlockW=*/1, blockShard[0] + 1),
      {blockInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, blockShard[1] + 1, blockShard[0]),
      {blockInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast2DHint(heightOutput, /*in0BlockW=*/1, blockShard[0]),
      {blockInput}));
}

TEST_F(OpRuleBookTest, MatmulMcast1DEnforcesDirectionAndShardGeometry) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto widthInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  auto widthOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  auto heightInput = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::HeightSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/0, /*endY=*/7);
  auto heightOutput = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::HeightSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/0, /*endY=*/7);
  auto blockInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});
  auto blockOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {1, 8});
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto widthShard = widthInput.getShardShape();
  auto heightShard = heightInput.getShardShape();
  ASSERT_EQ(widthShard.size(), 2u);
  ASSERT_EQ(heightShard.size(), 2u);
  MatmulRuleBook rules;

  auto widthHint =
      createMcast1DHint(widthOutput, /*in0BlockW=*/1, widthShard[0],
                        /*mcastIn0=*/true);
  auto heightHint =
      createMcast1DHint(heightOutput, /*in0BlockW=*/1, heightShard[0],
                        /*mcastIn0=*/false);
  EXPECT_TRUE(
      rules.isValidOutputHintForInputs(widthHint, {widthInput}));
  EXPECT_FALSE(getMatmulPreflightError({heightInput}, heightHint));
  EXPECT_TRUE(
      rules.isValidOutputHintForInputs(heightHint, {heightInput}));
  EXPECT_TRUE(
      rules.isValidOutputHintForInputs(widthHint, {interleaved}));
  EXPECT_FALSE(
      rules.isValidOutputHintForInputs(widthHint, {heightInput}));
  EXPECT_FALSE(
      rules.isValidOutputHintForInputs(heightHint, {widthInput}));
  EXPECT_FALSE(
      rules.isValidOutputHintForInputs(widthHint, {blockInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast1DHint(widthOutput, /*in0BlockW=*/1, widthShard[0] + 1,
                        /*mcastIn0=*/true),
      {widthInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast1DHint(widthOutput, widthShard[1] + 1, widthShard[0],
                        /*mcastIn0=*/true),
      {widthInput}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast1DHint(blockOutput, /*in0BlockW=*/1, widthShard[0],
                        /*mcastIn0=*/true),
      {widthInput}));
}

TEST_F(OpRuleBookTest, MatmulShardedOutputCBMustFitTensorShard) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto widthOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  auto blockOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});
  MatmulRuleBook rules;

  uint64_t widthCapacity =
      widthOutput.getShardSizeInBytes() / widthOutput.getElementSizeBytes();
  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      createMcast1DHint(widthOutput, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true, widthCapacity),
      {interleaved}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast1DHint(widthOutput, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true, widthCapacity + 1),
      {interleaved}));

  uint64_t blockCapacity =
      blockOutput.getShardSizeInBytes() / blockOutput.getElementSizeBytes();
  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, /*in0BlockW=*/1, /*perCoreM=*/1,
                        blockCapacity),
      {interleaved}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast2DHint(blockOutput, /*in0BlockW=*/1, /*perCoreM=*/1,
                        blockCapacity + 1),
      {interleaved}));
}

TEST_F(OpRuleBookTest, MatmulPreflightRejectsTensorBackedCBCapacity) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  llvm::SmallVector<int64_t> largeOutputShape = {1, 64, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto output = createTiledLayout(largeOutputShape, BufferType::L1,
                                  TensorMemoryLayout::WidthSharded, {1, 8});
  auto shardedInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  uint64_t outputCapacity =
      output.getShardSizeInBytes() / output.getElementSizeBytes();

  EXPECT_FALSE(getMatmulPreflightError(
      {interleaved, interleaved},
      createMcast1DHint(output, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true, outputCapacity)));
  EXPECT_TRUE(getMatmulPreflightError(
                  {interleaved, interleaved},
                  createMcast1DHint(output, /*in0BlockW=*/1, /*perCoreM=*/1,
                                    /*mcastIn0=*/true, outputCapacity + 1))
                  .value()
                  .find("output") != std::string::npos);

  EXPECT_TRUE(getMatmulPreflightError(
                  {shardedInput, interleaved},
                  createMcast1DHint(output, /*in0BlockW=*/1, /*perCoreM=*/2,
                                    /*mcastIn0=*/true, /*perCoreN=*/1))
                  .value()
                  .find("input 0") != std::string::npos);
  EXPECT_TRUE(getMatmulPreflightError(
                  {interleaved, shardedInput},
                  createMcast1DHint(output, /*in0BlockW=*/1, /*perCoreM=*/1,
                                    /*mcastIn0=*/true, /*perCoreN=*/9))
                  .value()
                  .find("input 1") != std::string::npos);
}

TEST_F(OpRuleBookTest, MatmulMcast1DBlockOutputRequiresPhysicalRowOrColumn) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto singleRow = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {1, 8},
      /*startX=*/0, /*startY=*/0, /*endX=*/7, /*endY=*/0);
  auto multiRow = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {1, 8},
      /*startX=*/0, /*startY=*/0, /*endX=*/3, /*endY=*/1);
  auto singleColumn = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/0, /*endY=*/7);
  auto multiColumn = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/1, /*endY=*/3);
  MatmulRuleBook rules;

  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      createMcast1DHint(singleRow, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true),
      {interleaved}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast1DHint(multiRow, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true),
      {interleaved}));
  EXPECT_TRUE(rules.isValidOutputHintForInputs(
      createMcast1DHint(singleColumn, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/false),
      {interleaved}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(
      createMcast1DHint(multiColumn, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/false),
      {interleaved}));
}

TEST_F(OpRuleBookTest, MatmulPreflightRejectsInvalidPhysical1DBlockGrid) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto singleRow = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {1, 8},
      /*startX=*/0, /*startY=*/0, /*endX=*/7, /*endY=*/0);
  auto multiRow = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {1, 8},
      /*startX=*/0, /*startY=*/0, /*endX=*/3, /*endY=*/1);
  auto singleColumn = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/0, /*endY=*/7);
  auto multiColumn = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/1, /*endY=*/3);

  EXPECT_FALSE(getMatmulPreflightError(
      {interleaved},
      createMcast1DHint(singleRow, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true)));
  EXPECT_TRUE(getMatmulPreflightError(
                  {interleaved},
                  createMcast1DHint(multiRow, /*in0BlockW=*/1, /*perCoreM=*/1,
                                    /*mcastIn0=*/true))
                  .value()
                  .find("physical row") != std::string::npos);
  EXPECT_FALSE(getMatmulPreflightError(
      {interleaved},
      createMcast1DHint(singleColumn, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/false)));
  EXPECT_TRUE(getMatmulPreflightError(
                  {interleaved}, createMcast1DHint(multiColumn, /*in0BlockW=*/1,
                                                   /*perCoreM=*/1,
                                                   /*mcastIn0=*/false))
                  .value()
                  .find("physical column") != std::string::npos);
}

TEST_F(OpRuleBookTest,
       Matmul1DHeightShardedOutputRequiresPhysicalColumn) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto physicalColumn = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::HeightSharded, {8, 1},
      /*startX=*/0, /*startY=*/0, /*endX=*/0, /*endY=*/7);
  // This is the geometry produced by the Qwen3 failing candidate: a
  // height-sharded 1D-column config spread over a 2D 11x3 core rectangle.
  auto rectangularGrid = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::HeightSharded, {3, 11},
      /*startX=*/0, /*startY=*/0, /*endX=*/10, /*endY=*/2);
  MatmulRuleBook rules;

  auto valid = createMcast1DHint(physicalColumn, /*in0BlockW=*/1,
                                 /*perCoreM=*/1, /*mcastIn0=*/false);
  auto invalid = createMcast1DHint(rectangularGrid, /*in0BlockW=*/1,
                                   /*perCoreM=*/1, /*mcastIn0=*/false);

  EXPECT_TRUE(rules.isValidOutputHintForInputs(valid, {interleaved}));
  EXPECT_FALSE(rules.isValidOutputHintForInputs(invalid, {interleaved}));
  EXPECT_FALSE(getMatmulPreflightError({interleaved}, valid));
  EXPECT_TRUE(getMatmulPreflightError({interleaved}, invalid)
                  .value()
                  .find("physical column") != std::string::npos);
}

TEST_F(OpRuleBookTest,
       MatmulExplicitConfigRejectsIgnoredPhysicalShardedOutput) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto physicalOutput = createTiledLayoutWithCoreRange(
      shape, TensorMemoryLayout::BlockSharded, {1, 8},
      /*startX=*/0, /*startY=*/0, /*endX=*/7, /*endY=*/0);
  auto ignoredOutput = physicalOutput.withIgnorePhysicalLayout(true);
  auto shardShape = physicalOutput.getShardShape();
  ASSERT_EQ(shardShape.size(), 2u);
  MatmulRuleBook rules;

  auto physicalConfig =
      createMcast2DHint(physicalOutput, /*in0BlockW=*/1, shardShape[0],
                        shardShape[1]);
  auto ignoredConfig =
      createMcast2DHint(ignoredOutput, /*in0BlockW=*/1, shardShape[0],
                        shardShape[1]);

  EXPECT_TRUE(
      rules.isValidOutputHintForInputs(physicalConfig, {interleaved}));
  EXPECT_FALSE(
      rules.isValidOutputHintForInputs(ignoredConfig, {interleaved}));
  EXPECT_FALSE(getMatmulPreflightError({interleaved}, physicalConfig));
  EXPECT_TRUE(getMatmulPreflightError({interleaved}, ignoredConfig)
                  .value()
                  .find("requires a physical") != std::string::npos);
}

TEST_F(OpRuleBookTest, MatmulExplicitConfigRejectsRowMajorShardedOutput) {
  llvm::SmallVector<int64_t> shape = {1024, 2560};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
  auto rowMajorOutput =
      TTNNLayoutAttr::Builder(&context, shape, builder.getBF16Type())
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayout::HeightSharded)
          .setGridShape({5, 1})
          .buildWithCanonicalCorePlacement(deviceAttr);
  auto config =
      createMcast2DHint(rowMajorOutput, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*perCoreN=*/1);

  auto error = getMatmulPreflightError({interleaved}, config);
  ASSERT_TRUE(error);
  EXPECT_NE(error->find("requires a tiled layout"), std::string::npos);
}

TEST_F(OpRuleBookTest, MatmulAutoPickerRejectsShardedOutput) {
  llvm::SmallVector<int64_t> shape = {1024, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto shardedOutput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  OpConfig config(shardedOutput);

  auto error = getMatmulPreflightError({interleaved}, config);
  ASSERT_TRUE(error);
  EXPECT_NE(error->find("requires an explicit program config"),
            std::string::npos);
}

TEST_F(OpRuleBookTest, MatmulAutoPickerRejectsShardedInput) {
  llvm::SmallVector<int64_t> shape = {1024, 2048};
  auto shardedInput = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto interleavedOutput = createDRAMInterleavedLayout(shape);
  OpConfig config(interleavedOutput);

  auto error = getMatmulPreflightError({shardedInput}, config);
  ASSERT_TRUE(error);
  EXPECT_NE(error->find("requires an explicit program config"),
            std::string::npos);
}

TEST_F(OpRuleBookTest, MatmulRejectsRowMajorShardedInput) {
  llvm::SmallVector<int64_t> shape = {1024, 2048};
  auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
  auto rowMajorInput =
      TTNNLayoutAttr::Builder(&context, shape, builder.getBF16Type())
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayout::HeightSharded)
          .setGridShape({8, 1})
          .buildWithCanonicalCorePlacement(deviceAttr);
  auto output = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto config =
      createMcast2DHint(output, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*perCoreN=*/1);

  auto error = getMatmulPreflightError({rowMajorInput}, config);
  ASSERT_TRUE(error);
  EXPECT_NE(error->find("requires a tiled layout"), std::string::npos);
}

TEST_F(OpRuleBookTest, MatmulPartialConfigDedupPreservesOutputGeometry) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto interleaved = createDRAMInterleavedLayout(shape);
  auto output8 = createTiledLayout(shape, BufferType::L1,
                                   TensorMemoryLayout::WidthSharded, {1, 8});
  auto output4 = createTiledLayout(shape, BufferType::L1,
                                   TensorMemoryLayout::WidthSharded, {1, 4});
  uint64_t capacity8 =
      output8.getShardSizeInBytes() / output8.getElementSizeBytes();
  uint64_t capacity4 =
      output4.getShardSizeInBytes() / output4.getElementSizeBytes();

  std::vector<OpConfig> configs{
      createMcast1DHint(output8, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true, capacity8),
      createMcast1DHint(output4, /*in0BlockW=*/1, /*perCoreM=*/1,
                        /*mcastIn0=*/true, capacity4)};
  auto partials = optimizer_utils::getUniqueTestConfigsForMatmulLinear(configs);

  ASSERT_EQ(partials.size(), 2u);
  MatmulRuleBook rules;
  for (const OpConfig &partial : partials) {
    EXPECT_FALSE(partial.outputLayout.getIgnorePhysicalLayout());
    EXPECT_TRUE(rules.isValidOutputHintForInputs(partial, {interleaved}));
  }
}

TEST_F(OpRuleBookTest, EmbeddingRejectsShardedIndicesAndWeights) {
  llvm::SmallVector<int64_t> shape = {1, 32, 2048};
  auto dramInterleaved = createDRAMInterleavedLayout(shape);
  auto l1Interleaved =
      createTiledLayout(shape, BufferType::L1, TensorMemoryLayout::Interleaved);
  auto widthSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::WidthSharded, {1, 8});
  auto heightSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::HeightSharded, {8, 1});
  auto blockSharded = createTiledLayout(
      shape, BufferType::L1, TensorMemoryLayout::BlockSharded, {2, 4});
  EmbeddingRuleBook rules;

  for (unsigned operandIdx = 0; operandIdx < 2; ++operandIdx) {
    auto filter = rules.getInputLayoutFilter(operandIdx);
    ASSERT_TRUE(filter);
    EXPECT_TRUE(filter(dramInterleaved));
    EXPECT_TRUE(filter(l1Interleaved));
    EXPECT_FALSE(filter(widthSharded));
    EXPECT_FALSE(filter(heightSharded));
    EXPECT_FALSE(filter(blockSharded));
  }
  EXPECT_FALSE(rules.getInputLayoutFilter(2));
}

//===----------------------------------------------------------------------===//
// shouldExploreReshards tests
//===----------------------------------------------------------------------===//

TEST_F(OpModelStrategyTest, ShouldExploreReshardsElementwiseTrue) {
  auto addOp = createMockAddOp();
  EXPECT_TRUE(shouldExploreReshards(addOp));
}

TEST_F(OpModelStrategyTest, ShouldExploreReshardsReshapeFalse) {
  auto reshapeOp = createMockReshapeOp();
  EXPECT_FALSE(shouldExploreReshards(reshapeOp));
}

TEST_F(OpModelStrategyTest, ShouldExploreReshardsMatmulTrue) {
  auto matmulOp = createMockMatmulOp();
  EXPECT_TRUE(shouldExploreReshards(matmulOp));
}

//===----------------------------------------------------------------------===//
// LayoutScore comparison tests
//===----------------------------------------------------------------------===//

TEST_F(OpModelStrategyTest, ScoreOrderingL1ShardedBest) {
  LayoutScore l1Sharded{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                        /*requiresReshard=*/false, /*outputL1Usage=*/1024};
  LayoutScore l1Interleaved{/*coreCount=*/1, /*isSharded=*/false,
                            /*isL1=*/true, /*requiresReshard=*/false,
                            /*outputL1Usage=*/1024};
  LayoutScore dram{/*coreCount=*/1, /*isSharded=*/false, /*isL1=*/false,
                   /*requiresReshard=*/false, /*outputL1Usage=*/0};

  EXPECT_TRUE(l1Sharded > l1Interleaved);
  EXPECT_TRUE(l1Interleaved > dram);
  EXPECT_TRUE(l1Sharded > dram);
}

TEST_F(OpModelStrategyTest, ScoreOrderingMoreCoresBetter) {
  LayoutScore moreCores{/*coreCount=*/64, /*isSharded=*/true, /*isL1=*/true,
                        /*requiresReshard=*/false, /*outputL1Usage=*/512};
  LayoutScore fewerCores{/*coreCount=*/8, /*isSharded=*/true, /*isL1=*/true,
                         /*requiresReshard=*/false, /*outputL1Usage=*/512};

  EXPECT_TRUE(moreCores > fewerCores);
}

TEST_F(OpModelStrategyTest, ScoreOrderingReshardPenalized) {
  LayoutScore noReshard{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                        /*requiresReshard=*/false, /*outputL1Usage=*/1024};
  LayoutScore withReshard{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                          /*requiresReshard=*/true, /*outputL1Usage=*/1024};

  EXPECT_TRUE(noReshard > withReshard);
}

TEST_F(OpModelStrategyTest, ScoreOrderingLowerL1UsageBetter) {
  LayoutScore lowUsage{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                       /*requiresReshard=*/false, /*outputL1Usage=*/512};
  LayoutScore highUsage{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                        /*requiresReshard=*/false, /*outputL1Usage=*/4096};

  EXPECT_TRUE(lowUsage > highUsage);
}

TEST_F(OpModelStrategyTest, ScoreOrderingEquality) {
  LayoutScore a{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                /*requiresReshard=*/false, /*outputL1Usage=*/1024};
  LayoutScore b{/*coreCount=*/32, /*isSharded=*/true, /*isL1=*/true,
                /*requiresReshard=*/false, /*outputL1Usage=*/1024};

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a < b);
}

//===----------------------------------------------------------------------===//
// scoreCandidate tests
//===----------------------------------------------------------------------===//

TEST_F(OpModelStrategyTest, ScoreCandidateL1ShardedResult) {
  auto addOp = createMockAddOp();
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto l1ShardedLayout = createL1ShardedLayout(shape, {8, 4});
  OpConfig config(l1ShardedLayout);

  op_constraint_validation::ValidationResult result;
  result.status = op_constraint_validation::ValidationStatus::Success;
  result.actualOutputLayouts = {l1ShardedLayout};
  result.outputL1Usage = 1024;

  LayoutScore score = scoreCandidate(addOp, config, result, false);

  EXPECT_TRUE(score.isL1);
  EXPECT_TRUE(score.isSharded);
  EXPECT_FALSE(score.requiresReshard);
  EXPECT_EQ(score.outputL1Usage, 1024u);
  EXPECT_GT(score.coreCount, 0);
}

TEST_F(OpModelStrategyTest, ScoreCandidateDRAMResult) {
  auto addOp = createMockAddOp();
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto dramLayout = createDRAMInterleavedLayout(shape);
  OpConfig config(dramLayout);

  op_constraint_validation::ValidationResult result;
  result.status = op_constraint_validation::ValidationStatus::Success;
  result.actualOutputLayouts = {dramLayout};
  result.outputL1Usage = 0;

  LayoutScore score = scoreCandidate(addOp, config, result, false);

  EXPECT_FALSE(score.isL1);
  EXPECT_FALSE(score.isSharded);
}

TEST_F(OpModelStrategyTest, ScoreCandidateWithReshard) {
  auto addOp = createMockAddOp();
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto layout = createL1InterleavedLayout(shape);
  OpConfig config(layout);

  op_constraint_validation::ValidationResult result;
  result.status = op_constraint_validation::ValidationStatus::Success;
  result.actualOutputLayouts = {layout};
  result.outputL1Usage = 512;

  LayoutScore withReshard = scoreCandidate(addOp, config, result, true);
  LayoutScore withoutReshard = scoreCandidate(addOp, config, result, false);

  EXPECT_TRUE(withReshard.requiresReshard);
  EXPECT_FALSE(withoutReshard.requiresReshard);
  EXPECT_TRUE(withoutReshard > withReshard);
}

//===----------------------------------------------------------------------===//
// End-to-end: validate + score tests
//===----------------------------------------------------------------------===//

TEST_F(OpModelStrategyTest, EndToEndAddOpValidateAndScore) {
  auto addOp = createMockAddOp();
  auto layouts = ttnn::utils::extractInputLayouts(addOp);
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};

  // Test with DRAM config.
  OpConfig dramConfig(createDRAMInterleavedLayout(shape));
  auto dramResult =
      op_constraint_validation::validateOperation(addOp, layouts, dramConfig);

  // Test with L1-interleaved config.
  OpConfig l1Config(createL1InterleavedLayout(shape));
  auto l1Result =
      op_constraint_validation::validateOperation(addOp, layouts, l1Config);

  // If both succeed, L1 should score higher than DRAM.
  if (dramResult.isSuccess() && l1Result.isSuccess()) {
    LayoutScore dramScore =
        scoreCandidate(addOp, dramConfig, dramResult, false);
    LayoutScore l1Score = scoreCandidate(addOp, l1Config, l1Result, false);

    EXPECT_TRUE(l1Score > dramScore);
  }
}
