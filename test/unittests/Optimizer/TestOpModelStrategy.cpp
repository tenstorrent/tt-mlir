// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;
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
    return TTNNLayoutAttr::get(
        &context, tensorShape, elementType, bufferType,
        mlir::tt::ttcore::GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
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

    auto input1 = builder.create<OnesOp>(
        builder.getUnknownLoc(), tensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    auto input2 = builder.create<OnesOp>(
        builder.getUnknownLoc(), tensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

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
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    llvm::SmallVector<int32_t> outputShapeI32(outputShape.begin(),
                                              outputShape.end());
    return builder.create<ReshapeOp>(builder.getUnknownLoc(), outputTensorType,
                                     input.getResult(),
                                     builder.getI32ArrayAttr(outputShapeI32),
                                     /*memory_config=*/nullptr);
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

    auto lhs = builder.create<OnesOp>(
        builder.getUnknownLoc(), lhsTensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, lhsShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    auto rhs = builder.create<OnesOp>(
        builder.getUnknownLoc(), rhsTensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, rhsShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    return builder.create<MatmulOp>(builder.getUnknownLoc(), outputTensorType,
                                    lhs.getResult(), rhs.getResult(),
                                    /*transpose_a=*/nullptr,
                                    /*transpose_b=*/nullptr,
                                    /*memory_config=*/nullptr,
                                    /*program_config=*/nullptr,
                                    /*compute_kernel_config=*/nullptr);
  }

  // Create a Conv2dOp for testing
  Conv2dOp
  createMockConv2dOp(
    const llvm::ArrayRef<int64_t> &inputShape = {1, 32, 32, 256},
    const llvm::ArrayRef<int64_t> &weightShape = {128, 256, 3, 3},
    const llvm::ArrayRef<int64_t> &biasShape = {1, 1, 1, 128}) {

    int32_t in_channels   = inputShape[3];
    int32_t out_channels  = weightShape[0];
    int32_t batch_size    = inputShape[0];
    int32_t input_height  = inputShape[1];
    int32_t input_width   = inputShape[2];
    int32_t groups        = 1;
    llvm::SmallVector<int32_t, 2> kernel_size = {
        static_cast<int32_t>(weightShape[2]), 
        static_cast<int32_t>(weightShape[3])};
    llvm::SmallVector<int32_t, 2> stride   = {1, 1};
    llvm::SmallVector<int32_t, 2> padding  = {0, 0};
    llvm::SmallVector<int32_t, 2> dilation = {1, 1};
    auto device = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(),
      MeshShapeAttr::get(&context, 1, 1),
      MeshOffsetAttr::get(&context, 0, 0));


    auto inputLayout = createL1InterleavedLayout(inputShape);
    auto inputTensorType =
        mlir::RankedTensorType::get(inputShape, 
          builder.getBF16Type(), inputLayout);
    auto input = builder.create<OnesOp>(
        builder.getUnknownLoc(), inputTensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, inputShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    auto weightLayout = createL1InterleavedLayout(weightShape);
    auto weightTensorType =
        mlir::RankedTensorType::get(weightShape, 
          builder.getBF16Type(), weightLayout);
    auto weight = builder.create<OnesOp>(
        builder.getUnknownLoc(), weightTensorType,
        /*device=*/nullptr, ShapeAttr::get(&context, weightShape),
        /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

    llvm::SmallVector<int64_t> outputShape = {
      batch_size,
      (input_height + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) 
        / stride[0] + 1,
      (input_width + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) 
        / stride[1] + 1,
      out_channels
    };
    auto outputLayout = createTiledLayout(outputShape, 
                          BufferType::L1,
                          TensorMemoryLayout::HeightSharded);
    auto outputTensorType =
        mlir::RankedTensorType::get(biasShape, 
          builder.getBF16Type(), outputLayout);
    
    return builder.create<Conv2dOp>(builder.getUnknownLoc(), outputTensorType,
                                    input.getResult(), weight.getResult(),
                                    /*bias=*/nullptr,
                                    /*device=*/device,
                                    /*in_channels=*/in_channels,
                                    /*out_channels=*/out_channels,
                                    /*batch_size=*/batch_size,
                                    /*input_height=*/input_height,
                                    /*input_width=*/input_width,
                                    /*kernel_size=*/kernel_size,
                                    /*stride=*/stride,
                                    /*padding=*/padding,
                                    /*dilation=*/dilation,
                                    /*groups=*/groups,
                                    /*dtype=*/nullptr,
                                    /*conv2d_config=*/nullptr,
                                    /*compute_config=*/nullptr,
                                    /*conv2d_slice_config=*/nullptr);
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

TEST_F(OpModelStrategyTest, Conv2dPreferHeightSharded) {
  auto conv2dOp = createMockConv2dOp();
  // Identical scores — same core count, same L1 usage (reproduces the tie
  // from issue #7474: BS 8x8 and HS 64x1 both score 64 cores, 32768 L1).
  LayoutScore tiedScore{/*coreCount=*/64, /*isSharded=*/true, /*isL1=*/true,
                        /*requiresReshard=*/false, /*outputL1Usage=*/32768};
  auto outputShape = conv2dOp.getResult().getType().getShape();

  auto interLayout = createTiledLayout(outputShape, 
                          BufferType::L1,
                          TensorMemoryLayout::Interleaved);
  auto hsLayout = createTiledLayout(outputShape, 
                          BufferType::L1,
                          TensorMemoryLayout::HeightSharded);
  auto wsLayout = createTiledLayout(outputShape, 
                          BufferType::L1,
                          TensorMemoryLayout::WidthSharded);
  auto bsLayout = createTiledLayout(outputShape, 
                          BufferType::L1,
                          TensorMemoryLayout::BlockSharded);
  auto ndLayout = createTiledLayout(outputShape, 
                          BufferType::L1,
                          TensorMemoryLayout::NDSharded);
  
  BeamCandidate interCandidate;
  interCandidate.score = tiedScore;
  interCandidate.outputLayouts = {interLayout};

  BeamCandidate hsCandidate;
  hsCandidate.score = tiedScore;
  hsCandidate.outputLayouts = {hsLayout};

  BeamCandidate wsCandidate;
  wsCandidate.score = tiedScore;
  wsCandidate.outputLayouts = {wsLayout};

  BeamCandidate bsCandidate;
  bsCandidate.score = tiedScore;
  bsCandidate.outputLayouts = {bsLayout};

  BeamCandidate ndCandidate;
  ndCandidate.score = tiedScore;
  ndCandidate.outputLayouts = {ndLayout};
  
  // Function preferCandidate() should prefer HeighSharded over any other.
  // HeighSharded vs Interleaved.
  EXPECT_TRUE(preferCandidate(conv2dOp, hsCandidate, interCandidate));
  EXPECT_FALSE(preferCandidate(conv2dOp, interCandidate, hsCandidate));

  // HeighSharded vs WidthSharded.
  EXPECT_TRUE(preferCandidate(conv2dOp, hsCandidate, wsCandidate));
  EXPECT_FALSE(preferCandidate(conv2dOp, wsCandidate, hsCandidate));

  // HeighSharded vs BlockSharded.
  EXPECT_TRUE(preferCandidate(conv2dOp, hsCandidate, bsCandidate));
  EXPECT_FALSE(preferCandidate(conv2dOp, bsCandidate, hsCandidate));

  // HeighSharded vs NDSharded.
  EXPECT_TRUE(preferCandidate(conv2dOp, hsCandidate, ndCandidate));
  EXPECT_FALSE(preferCandidate(conv2dOp, ndCandidate, hsCandidate));

  // Equal layouts, no preference.
  EXPECT_FALSE(preferCandidate(conv2dOp, hsCandidate, hsCandidate));
  EXPECT_FALSE(preferCandidate(conv2dOp, bsCandidate, bsCandidate));
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
