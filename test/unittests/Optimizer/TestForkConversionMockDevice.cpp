// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "MockDeviceFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include "gtest/gtest.h"

namespace mlir::tt::ttnn {
namespace {

class ForkConversionMockDeviceTest : public testing::Test {
protected:
  MLIRContext context;
  OpBuilder builder{&context};
  OwningOpRef<ModuleOp> module;
  func::FuncOp func;

  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect, TTNNDialect,
                        func::FuncDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());
    mlir::tt::ttcore::registerDevice(module.get());
    op_model::SingletonDeviceContext::setSystemDesc(
        mlir::tt::ttcore::getCurrentScopeSystemDesc(module.get()));
    op_model::SingletonDeviceContext::getInstance().reshapeMeshDevice({1, 1});
    module->getOperation()->setAttr(
        mlir::tt::ttnn::utils::g_TensorL1UsageCapAttrName,
        builder.getF32FloatAttr(1.0));
    ASSERT_TRUE(op_model::isMockDevice());
  }

  TTNNLayoutAttr makeLayout(ArrayRef<int64_t> shape, BufferType buffer,
                            TensorMemoryLayout memoryLayout,
                            ArrayRef<int64_t> grid = {1, 1}) {
    return TTNNLayoutAttr::Builder(
               &context, shape,
               mlir::tt::ttcore::TileType::get(builder.getBF16Type()))
        .setBufferType(buffer)
        .setMemoryLayout(memoryLayout)
        .setGridShape(grid)
        .buildWithCanonicalCorePlacement(
            mlir::tt::ttcore::lookupDevice(module.get()));
  }

  RankedTensorType beginFunc(ArrayRef<int64_t> shape) {
    auto type = RankedTensorType::get(
        shape, builder.getBF16Type(),
        makeLayout(shape, BufferType::L1, TensorMemoryLayout::Interleaved,
                   {8, 8}));
    func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "fork",
        builder.getFunctionType({type, type}, {type}));
    builder.setInsertionPointToStart(func.addEntryBlock());
    return type;
  }

  std::vector<OpConfig> makeConfigs(ArrayRef<int64_t> shape) {
    return {OpConfig(makeLayout(shape, BufferType::DRAM,
                                TensorMemoryLayout::Interleaved)),
            OpConfig(makeLayout(shape, BufferType::L1,
                                TensorMemoryLayout::Interleaved, {8, 8})),
            OpConfig(makeLayout(shape, BufferType::L1,
                                TensorMemoryLayout::HeightSharded, {8, 1}))};
  }

  void checkRepeatedUses(bool sharedConsumer);
};

void ForkConversionMockDeviceTest::checkRepeatedUses(bool sharedConsumer) {
  SmallVector<int64_t> shape{1, 1, 512, 512};
  auto type = beginFunc(shape);
  Value left = func.getBody().front().getArgument(0);
  Value right = func.getBody().front().getArgument(1);
  auto producer =
      builder.create<AddOp>(builder.getUnknownLoc(), type, left, right);
  auto matmul = builder.create<MatmulOp>(
      builder.getUnknownLoc(), type, producer.getResult(), producer.getResult(),
      /*transpose_a=*/false, /*transpose_b=*/false,
      /*matmul_program_config=*/nullptr, /*activation=*/nullptr);
  llvm::DenseMap<Operation *, std::vector<OpConfig>> configs;
  configs[producer] = makeConfigs(shape);
  configs[matmul] = makeConfigs(shape);
  MatmulOp otherMatmul;
  Value output = matmul.getResult();
  if (sharedConsumer) {
    otherMatmul = builder.create<MatmulOp>(
        builder.getUnknownLoc(), type, producer.getResult(),
        producer.getResult(),
        /*transpose_a=*/false, /*transpose_b=*/false,
        /*matmul_program_config=*/nullptr, /*activation=*/nullptr);
    auto join =
        builder.create<AddOp>(builder.getUnknownLoc(), type, matmul.getResult(),
                              otherMatmul.getResult());
    configs[otherMatmul] = makeConfigs(shape);
    configs[join] = makeConfigs(shape);
    output = join.getResult();
  }
  auto result = builder.create<ReluOp>(builder.getUnknownLoc(), type, output);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), result.getResult());
  configs[result] = makeConfigs(shape);
  TensorTypeLayoutsMap possibleLayouts;
  possibleLayouts[type][builder.getBF16Type()][getPageLayoutIndex(Layout::Tile)]
                 [getMemoryLayoutIndex(TensorMemoryLayout::HeightSharded)] = {
                     makeLayout(shape, BufferType::L1,
                                TensorMemoryLayout::HeightSharded, {8, 1})};
  MemoryLayoutPropagation propagation(func, configs, &possibleLayouts);
  propagation.run();

  const auto &beam = propagation.getBeamState();
  ASSERT_TRUE(beam.count(producer));
  ASSERT_GT(beam.find(producer)->second.size(), 1u);
  ASSERT_TRUE(beam.count(matmul));
  ASSERT_TRUE(propagation.getFinalChoice().count(matmul));
  const BeamCandidate &chosen =
      beam.find(matmul)->second[propagation.getFinalChoice().lookup(matmul)];
  ASSERT_EQ(chosen.inputLayouts.size(), 2u);
  ASSERT_EQ(chosen.producerCandidateIndices.size(), 2u);
  ASSERT_NE(chosen.inputLayouts[0], chosen.inputLayouts[1]);
  ASSERT_NE(chosen.producerCandidateIndices[0],
            chosen.producerCandidateIndices[1]);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(mlir::tt::ttnn::utils::getLayoutAttrFromTensor(
                  cast<RankedTensorType>(matmul->getOperand(i).getType())),
              chosen.inputLayouts[i]);
  }
  EXPECT_NE(matmul->getOperand(0), matmul->getOperand(1));
  if (sharedConsumer) {
    ASSERT_TRUE(beam.count(otherMatmul));
    ASSERT_TRUE(propagation.getFinalChoice().count(otherMatmul));
    const BeamCandidate &otherChosen =
        beam.find(otherMatmul)
            ->second[propagation.getFinalChoice().lookup(otherMatmul)];
    ASSERT_EQ(otherChosen.inputLayouts, chosen.inputLayouts);
    EXPECT_EQ(matmul->getOperand(0), otherMatmul->getOperand(0));
    EXPECT_EQ(matmul->getOperand(1), otherMatmul->getOperand(1));
  }

  // Independently rank these two distinct input requirements using the actual
  // selected layouts and real backend legality, then check the materialized IR.
  auto footprint = [](TTNNLayoutAttr layout) {
    uint64_t bytes = layout.getShardSizeInBytes();
    for (int64_t dimension : layout.getGridShape()) {
      bytes *= dimension;
    }
    return bytes;
  };
  auto selectedSource = mlir::tt::ttnn::utils::getLayoutAttrFromTensor(
      cast<RankedTensorType>(producer.getResult().getType()));
  uint64_t materializedBytes = 0;
  for (Value operand : matmul->getOperands()) {
    if (operand == producer.getResult()) {
      continue;
    }
    auto conversion = operand.getDefiningOp<ToMemoryConfigOp>();
    ASSERT_TRUE(conversion);
    ASSERT_EQ(conversion->getOperand(0), producer.getResult());
    materializedBytes +=
        footprint(selectedSource) +
        footprint(mlir::tt::ttnn::utils::getLayoutAttrFromTensor(
            cast<RankedTensorType>(operand.getType())));
  }
  EXPECT_GT(materializedBytes, 0u);
  for (const BeamCandidate &alternative : beam.find(producer)->second) {
    ASSERT_EQ(alternative.outputLayouts.size(), 1u);
    TTNNLayoutAttr source = alternative.outputLayouts[0];
    uint64_t bytes = 0;
    bool legal = true;
    for (TTNNLayoutAttr target : chosen.inputLayouts) {
      if (source == target) {
        continue;
      }
      legal &= op_constraint_validation::validateOperation<ToMemoryConfigOp>(
                   matmul, 0, shape, source, target)
                   .isSuccess();
      bytes += footprint(source) + footprint(target);
    }
    if (legal) {
      EXPECT_LE(materializedBytes, bytes);
    }
  }
  EXPECT_TRUE(succeeded(verify(module.get())));
}

TEST_F(ForkConversionMockDeviceTest, RepeatedMatmulOperandsKeepChosenInputs) {
  checkRepeatedUses(/*sharedConsumer=*/false);
}

TEST_F(ForkConversionMockDeviceTest, RepeatedConsumersShareMaterialization) {
  checkRepeatedUses(/*sharedConsumer=*/true);
}

} // namespace
} // namespace mlir::tt::ttnn

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
