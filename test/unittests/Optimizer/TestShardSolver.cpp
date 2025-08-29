// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;

constexpr int TensorDimX = 128;
constexpr int TensorDimY = 128;

class ShardSolverBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;

  void SetUp() override {
    context.loadDialect<TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
    createFuncOp();
  }

  llvm::SmallVector<int64_t, 2> getTensorShape() {
    return {TensorDimX, TensorDimY};
  }

  mlir::RankedTensorType getTensorRankedType() {
    return mlir::RankedTensorType::get(
        getTensorShape(), builder.getF32Type(),
        TTNNLayoutAttr::get(&context, getTensorShape(), builder.getF32Type(),
                            BufferType::DRAM,
                            mlir::tt::ttcore::GridAttr::get(&context, {1, 1}),
                            mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                                &context, TensorMemoryLayout::Interleaved)));
  }

  mlir::Value createEmptyTensor() {
    ShapeAttr shapeAttr = ShapeAttr::get(&context, getTensorShape());
    return builder.create<OnesOp>(builder.getUnknownLoc(),
                                  getTensorRankedType(), nullptr, shapeAttr,
                                  nullptr, nullptr, nullptr);
  }

  mlir::func::FuncOp createFuncOp() {
    mlir::SmallVector<mlir::Type> input;
    input.push_back(getTensorRankedType());

    mlir::SmallVector<mlir::Type> output;
    output.push_back(getTensorRankedType());

    auto funcType = builder.getType<mlir::FunctionType>(
        mlir::TypeRange(input), mlir::TypeRange(output));
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test",
                                              funcType);
    mlir::Block *block = func.addEntryBlock();
    block->addArgument(getTensorRankedType(), builder.getUnknownLoc());
    block->addArgument(getTensorRankedType(), builder.getUnknownLoc());

    builder.setInsertionPointToStart(block);

    return func;
  }

  void
  prepareOpForShardSolver(mlir::Operation *op,
                          std::vector<OpL1MemSpec> &opL1MemSpecs,
                          llvm::DenseSet<mlir::Operation *> &l1ChainedOps) {
    OpL1MemSpec opL1MemSpec;
    opL1MemSpec.op = op;
    opL1MemSpecs.push_back(opL1MemSpec);
    l1ChainedOps.insert(op);
  }

  void addConfigForOp(
      mlir::Operation *op,
      llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> &legalConfigs,
      BufferType memorySpace, TensorMemoryLayout tensorMemoryLayout,
      int gridWidth, int gridHeight) {
    if (legalConfigs.find(op) == legalConfigs.end()) {
      legalConfigs[op] = std::vector<OpConfig>{TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(), builder.getF32Type(),
          memorySpace,
          mlir::tt::ttcore::GridAttr::get(&context, {gridWidth, gridHeight}),
          mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                      tensorMemoryLayout))};
    } else {
      legalConfigs[op].push_back(TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(), builder.getF32Type(),
          memorySpace,
          mlir::tt::ttcore::GridAttr::get(&context, {gridWidth, gridHeight}),
          mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                      tensorMemoryLayout)));
    }
  }

  void TearDown() override {}
};

// Validate that ShardSolver can produce correct max core usage for a shard
// chain, total accumulated in the first op.
//
//    Op0 ----- (4, 8, 4)
//     |
//    Op1 ----- (8, 4, 4)
//    / \
//   /   \
//  Op2  Op3 -- (4, 4, 1) (4, 4, 1)
//   \   /
//    \ /
//    Op4 ----- (2, 1, 1)
//     |
//    Op5 ----- (2, 1, 1)
//
// Verification target:
//
//    Op0 ----- (24, 22, 12)
//     |
//    Op1 ----- (20, 14, 8)
//    / \
//   /   \
//  Op2  Op3 -- (6, 5, 3) (6, 5, 3)
//   \   /
//    \ /
//    Op4 ----- (4, 2, 2)
//     |
//    Op5 ----- (2, 1, 1)
//
TEST_F(ShardSolverBase, VerifyProduceMaxCoreUsage) {
  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  std::vector<OpL1MemSpec> opL1MemSpecs;
  llvm::DenseSet<mlir::Operation *> l1ChainedOps;
  constexpr unsigned usableL1CacheSize = 1024 * 1024;
  llvm::DenseSet<Edge> overrideReshardEdges;
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout;

  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *op =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  mlir::Operation *firstOp = op;

  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 4);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 8, 1);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 2, 2);

  rhs = op->getResult(0);
  op = builder.create<ReluOp>(builder.getUnknownLoc(), rhs.getType(), rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 8);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 4, 1);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 2, 2);

  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = op->getResult(0);

  op = builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 4);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 4, 1);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  op = builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 4);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 4, 1);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  lhs = opL1MemSpecs[opL1MemSpecs.size() - 2].op->getResult(0);
  rhs = opL1MemSpecs[opL1MemSpecs.size() - 1].op->getResult(0);
  op = builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 2);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 1, 1);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  rhs = op->getResult(0);
  op = builder.create<ReluOp>(builder.getUnknownLoc(), rhs.getType(), rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 2);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 1, 1);
  addConfigForOp(op, legalConfigs, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  // Create custom checkShardCompatible function.
  //
  std::function<llvm::Expected<TTNNLayoutAttr>(
      mlir::Value, const TTNNLayoutAttr &, mlir::Operation *, const OpConfig &)>
      checkShardCompatible =
          [&legalConfigs](
              mlir::Value producerOperand, const TTNNLayoutAttr &producerLayout,
              mlir::Operation *consumerOp, const OpConfig &consumerConfig)
      -> llvm::Expected<TTNNLayoutAttr> {
    // Interleaved to sharded is always supported.
    //
    if (producerLayout.hasInterleavedDRAMTensorMemoryLayout()) {
      return consumerConfig.outputLayout;
    }

    if (!consumerConfig.outputLayout) {
      // ShardSolver invokes this function with consumerConfig.outputLayout
      // being null, so we need to find the correct config among the
      // consumer legal configs. To do this, we will match the producer
      // layout with the consumer legal configs.
      auto *producerOp = producerOperand.getDefiningOp();
      assert(producerOp && "Producer op not found");
      // find which order is producerLaoyut among producer legal configs
      auto &producerConfigs = legalConfigs[producerOp];
      auto producerLayoutIndex =
          std::find(producerConfigs.begin(), producerConfigs.end(),
                    OpConfig(producerLayout));
      assert(producerLayoutIndex != producerConfigs.end() &&
             "Producer layout not found");

      return legalConfigs[consumerOp]
                         [producerLayoutIndex - producerConfigs.begin()]
                             .outputLayout;
    }

    // Simple shard compat assumption. Try to keep same shard layout.
    //
    if (producerLayout.getMemLayout() !=
        consumerConfig.outputLayout.getMemLayout()) {
      return llvm::createStringError("Output layout does not match");
    }

    return consumerConfig.outputLayout;
  };

  // tensorPossibleLayouts can be null since we expect ShardSolver won't need
  // them because custom checkShardCompatible function will handle all the
  // checks.
  ShardSolver shardSolver(/*tensorTypePossibleLayouts=*/nullptr, legalConfigs,
                          opL1MemSpecs, l1ChainedOps, usableL1CacheSize,
                          overrideReshardEdges, overrideOutputLayout,
                          checkShardCompatible);

  ASSERT_TRUE(shardSolver.resolve());

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<float, 64>>
      accMaxCoreUsage = shardSolver.produceMaxCoreUsage();

  ASSERT_EQ(accMaxCoreUsage[firstOp][0], 24);
  ASSERT_EQ(accMaxCoreUsage[firstOp][1], 22);
  ASSERT_EQ(accMaxCoreUsage[firstOp][2], 12);

  // Set layouts for all ops in ShardSolver and validate that their total core
  // usage matches the expected values. Picking legal layout at index 0 for all
  // ops should lead to accMaxCoreUsage[firstOp][0] total core usage.
  //
  for (auto &opL1MemSpec : opL1MemSpecs) {
    ShardSolver::RemainingConfigAttrs validLayouts =
        shardSolver.at(opL1MemSpec.op);
    const OpConfig *selectedConfig = validLayouts.begin().get();
    shardSolver.set(opL1MemSpec.op, *selectedConfig);
  }

  llvm::DenseMap<mlir::Operation *, OpConfig> selectedOpConfig =
      shardSolver.finish().selectedOpConfig;
  float totalCoreUsage = 0;
  for (const auto &opLayout : selectedOpConfig) {
    totalCoreUsage += opLayout.second.outputLayout.getGrid().getGridVolume();
  }

  ASSERT_EQ(totalCoreUsage, accMaxCoreUsage[firstOp][0]);
}
