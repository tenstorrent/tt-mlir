// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

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
                            mlir::tt::GridAttr::get(&context, {1, 1}),
                            mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                                &context, TensorMemoryLayout::Interleaved)));
  }

  mlir::Value createEmptyTensor() {
    ShapeAttr shapeAttr = ShapeAttr::get(&context, getTensorShape());
    return builder.create<OnesOp>(builder.getUnknownLoc(),
                                  getTensorRankedType(), shapeAttr, nullptr,
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
    func->setAttr(
        mlir::tt::DeviceAttr::name,
        mlir::tt::DeviceAttr::get(
            &context, mlir::tt::SystemDescAttr::getDefault(&context)));

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

  void
  addLayoutForOp(mlir::Operation *op,
                 llvm::DenseMap<mlir::Operation *, std::vector<TTNNLayoutAttr>>
                     &legalLayouts,
                 BufferType memorySpace, TensorMemoryLayout tensorMemoryLayout,
                 int gridWidth, int gridHeight) {
    if (legalLayouts.find(op) == legalLayouts.end()) {
      legalLayouts[op] = std::vector<TTNNLayoutAttr>{TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(), builder.getF32Type(),
          memorySpace,
          mlir::tt::GridAttr::get(&context, {gridWidth, gridHeight}),
          mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                      tensorMemoryLayout))};
    } else {
      legalLayouts[op].push_back(TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(), builder.getF32Type(),
          memorySpace,
          mlir::tt::GridAttr::get(&context, {gridWidth, gridHeight}),
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
  llvm::DenseMap<mlir::Operation *, std::vector<TTNNLayoutAttr>> legalLayouts;
  std::vector<OpL1MemSpec> opL1MemSpecs;
  llvm::DenseSet<mlir::Operation *> l1ChainedOps;
  constexpr unsigned usableL1CacheSize = 1024 * 1024;
  std::unordered_set<Edge> overrideReshardEdges;

  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *op =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, lhs.getType());
  mlir::Operation *firstOp = op;

  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 4);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 8, 1);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 2, 2);

  rhs = op->getResult(0);
  op = builder.create<ReluOp>(builder.getUnknownLoc(), rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 8);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 4, 1);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 2, 2);

  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = op->getResult(0);

  op = builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, lhs.getType());
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 4);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 4, 1);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  op = builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, lhs.getType());
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 4);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 4, 1);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  lhs = opL1MemSpecs[opL1MemSpecs.size() - 2].op->getResult(0);
  rhs = opL1MemSpecs[opL1MemSpecs.size() - 1].op->getResult(0);
  op = builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, lhs.getType());
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 2);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 1, 1);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  rhs = op->getResult(0);
  op = builder.create<ReluOp>(builder.getUnknownLoc(), rhs);
  prepareOpForShardSolver(op, opL1MemSpecs, l1ChainedOps);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::WidthSharded, 1, 2);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::HeightSharded, 1, 1);
  addLayoutForOp(op, legalLayouts, BufferType::L1,
                 TensorMemoryLayout::BlockSharded, 1, 1);

  // Create custom checkShardCompatible function.
  //
  std::function<bool(mlir::Value, TTNNLayoutAttr const &, mlir::Operation *,
                     TTNNLayoutAttr const &)>
      checkShardCompatible = [](mlir::Value producerOperand,
                                TTNNLayoutAttr const &producerLayout,
                                mlir::Operation *consumerOp,
                                TTNNLayoutAttr const &consumerLayout) {
        // Interleaved to sharded is always supported.
        //
        if (producerLayout.hasInterleavedDRAMTensorMemoryLayout()) {
          return true;
        }

        // Simple shard compat assumption. Try to keep same shard layout.
        //
        if (producerLayout.getMemLayout() != consumerLayout.getMemLayout()) {
          return false;
        }

        return true;
      };

  ShardSolver shardSolver(legalLayouts, opL1MemSpecs, l1ChainedOps,
                          usableL1CacheSize, overrideReshardEdges,
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
    ShardSolver::RemainingLayoutAttrs validLayouts =
        shardSolver.at(opL1MemSpec.op);
    const TTNNLayoutAttr *selectedLayout = validLayouts.begin().get();
    shardSolver.set(opL1MemSpec.op, *selectedLayout);
  }

  llvm::DenseMap<mlir::Operation *, TTNNLayoutAttr> selectedOpLayout =
      shardSolver.finish().selectedOpLayout;
  float totalCoreUsage = 0;
  for (const auto &opLayout : selectedOpLayout) {
    totalCoreUsage += opLayout.second.getGrid().getGridVolume();
  }

  ASSERT_EQ(totalCoreUsage, accMaxCoreUsage[firstOp][0]);
}
