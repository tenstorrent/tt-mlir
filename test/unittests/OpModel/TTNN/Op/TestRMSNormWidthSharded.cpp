// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttnn {

class OpModelTest : public OpModelFixture {};

// Test that RMS norm with width-sharded input causes a crash in metal.
// This test verifies the workaround is still needed.
//
// IMPORTANT: If this test FAILS (code doesn't crash), it means the metal issue
// is fixed and uplifted. In that case, remove the workaround in
// RMSNormOp::getOpConstraints that rejects sharded inputs.
TEST_F(OpModelTest, RMSNormWidthShardedInputCrashTest) {
  constexpr int64_t h = 32;
  constexpr int64_t w = 2048;
  constexpr int64_t numCoresX = 8;
  constexpr int64_t numCoresY = 8;
  constexpr int64_t shardHeight = h;
  constexpr int64_t shardWidth = w / (numCoresX * numCoresY); // 2048 / 64 = 32

  llvm::SmallVector<int64_t> inputShape = {1, h, w};
  TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::WidthSharded,
      llvm::SmallVector<int64_t>{numCoresY, numCoresX},
      llvm::SmallVector<int64_t>{shardHeight, shardWidth});

  llvm::SmallVector<int64_t> weightShape = {w};
  TTNNLayoutAttr weightLayout = CreateTiledLayout(
      weightShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  llvm::SmallVector<int64_t> outputShape = {1, h, w};
  TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, BufferType::L1, TensorMemoryLayout::WidthSharded,
      llvm::SmallVector<int64_t>{numCoresY, numCoresX},
      llvm::SmallVector<int64_t>{shardHeight, shardWidth});

  auto deviceGrid = CreateWorkerGrid();
  llvm::APFloat epsilon(1e-5f);

  // This should crash without the workaround in RMSNormOp::getOpConstraints.
  EXPECT_DEATH(
      {
        auto constraintsExp = op_model::OpModel<RMSNormOp>::getOpConstraints(
            deviceGrid, inputShape, inputLayout, weightShape, weightLayout,
            /*biasShape=*/std::nullopt, /*biasLayout=*/std::nullopt, epsilon,
            outputLayout);
        if (constraintsExp) {
          (void)*constraintsExp;
        } else {
          llvm::consumeError(constraintsExp.takeError());
        }
      },
      ".*"); // Match any crash message
}

} // namespace mlir::tt::ttnn
