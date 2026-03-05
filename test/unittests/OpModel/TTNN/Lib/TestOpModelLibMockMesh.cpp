// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test binary runs in its own process so that MetalContext is initialized
// with a mock {1,8} mesh topology from the start, without being polluted by
// a real-hardware topology from other test binaries.

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn::op_model {

// Fixture that opens a mock {1,8} mesh device from the start, so that
// MetalContext is initialized with the mock topology (not real hardware).
class OpModelLibMockMeshTest : public OpModelFixture {
public:
  void SetUp() override {
    // Initialize MLIR context and module without opening a real device.
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());

    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &context, ttcore::Arch::WormholeB0, {1, 8});
    SingletonDeviceContext::setSystemDesc(systemDesc);
    SingletonDeviceContext::getInstance().openMockDevice(
        /*traceRegionSize=*/6000000, std::make_pair<size_t, size_t>(1, 8));

    mlir::tt::ttcore::registerDevice(module.get());
  }
};

TEST_F(OpModelLibMockMeshTest, MeshPartitionOp) {
  const llvm::SmallVector<int64_t> inputShape = {8, 1025};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAMRowMajor = CreateRowMajorLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Tiled = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  const int32_t dim = 0;
  const std::optional<uint32_t> clusterAxis = 1;

  // Row-major layouts should succeed.
  auto constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutDRAMRowMajor, dim, clusterAxis,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Tiled layouts should fail validation (not supported by mesh_partition for
  // this input shape).
  constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutDRAMTiled, dim, clusterAxis,
      layoutDRAMTiled);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  if (!constraintsExp) {
    llvm::consumeError(constraintsExp.takeError());
  }

  constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutL1Tiled, dim, clusterAxis,
      layoutL1Tiled);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  if (!constraintsExp) {
    llvm::consumeError(constraintsExp.takeError());
  }
}

} // namespace mlir::tt::ttnn::op_model
