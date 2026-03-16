// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test binary runs in its own process so that MetalContext is initialized
// with a mock topology from the start, without being polluted by a
// real-hardware topology from other test binaries.
//
// Mock mode is configured once per binary via MockDeviceEnvironment (registered
// in main()). Individual tests reshape the MeshDevice to the desired topology
// via reshapeMeshDevice(). New test suites can be added without worrying about
// mock mode lifecycle.

#include "MockDeviceFixture.h"
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

// Base fixture for mock device lib tests. Op-agnostic — sets up MLIR context
// and reshapes the mock device. Per-op test classes inherit from this and add
// their own WithParamInterface.
class OpModelLibMockDeviceBase : public OpModelFixture {
public:
  void setupMockDevice(size_t meshRows, size_t meshCols) {
    // Initialize MLIR context and module.
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());

    // Update system desc and reshape the mock device for this test's topology.
    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &context, ttcore::Arch::WormholeB0,
        {static_cast<int>(meshRows), static_cast<int>(meshCols)});
    SingletonDeviceContext::setSystemDesc(systemDesc);
    SingletonDeviceContext::getInstance().reshapeMeshDevice(
        {meshRows, meshCols});

    mlir::tt::ttcore::registerDevice(module.get());
  }

  void SetUp() override {
    // Override OpModelFixture::SetUp to prevent it from opening a real device;
    // MockDeviceEnvironment opens the mock device once per binary.
    // Subclasses should call setupMockDevice() instead to get desired grid
    // shapes.
  }

  void TearDown() override {
    // Override OpModelFixture::TearDown to prevent it from closing the device;
    // MockDeviceEnvironment handles the final close.
  }
};

// --- MeshPartitionOp tests ---

// Param: {meshRows, meshCols, dim, clusterAxis}
struct MeshPartitionParam {
  size_t meshRows;
  size_t meshCols;
  int32_t dim;
  uint32_t clusterAxis;
};

class MeshPartitionLibMockDeviceTest
    : public OpModelLibMockDeviceBase,
      public ::testing::WithParamInterface<MeshPartitionParam> {
public:
  void SetUp() override {
    const auto &p = GetParam();
    setupMockDevice(p.meshRows, p.meshCols);
  }
};

TEST_P(MeshPartitionLibMockDeviceTest, MeshPartitionOp) {
  const auto &p = GetParam();

  // {64, 128} — both dims tile-aligned (multiples of 32). After splitting,
  // tiling succeeds only if the split dimension stays a multiple of 32.
  // e.g. 128/2=64 ✓, 128/4=32 ✓, 128/8=16 ✗, 64/2=32 ✓, 64/4=16 ✗
  const llvm::SmallVector<int64_t> inputShape = {64, 128};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAMRowMajor = CreateRowMajorLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Tiled = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  const int32_t dim = p.dim;
  const std::optional<uint32_t> clusterAxis = p.clusterAxis;

  // Compute whether the post-split shape remains tile-aligned (multiple of 32).
  const size_t meshDims[] = {p.meshRows, p.meshCols};
  const int64_t splitFactor = static_cast<int64_t>(meshDims[p.clusterAxis]);
  const bool expectTilingSuccess = (inputShape[p.dim] / splitFactor) % 32 == 0;

  // Row-major layouts should always succeed.
  auto constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutDRAMRowMajor, dim, clusterAxis,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Tiled DRAM layout — succeeds only if post-split shape is tile-aligned.
  constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutDRAMTiled, dim, clusterAxis,
      layoutDRAMTiled);
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectTilingSuccess);
  if (!constraintsExp) {
    llvm::consumeError(constraintsExp.takeError());
  }

  // Tiled L1 layout — same condition.
  constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutL1Tiled, dim, clusterAxis,
      layoutL1Tiled);
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectTilingSuccess);
  if (!constraintsExp) {
    llvm::consumeError(constraintsExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    MeshPartition, MeshPartitionLibMockDeviceTest,
    ::testing::Values(
        // {1,8} mesh: axis 0=1, axis 1=8. Only axis 1 splits.
        MeshPartitionParam{1, 8, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 8, 1, 1}, // split dim 1 on axis 1
        // {8,1} mesh: axis 0=8, axis 1=1. Only axis 0 splits.
        MeshPartitionParam{8, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{8, 1, 1, 0}, // split dim 1 on axis 0
        // {2,4} mesh: both axes split (axis 0=2, axis 1=4).
        MeshPartitionParam{2, 4, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{2, 4, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{2, 4, 1, 0}, // split dim 1 on axis 0
        MeshPartitionParam{2, 4, 1, 1}, // split dim 1 on axis 1
        // {4,2} mesh: both axes split (axis 0=4, axis 1=2).
        MeshPartitionParam{4, 2, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{4, 2, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{4, 2, 1, 0}, // split dim 1 on axis 0
        MeshPartitionParam{4, 2, 1, 1}, // split dim 1 on axis 1
        // {1,2} mesh: axis 0=1, axis 1=2. Only axis 1 splits.
        MeshPartitionParam{1, 2, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 2, 1, 1}, // split dim 1 on axis 1
        // {2,1} mesh: axis 0=2, axis 1=1. Only axis 0 splits.
        MeshPartitionParam{2, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{2, 1, 1, 0}, // split dim 1 on axis 0
        // {1,4} mesh: axis 0=1, axis 1=4. Only axis 1 splits.
        MeshPartitionParam{1, 4, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 4, 1, 1}, // split dim 1 on axis 1
        // {4,1} mesh: axis 0=4, axis 1=1. Only axis 0 splits.
        MeshPartitionParam{4, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{4, 1, 1, 0}  // split dim 1 on axis 0
        ));

} // namespace mlir::tt::ttnn::op_model

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - GTest takes ownership.
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
