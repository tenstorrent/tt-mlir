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

    // Reshape the device first so getComputeGridShape() reflects the actual
    // hardware grid (which depends on the cluster descriptor's harvesting masks).
    SingletonDeviceContext::getInstance().reshapeMeshDevice({meshRows, meshCols});

    // Build system desc from the actual device grid so the registered MLIR
    // device attribute matches what checkDeviceWorkerGrid sees.
    auto gridShape = SingletonDeviceContext::getInstance().getComputeGridShape();
    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &context, ttcore::Arch::WormholeB0,
        {static_cast<int>(meshRows), static_cast<int>(meshCols)}, gridShape);
    SingletonDeviceContext::setSystemDesc(systemDesc);

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
  const auto workerGrid = CreateWorkerGrid();
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
        // {1,8} mesh (T3K): axis 1 splits, axis 0=1.
        MeshPartitionParam{1, 8, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 8, 1, 1}, // split dim 1 on axis 1
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
        // {1,2} mesh (N300): axis 1 splits, axis 0=1.
        MeshPartitionParam{1, 2, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 2, 1, 1}, // split dim 1 on axis 1
        // {2,1} mesh: axis 0 splits, axis 1=1.
        MeshPartitionParam{2, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{2, 1, 1, 0}, // split dim 1 on axis 0
        // {1,4} mesh: axis 1 splits, axis 0=1.
        MeshPartitionParam{1, 4, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 4, 1, 1}, // split dim 1 on axis 1
        // {4,1} mesh: axis 0 splits, axis 1=1.
        MeshPartitionParam{4, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{4, 1, 1, 0}, // split dim 1 on axis 0
        // {4,8} mesh (Galaxy 6U): both axes split.
        MeshPartitionParam{4, 8, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{4, 8, 1, 1}  // split dim 1 on axis 1
        ));

// --- AllGatherOp tests ---

struct AllGatherParam {
  size_t meshRows;
  size_t meshCols;
  int32_t allGatherDim;
  uint32_t clusterAxis;
};

class AllGatherLibMockDeviceTest
    : public OpModelLibMockDeviceBase,
      public ::testing::WithParamInterface<AllGatherParam> {
public:
  void SetUp() override {
    const auto &p = GetParam();
    setupMockDevice(p.meshRows, p.meshCols);
  }
};

TEST_P(AllGatherLibMockDeviceTest, AllGatherOp) {
  const auto &p = GetParam();

  // Interleaved DRAM tiled input: [1, 1, 32, 128] — each device holds 1 shard.
  const llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 128};
  const auto workerGrid = CreateWorkerGrid();
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<AllGatherOp>::getOpConstraints(
      workerGrid, inputShape, layoutDRAMTiled, p.allGatherDim, p.clusterAxis,
      /*subDeviceId=*/std::nullopt,
      /*numLinks=*/std::optional<uint32_t>(1),
      /*topology=*/std::optional<ttcore::Topology>(ttcore::Topology::Linear),
      layoutDRAMTiled);

  bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    GTEST_LOG_(INFO) << "all_gather constraints error: "
                     << llvm::toString(constraintsExp.takeError());
  } else {
    GTEST_LOG_(INFO) << "all_gather constraints ok: cbPeak="
                     << constraintsExp->cbL1PeakSize;
  }
  EXPECT_TRUE(ok);
}

INSTANTIATE_TEST_SUITE_P(
    AllGather, AllGatherLibMockDeviceTest,
    ::testing::Values(
        // 1x2 mesh (N300): gather along last dim on axis 1
        AllGatherParam{1, 2, 3, 1},
        // 1x2 mesh (N300): gather along dim 2 on axis 1
        AllGatherParam{1, 2, 2, 1},
        // 2x1 mesh: gather along last dim on axis 0
        AllGatherParam{2, 1, 3, 0},
        // 1x8 mesh (T3K): gather along last dim on col axis
        AllGatherParam{1, 8, 3, 1},
        // 4x8 mesh (Galaxy 6U): gather along last dim on col axis
        AllGatherParam{4, 8, 3, 1},
        // 4x8 mesh (Galaxy 6U): gather along last dim on row axis
        AllGatherParam{4, 8, 3, 0}));

// --- ReduceScatterOp tests ---

struct ReduceScatterParam {
  size_t meshRows;
  size_t meshCols;
  int32_t scatterDim;
  uint32_t clusterAxis;
};

class ReduceScatterLibMockDeviceTest
    : public OpModelLibMockDeviceBase,
      public ::testing::WithParamInterface<ReduceScatterParam> {
public:
  void SetUp() override {
    const auto &p = GetParam();
    setupMockDevice(p.meshRows, p.meshCols);
  }
};

TEST_P(ReduceScatterLibMockDeviceTest, ReduceScatterOp) {
  const auto &p = GetParam();

  // Interleaved DRAM tiled input: [1, 1, 32, 256] — scatter halves each dim.
  const llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 256};
  const auto workerGrid = CreateWorkerGrid();
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<ReduceScatterOp>::getOpConstraints(
      workerGrid, inputShape, layoutDRAMTiled, p.scatterDim, p.clusterAxis,
      /*subDeviceId=*/std::nullopt,
      /*numLinks=*/std::optional<uint32_t>(1),
      /*topology=*/std::optional<ttcore::Topology>(ttcore::Topology::Linear),
      layoutDRAMTiled);

  bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    GTEST_LOG_(INFO) << "reduce_scatter constraints error: "
                     << llvm::toString(constraintsExp.takeError());
  } else {
    GTEST_LOG_(INFO) << "reduce_scatter constraints ok: cbPeak="
                     << constraintsExp->cbL1PeakSize;
  }
  EXPECT_TRUE(ok);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter, ReduceScatterLibMockDeviceTest,
    ::testing::Values(
        // 1x2 mesh (N300): scatter along last dim on axis 1
        ReduceScatterParam{1, 2, 3, 1},
        // 1x2 mesh (N300): scatter along dim 2 on axis 1
        ReduceScatterParam{1, 2, 2, 1},
        // 2x1 mesh: scatter along last dim on axis 0
        ReduceScatterParam{2, 1, 3, 0},
        // 1x8 mesh (T3K): scatter along last dim on col axis
        ReduceScatterParam{1, 8, 3, 1},
        // 4x8 mesh (Galaxy 6U): scatter along last dim on col axis
        ReduceScatterParam{4, 8, 3, 1},
        // 4x8 mesh (Galaxy 6U): scatter along last dim on row axis
        ReduceScatterParam{4, 8, 3, 0}));

// --- AllReduceOp tests ---

struct AllReduceParam {
  size_t meshRows;
  size_t meshCols;
  uint32_t clusterAxis;
};

class AllReduceLibMockDeviceTest
    : public OpModelLibMockDeviceBase,
      public ::testing::WithParamInterface<AllReduceParam> {
public:
  void SetUp() override {
    const auto &p = GetParam();
    setupMockDevice(p.meshRows, p.meshCols);
  }
};

TEST_P(AllReduceLibMockDeviceTest, AllReduceOp) {
  const auto &p = GetParam();

  const llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 128};
  const auto workerGrid = CreateWorkerGrid();
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<AllReduceOp>::getOpConstraints(
      workerGrid, inputShape, layoutDRAMTiled, p.clusterAxis,
      /*subDeviceId=*/std::nullopt,
      /*numLinks=*/std::optional<uint32_t>(1),
      /*topology=*/std::optional<ttcore::Topology>(ttcore::Topology::Linear),
      layoutDRAMTiled);

  bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    GTEST_LOG_(INFO) << "all_reduce constraints error: "
                     << llvm::toString(constraintsExp.takeError());
  } else {
    GTEST_LOG_(INFO) << "all_reduce constraints ok: cbPeak="
                     << constraintsExp->cbL1PeakSize;
  }
  EXPECT_TRUE(ok);
}

INSTANTIATE_TEST_SUITE_P(
    AllReduce, AllReduceLibMockDeviceTest,
    ::testing::Values(
        // 1x2 mesh (N300): reduce on col axis
        AllReduceParam{1, 2, 1},
        // 1x8 mesh (T3K): reduce on col axis
        AllReduceParam{1, 8, 1},
        // 4x8 mesh (Galaxy 6U): reduce on col axis
        AllReduceParam{4, 8, 1},
        // 4x8 mesh (Galaxy 6U): reduce on row axis
        AllReduceParam{4, 8, 0}));

// --- DistributedRMSNormOp tests ---
//
// fused_rms_minimal constraints (from the tt-metal test):
//   8 cores × 2 tiles (N=512) on a 1×8 mesh.
//   input: (1,1,32,512) WIDTH_SHARDED L1, each core holds (32,64).
//   weight: (1,1,16,32) ROW_MAJOR L1.
//   stats: (1,1,32,512) WIDTH_SHARDED L1, all on core (0,0).

class DistributedRMSNormLibMockDeviceTest : public OpModelLibMockDeviceBase {
public:
  static constexpr uint32_t kNumCores = 8;
  static constexpr uint32_t kTilesPerCore = 2;
  static constexpr uint32_t kTileSize = 32;
  static constexpr uint32_t kN = kNumCores * kTilesPerCore * kTileSize; // 512

  void SetUp() override { setupMockDevice(1, kNumCores); }
};

TEST_F(DistributedRMSNormLibMockDeviceTest, FusedRmsMinimal1x8) {
  const llvm::SmallVector<int64_t> inputShape = {1, 1, kTileSize, kN};
  const auto workerGrid = CreateWorkerGrid();

  // WIDTH_SHARDED tiled input: virtualGrid {1, 8} → each core gets (32,64).
  const TTNNLayoutAttr inputLayout =
      CreateTiledLayout(inputShape, BufferType::L1,
                        TensorMemoryLayout::WidthSharded,
                        /*virtualGrid=*/llvm::SmallVector<int64_t>{1, kNumCores});

  // Weight: ROW_MAJOR, shape (N/32, 32).
  const llvm::SmallVector<int64_t> weightShape = {1, 1, kN / kTileSize,
                                                  kTileSize};
  const TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  // Stats: WIDTH_SHARDED on 1 core only (aggregation point).
  const llvm::SmallVector<int64_t> statsShape = {1, 1, kTileSize, kN};
  const TTNNLayoutAttr statsLayout =
      CreateTiledLayout(statsShape, BufferType::L1,
                        TensorMemoryLayout::WidthSharded,
                        /*virtualGrid=*/llvm::SmallVector<int64_t>{1, 1});

  // program_config: grid (x=8, y=1), block_w=2 tiles per core.
  auto gridCoord =
      mlir::tt::ttnn::CoreCoordAttr::get(&context, kNumCores, /*y=*/1);
  auto programConfig = LayerNormShardedMultiCoreProgramConfigAttr::get(
      &context, gridCoord, /*subblock_w=*/1, /*block_h=*/1,
      /*block_w=*/kTilesPerCore, /*inplace=*/false);

  auto constraintsExp = OpModel<DistributedRMSNormOp>::getOpConstraints(
      workerGrid, inputShape, inputLayout,
      std::optional<llvm::ArrayRef<int64_t>>(weightShape),
      std::optional<TTNNLayoutAttr>(weightLayout),
      /*residualShape=*/std::nullopt, /*residualLayout=*/std::nullopt,
      std::optional<llvm::ArrayRef<int64_t>>(statsShape),
      std::optional<TTNNLayoutAttr>(statsLayout),
      /*clusterAxis=*/1u, llvm::APFloat(1e-5f),
      /*numLinks=*/std::optional<uint32_t>(1),
      std::optional<ttcore::Topology>(ttcore::Topology::Linear),
      /*computeConfig=*/std::nullopt,
      std::optional<LayerNormShardedMultiCoreProgramConfigAttr>(programConfig),
      inputLayout);

  bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    GTEST_LOG_(INFO) << "distributed_rms_norm constraints error: "
                     << llvm::toString(constraintsExp.takeError());
  } else {
    GTEST_LOG_(INFO) << "distributed_rms_norm constraints ok: cbPeak="
                     << constraintsExp->cbL1PeakSize;
  }
  EXPECT_TRUE(ok);
}

} // namespace mlir::tt::ttnn::op_model

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - GTest takes ownership.
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
