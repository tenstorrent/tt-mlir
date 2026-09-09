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
  const TTNNLayoutAttr layoutDRAMRowMajor = CreateRowMajorLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Tiled = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  const int32_t dim = p.dim;
  const std::optional<uint32_t> clusterAxis = p.clusterAxis;

  // Compute whether the post-split shape remains tile-aligned (multiple of 32).
  const size_t meshDims[] = {p.meshRows, p.meshCols};
  const int64_t splitFactor = static_cast<int64_t>(meshDims[p.clusterAxis]);
  const bool expectTilingSuccess = (inputShape[p.dim] / splitFactor) % 32 == 0;

  // Row-major layouts should always succeed.
  auto constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      inputShape, layoutDRAMRowMajor, dim, clusterAxis, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Tiled DRAM layout — succeeds only if post-split shape is tile-aligned.
  constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      inputShape, layoutDRAMTiled, dim, clusterAxis, layoutDRAMTiled);
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectTilingSuccess);
  if (!constraintsExp) {
    llvm::consumeError(constraintsExp.takeError());
  }

  // Tiled L1 layout — same condition.
  constraintsExp = OpModel<MeshPartitionOp>::getOpConstraints(
      inputShape, layoutL1Tiled, dim, clusterAxis, layoutL1Tiled);
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

// --- IndexerScoreDsaOp stateful-constraint test ---

// ttnn.experimental.indexer_score_dsa is Blackhole-only (its device op asserts
// arch == BLACKHOLE), so this fixture reopens the per-binary singleton on a
// Blackhole mock and restores the Wormhole 1x8 device afterwards -- leaving the
// singleton on Blackhole would poison every later test in this binary.
class IndexerScoreDsaLibMockDeviceTest : public OpModelLibMockDeviceBase {
public:
  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());

    reopenMockDevice(ttcore::Arch::Blackhole, {1, 1});
    mlir::tt::ttcore::registerDevice(module.get());
  }

  void TearDown() override {
    // Restore the topology MockDeviceEnvironment established for this binary.
    reopenMockDevice(ttcore::Arch::WormholeB0, {1, maxMockChips});
  }

private:
  // SystemDescAttr::getDefault's Blackhole worker grid (TTCoreOpsTypes.cpp) is
  // 13 wide, but tt-metal's mock blackhole_P150.yaml is 2-column harvested and
  // its device reports 11. SingletonDeviceContext::openDevice validates the two
  // against each other and hard-errors on a mismatch, so give this test a desc
  // whose grid matches the mock device it opens. Only the grid differs from the
  // default; the shared default is deliberately left alone, since it is the
  // fallback for every Blackhole compile and correcting it is a separate change
  // with its own golden churn.
  static constexpr int64_t mockBlackholeGridX = 11;

  static ttcore::SystemDescAttr
  withGridMatchingMockDevice(mlir::MLIRContext *ctx,
                             ttcore::SystemDescAttr desc) {
    llvm::SmallVector<ttcore::ChipDescAttr> chipDescs;
    for (ttcore::ChipDescAttr chip : desc.getChipDescs()) {
      llvm::SmallVector<int64_t> grid(chip.getGrid());
      assert(grid.size() == 2 && "expected a 2D worker grid");
      grid[1] = mockBlackholeGridX;
      chipDescs.push_back(ttcore::ChipDescAttr::get(
          ctx, chip.getArch(), grid, chip.getCoordTranslationOffsets(),
          chip.getL1Size(), chip.getNumDramChannels(),
          chip.getDramChannelSize(), chip.getNocL1AddressAlignBytes(),
          chip.getPcieAddressAlignBytes(), chip.getNocDRAMAddressAlignBytes(),
          chip.getL1UnreservedBase(), chip.getEriscL1UnreservedBase(),
          chip.getDramUnreservedBase(), chip.getDramUnreservedEnd(),
          chip.getSupportedDataTypes(), chip.getSupportedTileSizes(),
          chip.getDstPhysicalSizeTiles(), chip.getNumCBs(),
          chip.getNumComputeThreads(), chip.getNumDatamovementThreads(),
          chip.getDramGrid(), chip.getDramBankToLogicalWorkerNoc0(),
          chip.getDramBankToLogicalWorkerNoc1()));
    }
    return ttcore::SystemDescAttr::get(
        ctx, desc.getCpuDescs(), chipDescs, desc.getChipDescIndices(),
        desc.getChipCapabilities(), desc.getChipCoords(),
        desc.getChipChannels());
  }

  void reopenMockDevice(ttcore::Arch arch,
                        const std::pair<size_t, size_t> &meshShape) {
    mlir::MLIRContext tmpCtx;
    tmpCtx.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    ttcore::SystemDescAttr desc = ttcore::SystemDescAttr::getDefault(
        &tmpCtx, arch,
        {static_cast<int>(meshShape.first),
         static_cast<int>(meshShape.second)});
    if (arch == ttcore::Arch::Blackhole) {
      desc = withGridMatchingMockDevice(&tmpCtx, desc);
    }
    SingletonDeviceContext::closeInstance();
    SingletonDeviceContext::setSystemDesc(desc);
    SingletonDeviceContext::getInstance().openMockDevice(
        /*traceRegionSize=*/0, meshShape);
  }
};

// Pins the branch that the stateful migration introduced: a null initialState
// takes the stateless query, which reports no output allocations; a non-null
// one takes the with-state query, which really allocates and reports the
// output's record. The indexer's output inherits q's memory config, so an
// L1-resident q yields an L1 record -- the untracked-L1-output gap this
// migration closes.
TEST_F(IndexerScoreDsaLibMockDeviceTest, StatefulReportsOutputAllocation) {
  // Shapes mirror the transformer lit tests: query [B, Hi, Sq, D],
  // key [B, 1, T, D], weights [B, Hi, Sq, 1] -> score [B, 1, Sq, T].
  const llvm::SmallVector<int64_t> queryShape = {1, 8, 32, 128};
  const llvm::SmallVector<int64_t> keyShape = {1, 1, 32, 128};
  const llvm::SmallVector<int64_t> weightsShape = {1, 8, 32, 1};
  const llvm::SmallVector<int64_t> outputShape = {1, 1, 32, 32};

  const llvm::SmallVector<int64_t> physicalGrid = {1, 1};

  // q in L1 so the output inherits an L1 memory config.
  const TTNNLayoutAttr queryLayout = CreateTiledLayout(
      queryShape, BufferType::L1, TensorMemoryLayout::Interleaved,
      /*virtualGrid=*/std::nullopt, physicalGrid);
  const TTNNLayoutAttr keyLayout = CreateTiledLayout(
      keyShape, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      /*virtualGrid=*/std::nullopt, physicalGrid);
  const TTNNLayoutAttr weightsLayout = CreateTiledLayout(
      weightsShape, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      /*virtualGrid=*/std::nullopt, physicalGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, BufferType::L1, TensorMemoryLayout::Interleaved,
      /*virtualGrid=*/std::nullopt, physicalGrid);

  // Stateless: no state to apply, so tt-metal runs the NO_DISPATCH capture and
  // reports no output allocations.
  auto statelessExp = OpModel<IndexerScoreDsaOp>::getOpConstraints(
      queryShape, queryLayout, keyShape, keyLayout, weightsShape, weightsLayout,
      /*chunkStartIdx=*/0, outputLayout, /*initialState=*/nullptr);
  ASSERT_TRUE(static_cast<bool>(statelessExp))
      << llvm::toString(statelessExp.takeError());
  EXPECT_TRUE(statelessExp->outputAllocations.empty());

  // Stateful: an empty live set still yields a non-null state, which selects
  // the with-initial-state query and its NORMAL-mode allocating capture.
  std::shared_ptr<MockAllocatorState> state = buildInitialState({});
  ASSERT_NE(state, nullptr);

  auto statefulExp = OpModel<IndexerScoreDsaOp>::getOpConstraints(
      queryShape, queryLayout, keyShape, keyLayout, weightsShape, weightsLayout,
      /*chunkStartIdx=*/0, outputLayout, state.get());
  ASSERT_TRUE(static_cast<bool>(statefulExp))
      << llvm::toString(statefulExp.takeError());
  ASSERT_EQ(statefulExp->outputAllocations.size(), 1u);
  EXPECT_EQ(statefulExp->outputAllocations[0].bufferType, BufferType::L1);
}

} // namespace mlir::tt::ttnn::op_model

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - GTest takes ownership.
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
