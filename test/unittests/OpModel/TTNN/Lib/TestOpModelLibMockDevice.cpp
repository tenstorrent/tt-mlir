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
#include <string>
#include <vector>

namespace mlir::tt::ttnn::op_model {

// Arches exercised by the mock-device tests. Multi-chip mock cluster
// descriptors exist for both Wormhole and Blackhole; see
// get_mock_cluster_desc_name() in tt-metal for the supported (arch, num_chips)
// combinations. The 32-chip Galaxy 6U mesh is Wormhole-only (no Blackhole
// 32-chip mock cluster descriptor exists yet).
//
// NOTE: the 32-chip Galaxy 6U (WH 4x8) cases need a tt-metal build where the
// mock 6u fabric setup is fixed. Root cause: the UMD 6u cluster descriptor
// (the one shipped in packaged tt-metal) lacks the chip->bus_id mapping, so
// get_ubb_id() can't derive the rack tray positions; galaxy corner-pinning then
// fatals with "Failed to add pinning constraints" (tt-umd#2896). It passes
// locally only because the dev tree also exposes tt-metal's custom 6u
// descriptor (which has chip_to_bus_id) on the search path. Fixed by either
// tt-umd#2896 (add chip_to_bus_id to the UMD 6u desc) or tt-metal#47731 (skip
// galaxy corner-pinning for mock clusters). These cases are kept here ready;
// they pass once tt-mlir's pinned tt-metal includes one of those fixes.
inline std::string archName(ttcore::Arch arch) {
  switch (arch) {
  case ttcore::Arch::WormholeB0:
    return "WormholeB0";
  case ttcore::Arch::Blackhole:
    return "Blackhole";
  default:
    return "UnknownArch";
  }
}

// Base fixture for mock device lib tests. Op-agnostic — sets up MLIR context
// and reshapes the mock device. Per-op test classes inherit from this and add
// their own WithParamInterface.
class OpModelLibMockDeviceBase : public OpModelFixture {
public:
  void setupMockDevice(ttcore::Arch arch, size_t meshRows, size_t meshCols) {
    // Initialize MLIR context and module.
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());

    SingletonDeviceContext &ctx = SingletonDeviceContext::getInstance();

    // reshapeMeshDevice keeps the arch of the currently open env, so to target
    // a specific arch we (re)open a fresh single-chip device of that arch
    // first. A single-chip mock reports the canonical worker grid for the arch,
    // which matches the default system descriptor for that arch, so the
    // open-time grid validation passes. We then reshape to the requested
    // topology below.
    if (ctx.isDeviceInitialized()) {
      SingletonDeviceContext::closeInstance();
    }
    SingletonDeviceContext::setSystemDesc(
        ttcore::SystemDescAttr::getDefault(&context, arch, {1}));
    ctx.openMockDevice(
        /*traceRegionSize=*/0,
        std::make_pair(static_cast<size_t>(1), static_cast<size_t>(1)));

    // Clear the bootstrap descriptor before reshaping: the device grid is
    // topology-dependent (different mock cluster descriptors report different
    // worker grids), and reshapeMeshDevice validates the new device grid
    // against the registered descriptor. We register a fresh descriptor built
    // from the reshaped device grid below.
    SingletonDeviceContext::setSystemDesc(ttcore::SystemDescAttr());

    // Reshape to the requested topology so getComputeGridShape() reflects the
    // actual hardware grid (which depends on the cluster descriptor's
    // harvesting masks).
    ctx.reshapeMeshDevice({meshRows, meshCols});

    // Build system desc from the actual device grid so the registered MLIR
    // device attribute matches the grid the open device reports.
    llvm::SmallVector<int64_t> gridShape =
        llvm::to_vector(ctx.getComputeGridShape());
    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &context, arch,
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

// Param: {arch, meshRows, meshCols, dim, clusterAxis}
struct MeshPartitionParam {
  ttcore::Arch arch;
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
    setupMockDevice(p.arch, p.meshRows, p.meshCols);
  }
};

static std::string
meshPartitionName(const ::testing::TestParamInfo<MeshPartitionParam> &info) {
  const auto &p = info.param;
  return archName(p.arch) + "_" + std::to_string(p.meshRows) + "x" +
         std::to_string(p.meshCols) + "_dim" + std::to_string(p.dim) + "_axis" +
         std::to_string(p.clusterAxis);
}

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

static std::vector<MeshPartitionParam> meshPartitionParams() {
  // {meshRows, meshCols, dim, clusterAxis}; chips = rows*cols.
  struct Topo {
    size_t rows, cols;
    int32_t dim;
    uint32_t axis;
  };
  // Topologies with <= 8 chips run on both Wormhole and Blackhole.
  const Topo common[] = {
      {1, 8, 0, 1}, {1, 8, 1, 1}, // T3K
      {2, 4, 0, 0}, {2, 4, 0, 1}, {2, 4, 1, 0}, {2, 4, 1, 1},
      {4, 2, 0, 0}, {4, 2, 0, 1}, {4, 2, 1, 0}, {4, 2, 1, 1},
      {1, 2, 0, 1}, {1, 2, 1, 1}, // N300
      {2, 1, 0, 0}, {2, 1, 1, 0}, {1, 4, 0, 1}, {1, 4, 1, 1},
      {4, 1, 0, 0}, {4, 1, 1, 0},
  };
  // 32-chip Galaxy 6U: Wormhole only (no Blackhole 32-chip mock descriptor).
  // Needs the 6u mock fix (tt-umd#2896 / tt-metal#47731); see file header.
  const Topo whOnly[] = {{4, 8, 0, 0}, {4, 8, 1, 1}};
  std::vector<MeshPartitionParam> params;
  for (ttcore::Arch arch :
       {ttcore::Arch::WormholeB0, ttcore::Arch::Blackhole}) {
    for (const Topo &t : common) {
      params.push_back({arch, t.rows, t.cols, t.dim, t.axis});
    }
  }
  for (const Topo &t : whOnly) {
    params.push_back({ttcore::Arch::WormholeB0, t.rows, t.cols, t.dim, t.axis});
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(MeshPartition, MeshPartitionLibMockDeviceTest,
                         ::testing::ValuesIn(meshPartitionParams()),
                         meshPartitionName);

// --- AllGatherOp tests ---

struct AllGatherParam {
  ttcore::Arch arch;
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
    setupMockDevice(p.arch, p.meshRows, p.meshCols);
  }
};

static std::string
allGatherName(const ::testing::TestParamInfo<AllGatherParam> &info) {
  const auto &p = info.param;
  return archName(p.arch) + "_" + std::to_string(p.meshRows) + "x" +
         std::to_string(p.meshCols) + "_dim" + std::to_string(p.allGatherDim) +
         "_axis" + std::to_string(p.clusterAxis);
}

TEST_P(AllGatherLibMockDeviceTest, AllGatherOp) {
  const auto &p = GetParam();

  // Interleaved DRAM tiled input: [1, 1, 32, 128] — each device holds 1 shard.
  const llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 128};
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<AllGatherOp>::getOpConstraints(
      inputShape, layoutDRAMTiled, p.allGatherDim, p.clusterAxis,
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

static std::vector<AllGatherParam> allGatherParams() {
  struct Topo {
    size_t rows, cols;
    int32_t dim;
    uint32_t axis;
  };
  // <= 8 chips: run on both Wormhole and Blackhole.
  const Topo common[] = {
      {1, 2, 3, 1}, // N300: gather last dim on axis 1
      {1, 2, 2, 1}, // N300: gather dim 2 on axis 1
      {2, 1, 3, 0}, // gather last dim on axis 0
      {1, 8, 3, 1}, // T3K: gather last dim on col axis
  };
  // 32-chip Galaxy 6U: Wormhole only (needs the 6u mock fix; see file header).
  const Topo whOnly[] = {
      {4, 8, 3, 1}, // gather last dim on col axis
      {4, 8, 3, 0}, // gather last dim on row axis
  };
  std::vector<AllGatherParam> params;
  for (ttcore::Arch arch :
       {ttcore::Arch::WormholeB0, ttcore::Arch::Blackhole}) {
    for (const Topo &t : common) {
      params.push_back({arch, t.rows, t.cols, t.dim, t.axis});
    }
  }
  for (const Topo &t : whOnly) {
    params.push_back({ttcore::Arch::WormholeB0, t.rows, t.cols, t.dim, t.axis});
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(AllGather, AllGatherLibMockDeviceTest,
                         ::testing::ValuesIn(allGatherParams()), allGatherName);

// --- ReduceScatterOp tests ---

struct ReduceScatterParam {
  ttcore::Arch arch;
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
    setupMockDevice(p.arch, p.meshRows, p.meshCols);
  }
};

static std::string
reduceScatterName(const ::testing::TestParamInfo<ReduceScatterParam> &info) {
  const auto &p = info.param;
  return archName(p.arch) + "_" + std::to_string(p.meshRows) + "x" +
         std::to_string(p.meshCols) + "_dim" + std::to_string(p.scatterDim) +
         "_axis" + std::to_string(p.clusterAxis);
}

TEST_P(ReduceScatterLibMockDeviceTest, ReduceScatterOp) {
  const auto &p = GetParam();

  // Interleaved DRAM tiled input: [1, 1, 32, 256] — scatter halves each dim.
  const llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 256};
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<ReduceScatterOp>::getOpConstraints(
      inputShape, layoutDRAMTiled, p.scatterDim, p.clusterAxis,
      /*subDeviceId=*/std::nullopt,
      /*numLinks=*/std::optional<uint32_t>(1),
      /*topology=*/std::optional<ttcore::Topology>(ttcore::Topology::Linear),
      /*computeConfig=*/std::nullopt, layoutDRAMTiled);

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

static std::vector<ReduceScatterParam> reduceScatterParams() {
  struct Topo {
    size_t rows, cols;
    int32_t dim;
    uint32_t axis;
  };
  // <= 8 chips: run on both Wormhole and Blackhole.
  const Topo common[] = {
      {1, 2, 3, 1}, // N300: scatter last dim on axis 1
      {1, 2, 2, 1}, // N300: scatter dim 2 on axis 1
      {2, 1, 3, 0}, // scatter last dim on axis 0
      {1, 8, 3, 1}, // T3K: scatter last dim on col axis
  };
  // 32-chip Galaxy 6U: Wormhole only (needs the 6u mock fix; see file header).
  const Topo whOnly[] = {
      {4, 8, 3, 1}, // scatter last dim on col axis
      {4, 8, 3, 0}, // scatter last dim on row axis
  };
  std::vector<ReduceScatterParam> params;
  for (ttcore::Arch arch :
       {ttcore::Arch::WormholeB0, ttcore::Arch::Blackhole}) {
    for (const Topo &t : common) {
      params.push_back({arch, t.rows, t.cols, t.dim, t.axis});
    }
  }
  for (const Topo &t : whOnly) {
    params.push_back({ttcore::Arch::WormholeB0, t.rows, t.cols, t.dim, t.axis});
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(ReduceScatter, ReduceScatterLibMockDeviceTest,
                         ::testing::ValuesIn(reduceScatterParams()),
                         reduceScatterName);

// --- AllReduceOp tests ---

struct AllReduceParam {
  ttcore::Arch arch;
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
    setupMockDevice(p.arch, p.meshRows, p.meshCols);
  }
};

static std::string
allReduceName(const ::testing::TestParamInfo<AllReduceParam> &info) {
  const auto &p = info.param;
  return archName(p.arch) + "_" + std::to_string(p.meshRows) + "x" +
         std::to_string(p.meshCols) + "_axis" + std::to_string(p.clusterAxis);
}

TEST_P(AllReduceLibMockDeviceTest, AllReduceOp) {
  const auto &p = GetParam();

  const llvm::SmallVector<int64_t> inputShape = {1, 1, 32, 128};
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<AllReduceOp>::getOpConstraints(
      inputShape, layoutDRAMTiled, p.clusterAxis,
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

static std::vector<AllReduceParam> allReduceParams() {
  struct Topo {
    size_t rows, cols;
    uint32_t axis;
  };
  // <= 8 chips: run on both Wormhole and Blackhole.
  const Topo common[] = {
      {1, 2, 1}, // N300: reduce on col axis
      {1, 8, 1}, // T3K: reduce on col axis
  };
  // 32-chip Galaxy 6U: Wormhole only (needs the 6u mock fix; see file header).
  const Topo whOnly[] = {
      {4, 8, 1}, // reduce on col axis
      {4, 8, 0}, // reduce on row axis
  };
  std::vector<AllReduceParam> params;
  for (ttcore::Arch arch :
       {ttcore::Arch::WormholeB0, ttcore::Arch::Blackhole}) {
    for (const Topo &t : common) {
      params.push_back({arch, t.rows, t.cols, t.axis});
    }
  }
  for (const Topo &t : whOnly) {
    params.push_back({ttcore::Arch::WormholeB0, t.rows, t.cols, t.axis});
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(AllReduce, AllReduceLibMockDeviceTest,
                         ::testing::ValuesIn(allReduceParams()), allReduceName);

// --- CCL L1-sharded output coverage ---
//
// The collective tests above only exercise DRAM-interleaved outputs. These
// tests pin down that the op-model constraint queries also accept L1 outputs —
// interleaved and width/height/block sharded — and that an output memory config
// the op cannot satisfy is rejected *gracefully* (a returned error, not an
// abort). The graceful-error contract is what TTNNRowMajorLayoutPropagation
// relies on to fall back instead of crashing. Fixed on a 1x8 Wormhole (T3K)
// mesh; L1 sharding is over each chip's worker grid, independent of the mesh.
namespace {
// The op accepts this output config: for an L1 output the reported output
// buffer must live in L1 (non-zero), and peak L1 (CB + buffers) is non-zero.
void expectAccepted(llvm::Expected<OpConstraints> &exp,
                    const std::string &label, BufferType buf) {
  if (!exp) {
    ADD_FAILURE() << label << " was unexpectedly rejected: "
                  << llvm::toString(exp.takeError());
    return;
  }
  if (buf == BufferType::L1) {
    EXPECT_GT(exp->outputL1BufferSize, 0u) << label << ": output not in L1";
  }
  EXPECT_GT(exp->peakL1MemorySize, 0u) << label << ": no L1 usage reported";
}

// The op cannot satisfy this output config and must say so gracefully — a
// returned error, never an abort. Consume the error.
void expectGracefulReject(llvm::Expected<OpConstraints> &exp,
                          const std::string &label) {
  if (exp) {
    ADD_FAILURE() << label << " was expected to be rejected but was accepted";
    return;
  }
  llvm::consumeError(exp.takeError());
}
} // namespace

class CCLShardedOutputTest : public OpModelLibMockDeviceBase {};

TEST_F(CCLShardedOutputTest, AllGatherOutputLayouts) {
  setupMockDevice(ttcore::Arch::WormholeB0, /*meshRows=*/1, /*meshCols=*/8);
  // gather last dim on axis 1: [1,1,32,128] -> [1,1,32,1024].
  const llvm::SmallVector<int64_t> in = {1, 1, 32, 128};
  const llvm::SmallVector<int64_t> out = {1, 1, 32, 1024};
  const TTNNLayoutAttr inL =
      CreateTiledLayout(in, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  auto query = [&](BufferType buf, TensorMemoryLayout mem) {
    return OpModel<AllGatherOp>::getOpConstraints(
        in, inL, /*allGatherDim=*/3, /*clusterAxis=*/1,
        /*subDeviceId=*/std::nullopt, /*numLinks=*/std::optional<uint32_t>(1),
        std::optional<ttcore::Topology>(ttcore::Topology::Linear),
        CreateTiledLayout(out, buf, mem));
  };
  auto dram = query(BufferType::DRAM, TensorMemoryLayout::Interleaved);
  expectAccepted(dram, "all_gather DRAM-interleaved", BufferType::DRAM);
  auto l1i = query(BufferType::L1, TensorMemoryLayout::Interleaved);
  expectAccepted(l1i, "all_gather L1-interleaved", BufferType::L1);
  auto ws = query(BufferType::L1, TensorMemoryLayout::WidthSharded);
  expectAccepted(ws, "all_gather L1-width-sharded", BufferType::L1);
  auto hs = query(BufferType::L1, TensorMemoryLayout::HeightSharded);
  expectAccepted(hs, "all_gather L1-height-sharded", BufferType::L1);
  auto bs = query(BufferType::L1, TensorMemoryLayout::BlockSharded);
  expectAccepted(bs, "all_gather L1-block-sharded", BufferType::L1);
}

TEST_F(CCLShardedOutputTest, ReduceScatterOutputLayouts) {
  setupMockDevice(ttcore::Arch::WormholeB0, /*meshRows=*/1, /*meshCols=*/8);
  // scatter last dim on axis 1: [1,1,32,256] -> [1,1,32,32].
  const llvm::SmallVector<int64_t> in = {1, 1, 32, 256};
  const llvm::SmallVector<int64_t> out = {1, 1, 32, 32};
  const TTNNLayoutAttr inL =
      CreateTiledLayout(in, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  auto query = [&](BufferType buf, TensorMemoryLayout mem) {
    return OpModel<ReduceScatterOp>::getOpConstraints(
        in, inL, /*scatterDim=*/3, /*clusterAxis=*/1,
        /*subDeviceId=*/std::nullopt, /*numLinks=*/std::optional<uint32_t>(1),
        std::optional<ttcore::Topology>(ttcore::Topology::Linear),
        /*computeConfig=*/std::nullopt, CreateTiledLayout(out, buf, mem));
  };
  auto dram = query(BufferType::DRAM, TensorMemoryLayout::Interleaved);
  expectAccepted(dram, "reduce_scatter DRAM-interleaved", BufferType::DRAM);
  auto l1i = query(BufferType::L1, TensorMemoryLayout::Interleaved);
  expectAccepted(l1i, "reduce_scatter L1-interleaved", BufferType::L1);
  auto ws = query(BufferType::L1, TensorMemoryLayout::WidthSharded);
  expectAccepted(ws, "reduce_scatter L1-width-sharded", BufferType::L1);
  auto hs = query(BufferType::L1, TensorMemoryLayout::HeightSharded);
  expectAccepted(hs, "reduce_scatter L1-height-sharded", BufferType::L1);
  auto bs = query(BufferType::L1, TensorMemoryLayout::BlockSharded);
  expectAccepted(bs, "reduce_scatter L1-block-sharded", BufferType::L1);
}

TEST_F(CCLShardedOutputTest, AllReduceOutputLayouts) {
  setupMockDevice(ttcore::Arch::WormholeB0, /*meshRows=*/1, /*meshCols=*/8);
  // all_reduce preserves shape: [1,1,32,128].
  const llvm::SmallVector<int64_t> shape = {1, 1, 32, 128};
  const TTNNLayoutAttr inL = CreateTiledLayout(shape, BufferType::DRAM,
                                               TensorMemoryLayout::Interleaved);
  auto query = [&](BufferType buf, TensorMemoryLayout mem) {
    return OpModel<AllReduceOp>::getOpConstraints(
        shape, inL, /*clusterAxis=*/1, /*subDeviceId=*/std::nullopt,
        /*numLinks=*/std::optional<uint32_t>(1),
        std::optional<ttcore::Topology>(ttcore::Topology::Linear),
        CreateTiledLayout(shape, buf, mem));
  };
  auto dram = query(BufferType::DRAM, TensorMemoryLayout::Interleaved);
  expectAccepted(dram, "all_reduce DRAM-interleaved", BufferType::DRAM);
  auto l1i = query(BufferType::L1, TensorMemoryLayout::Interleaved);
  expectAccepted(l1i, "all_reduce L1-interleaved", BufferType::L1);
  auto hs = query(BufferType::L1, TensorMemoryLayout::HeightSharded);
  expectAccepted(hs, "all_reduce L1-height-sharded", BufferType::L1);
  // A 1x4-tile output cannot tile-fit a width/block shard grid over the worker
  // cores; the query must reject gracefully rather than abort.
  auto ws = query(BufferType::L1, TensorMemoryLayout::WidthSharded);
  expectGracefulReject(ws, "all_reduce L1-width-sharded (too small)");
  auto bs = query(BufferType::L1, TensorMemoryLayout::BlockSharded);
  expectGracefulReject(bs, "all_reduce L1-block-sharded (too small)");
}

// --- DistributedRMSNormOp tests ---
//
// fused_rms_minimal constraints (from the tt-metal test):
//   8 cores × 2 tiles (N=512) on a 1×8 mesh.
//   input: (1,1,32,512) WIDTH_SHARDED L1, each core holds (32,64).
//   weight: (1,1,16,32) ROW_MAJOR L1.
//   stats: (1,1,32,512) WIDTH_SHARDED L1, all on core (0,0).

class DistributedRMSNormLibMockDeviceTest
    : public OpModelLibMockDeviceBase,
      public ::testing::WithParamInterface<ttcore::Arch> {
public:
  static constexpr uint32_t kNumCores = 8;
  static constexpr uint32_t kTilesPerCore = 2;
  static constexpr uint32_t kTileSize = 32;
  static constexpr uint32_t kN = kNumCores * kTilesPerCore * kTileSize; // 512

  // 1x8 mesh (8 chips) is available as a mock cluster descriptor on both
  // Wormhole (t3k) and Blackhole (8xP150).
  void SetUp() override { setupMockDevice(GetParam(), 1, kNumCores); }
};

TEST_P(DistributedRMSNormLibMockDeviceTest, FusedRmsMinimal1x8) {
  const llvm::SmallVector<int64_t> inputShape = {1, 1, kTileSize, kN};

  // WIDTH_SHARDED tiled input: virtualGrid {1, 8} → each core gets (32,64).
  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::WidthSharded,
      /*virtualGrid=*/llvm::SmallVector<int64_t>{1, kNumCores});

  // Weight: ROW_MAJOR, shape (N/32, 32).
  const llvm::SmallVector<int64_t> weightShape = {1, 1, kN / kTileSize,
                                                  kTileSize};
  const TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  // Stats: WIDTH_SHARDED on 1 core only (aggregation point).
  const llvm::SmallVector<int64_t> statsShape = {1, 1, kTileSize, kN};
  const TTNNLayoutAttr statsLayout = CreateTiledLayout(
      statsShape, BufferType::L1, TensorMemoryLayout::WidthSharded,
      /*virtualGrid=*/llvm::SmallVector<int64_t>{1, 1});

  // program_config: grid (x=8, y=1), block_w=2 tiles per core.
  auto gridCoord =
      mlir::tt::ttnn::CoreCoordAttr::get(&context, kNumCores, /*y=*/1);
  auto programConfig = LayerNormShardedMultiCoreProgramConfigAttr::get(
      &context, gridCoord, /*subblock_w=*/1, /*block_h=*/1,
      /*block_w=*/kTilesPerCore, /*inplace=*/false);

  auto constraintsExp = OpModel<DistributedRMSNormOp>::getOpConstraints(
      inputShape, inputLayout,
      std::optional<llvm::ArrayRef<int64_t>>(weightShape),
      std::optional<TTNNLayoutAttr>(weightLayout),
      /*residualShape=*/std::nullopt, /*residualLayout=*/std::nullopt,
      std::optional<llvm::ArrayRef<int64_t>>(statsShape),
      std::optional<TTNNLayoutAttr>(statsLayout),
      /*clusterAxis=*/1u, llvm::APFloat(1e-5f),
      /*subDeviceId=*/std::nullopt,
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
                     << constraintsExp->cbL1PeakSize
                     << " tensorPeak=" << constraintsExp->tensorL1PeakSize
                     << " peakL1=" << constraintsExp->peakL1MemorySize
                     << " outputBuf=" << constraintsExp->outputL1BufferSize;
  }
  EXPECT_TRUE(ok);
}

INSTANTIATE_TEST_SUITE_P(
    DistributedRMSNorm, DistributedRMSNormLibMockDeviceTest,
    ::testing::Values(ttcore::Arch::WormholeB0, ttcore::Arch::Blackhole),
    [](const ::testing::TestParamInfo<ttcore::Arch> &info) {
      return archName(info.param);
    });

} // namespace mlir::tt::ttnn::op_model

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - GTest takes ownership.
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
