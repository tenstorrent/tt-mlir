// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Canary test for tt-metal mock device crash.
//
// Currently segfaults in Inspector/WatcherServer because CreateKernel
// dereferences global singletons that aren't initialized for mock contexts.
// Tracked in https://github.com/tenstorrent/tt-metal/issues/39849
//
// Uses EXPECT_DEATH so this is an expected failure:
//   - While #39849 is open: subprocess crashes -> EXPECT_DEATH passes -> CI green
//   - When #39849 is fixed: subprocess doesn't crash -> EXPECT_DEATH fails -> CI red
//     This signals that the fix has landed and the TTMLIR_DISABLE_MOCK_DEVICE
//     default can be flipped to OFF.
// ============================================================================

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

namespace mlir::tt::ttnn::op_model {

// Fixture that sets up MLIR context on top of the mock device opened by
// MockDeviceEnvironment. Mirrors OpModelLibMockDeviceBase but for a 1x1 mesh.
class MockDeviceCrashTest : public OpModelFixture {
public:
  void SetUp() override {
    // Do NOT call OpModelFixture::SetUp — it opens a real device.
    // MockDeviceEnvironment already opened the mock device.
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
  }

  void TearDown() override {
    // MockDeviceEnvironment handles the final close.
  }
};

// IMPORTANT: If this test FAILS (EXPECT_DEATH fails because code doesn't
// crash), it means tt-metal#39849 is fixed and uplifted.
// In that case, flip TTMLIR_DISABLE_MOCK_DEVICE default to OFF in CMakeLists.
TEST_F(MockDeviceCrashTest, ExpectCrashUntil39849IsFixed) {
  const llvm::SmallVector<int64_t> shape = {1, 1, 64, 128};
  const auto workerGrid = CreateWorkerGrid();
  const TTNNLayoutAttr layout = CreateTiledLayout(
      shape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  EXPECT_DEATH(
      {
        auto result = OpModel<ReluOp>::getOpConstraints(
            workerGrid, shape, layout, layout);
        if (result) {
          (void)result.get();
        } else {
          llvm::consumeError(result.takeError());
        }
        _exit(0);
      },
      ".*");
}

} // namespace mlir::tt::ttnn::op_model

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Force mock mode via env var so openMockDevice actually uses mock mode.
  setenv("TTMLIR_DISABLE_MOCK_DEVICE", "0", /*overwrite=*/1);
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - GTest takes ownership.
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
