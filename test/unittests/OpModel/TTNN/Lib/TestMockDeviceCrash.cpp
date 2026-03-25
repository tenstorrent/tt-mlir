// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Crash canary for tt-metal inspector + mock device interaction.
//
// Forces TT_METAL_INSPECTOR=1 to override the default workaround
// (TT_METAL_INSPECTOR=0 set by SingletonDeviceContext::openDevice) and verify
// the inspector crash (tt-metal#40630) still exists.
//
// Uses EXPECT_DEATH with "threadsafe" style (fork+exec) so the child gets a
// clean process with TT_METAL_INSPECTOR=1 set before device initialization.
//
//   - While #40630 is open: subprocess crashes -> EXPECT_DEATH passes -> green
//   - When #40630 is fixed: subprocess exits cleanly -> EXPECT_DEATH fails
//     Remove this test and the TT_METAL_INSPECTOR=0 workaround in openDevice.
// ============================================================================

#include "MockDeviceFixture.h"
#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir::tt::ttnn::op_model {

class MockDeviceCrashTest : public OpModelFixture {
public:
  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
  }

  void TearDown() override {}
};

TEST_F(MockDeviceCrashTest, InspectorCrashUntil40630IsFixed) {
  const llvm::SmallVector<int64_t> shape = {1, 1, 64, 128};
  const auto workerGrid = CreateWorkerGrid();
  const TTNNLayoutAttr layout = CreateTiledLayout(
      shape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  EXPECT_DEATH(
      {
        auto result = OpModel<ReluOp>::getOpConstraints(workerGrid, shape,
                                                        layout, layout);
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
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // Force inspector ON to trigger the crash — overrides the openDevice
  // workaround which uses overwrite=0.
  setenv("TT_METAL_INSPECTOR", "1", /*overwrite=*/1);

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - GTest takes ownership.
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
