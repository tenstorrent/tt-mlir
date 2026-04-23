// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Provides a Google Test environment that configures mock mode once per binary.
// This exists because Metal's configure_mock_mode/disable_mock_mode cannot be
// reliably cycled within the same process — only the initial
// configure_mock_mode call works. By using a global test environment, mock mode
// is set up once before any test suite runs, and torn down once after all
// suites finish.
//
// Usage:
//   1. In your test binary's main(), register the environment:
//        int main(int argc, char **argv) {
//          ::testing::InitGoogleTest(&argc, argv);
//          ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
//          return RUN_ALL_TESTS();
//        }
//
//   2. Test fixtures can reshape the mock device per-test in SetUp():
//        SingletonDeviceContext::getInstance().reshapeMeshDevice({rows, cols});
//
//   3. Leave TearDown() empty — the environment handles the final close.

#ifndef UNITTESTS_OPMODEL_TTNN_MOCKDEVICEFIXTURE_H
#define UNITTESTS_OPMODEL_TTNN_MOCKDEVICEFIXTURE_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <utility>

// Base mock topology: Galaxy 6U (4×8, 32 chips).
// configure_mock_mode is called once per binary with this chip count (uses
// 6u_cluster_desc.yaml). Individual tests reshape to any sub-topology with
// rows ≤ 4 and cols ≤ 8 — covers N300 (1×2), T3K (1×8), Galaxy (4×8), etc.
static constexpr size_t mockMeshRows = 4;
static constexpr size_t mockMeshCols = 8;

// Configures mock mode once for the entire test binary.
// Register via ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment())
// in main().
class MockDeviceEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    mlir::MLIRContext tmpCtx;
    tmpCtx.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    auto systemDesc = mlir::tt::ttcore::SystemDescAttr::getDefault(
        &tmpCtx, mlir::tt::ttcore::Arch::WormholeB0,
        {static_cast<int>(mockMeshRows), static_cast<int>(mockMeshCols)});
    mlir::tt::ttnn::op_model::SingletonDeviceContext::setSystemDesc(systemDesc);
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .openMockDevice(
            /*traceRegionSize=*/0,
            /*meshShape=*/
            std::make_pair(mockMeshRows, mockMeshCols));
  }

  void TearDown() override {
    if (mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
            .isDeviceInitialized()) {
      mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
          .closeInstance();
    }
  }
};

#endif // UNITTESTS_OPMODEL_TTNN_MOCKDEVICEFIXTURE_H
