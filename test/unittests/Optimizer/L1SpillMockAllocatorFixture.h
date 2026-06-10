// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fixture for exercising the *real* stateful path of MockAllocatorL1Tracker:
// liveRecords -> buildInitialState -> getOpConstraintsWithState ->
// tt-metal build-from-records -> output_allocations -> pendingRecords/Snapshot.
//
// Unlike L1SpillTestFixture (which injects a synthetic backendValidator and
// never touches the op-model), this fixture leaves backendValidator NULL so
// MockAllocatorL1Tracker::validate() routes through the real op-model. The
// op-model runs against a MOCK DEVICE (no silicon), configured once per binary
// by MockDeviceEnvironment. Requires TTMLIR_ENABLE_OPMODEL.
//
// The test binary's main() MUST register the mock device environment:
//   ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());

#ifndef TEST_UNITTESTS_OPTIMIZER_L1SPILLMOCKALLOCATORFIXTURE_H
#define TEST_UNITTESTS_OPTIMIZER_L1SPILLMOCKALLOCATORFIXTURE_H

#include "L1SpillTestFixture.h"

#include <memory>

namespace mlir::tt::ttnn::test {

//===----------------------------------------------------------------------===//
// L1SpillMockAllocatorFixture
//
// Reuses every IR-builder, layout, observer, and inspection helper from
// L1SpillTestFixture unchanged. Adds runMock(), which drives
// L1SpillManagement<MockAllocatorL1Tracker> WITHOUT installing a fake
// validator, so the allocator-backed (stateful op-model) path executes.
//===----------------------------------------------------------------------===//
class L1SpillMockAllocatorFixture : public L1SpillTestFixture {
public:
  struct MockRunResult {
    RecordingObserver *observer;
  };

  /// Pass instance kept alive as a fixture member so the observer (owned by the
  /// pass) outlives runMock() and stays reachable via the returned pointer, and
  /// so getMockTracker() can inspect final tracker state after the run.
  std::unique_ptr<L1SpillManagement<MockAllocatorL1Tracker>> passMock;

  /// Run the allocator-backed pass on `func`. No backendValidator is installed,
  /// so MockAllocatorL1Tracker::validate() issues real (uncached) stateful
  /// op-model queries against the mock device.
  MockRunResult runMock() {
    auto obs = std::make_unique<RecordingObserver>();
    auto *rawObs = obs.get();

    auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
    ttcore::GridAttr deviceGrid = deviceAttr.getWorkerGrid();

    passMock = std::make_unique<L1SpillManagement<MockAllocatorL1Tracker>>(
        func, deviceGrid, l1BudgetPerCore, std::move(obs));
    // Intentionally leave backendValidator unset -> real op-model path.
    passMock->run();
    return {rawObs};
  }

  MockAllocatorL1Tracker &getMockTracker() {
    return passMock->getMemoryTracker();
  }

  bool mockPassFailed() const { return passMock && passMock->hasFailed(); }

  /// Number of currently-live allocation records the tracker is holding.
  /// After a completed run, this reflects tensors still resident in L1.
  size_t liveRecordCount() { return getMockTracker().liveRecords.size(); }
};

} // namespace mlir::tt::ttnn::test

#endif // TEST_UNITTESTS_OPTIMIZER_L1SPILLMOCKALLOCATORFIXTURE_H
