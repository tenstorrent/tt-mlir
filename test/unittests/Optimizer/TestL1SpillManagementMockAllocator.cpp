// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the allocator-backed stateful L1 spill path
// (MockAllocatorL1Tracker). These run against a MOCK DEVICE (no silicon),
// configured once per binary by MockDeviceEnvironment, and exercise the real
// op-model query path end to end:
//
//   liveRecords -> buildInitialState -> getOpConstraintsWithState
//     -> tt-metal build-from-records -> output_allocations
//     -> pendingRecords -> addTensor association -> Snapshot on eviction.
//
// Unlike TestL1SpillManagement.cpp (synthetic backendValidator, no device),
// nothing is injected here: sizes and fit decisions come from the real
// op-model / mock allocator. Each 1024x1024 bf16 tile, height-sharded on an
// {8,1} grid, is 256 KiB per core (2 MiB / 8 cores) -- budgets below are
// expressed as multiples of that.
//
// Guarded by TTMLIR_ENABLE_OPMODEL at the CMake level.

#include "L1SpillMockAllocatorFixture.h"
#include "MockDeviceFixture.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace mlir::tt::ttnn::test;

namespace {

// Per-core L1 footprint of one 1024x1024 bf16 height-sharded {8,1} tensor, as
// computed by the real mock op-model (verified empirically).
constexpr uint64_t kOpL1 = 256 * kKiB;

//===----------------------------------------------------------------------===//
// SmallGraphNoSpill
//
// Baseline for the real op-model path: a small graph well under device L1 must
// produce zero spills, and every op must report the true 256 KiB/core size
// (proving the op-model actually ran -- the fake `l1UsageBytes` arg is
// ignored).
//===----------------------------------------------------------------------===//
class MockAllocatorNoSpillTest : public L1SpillMockAllocatorFixture {};

TEST_F(MockAllocatorNoSpillTest, SmallGraphNoSpill) {
  l1BudgetPerCore = 100u * kKiB * 1024u; // 100 MiB: fixture budget never binds
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto tt = tensorType(shape, makeL1Sharded(shape));

  auto args = beginFunc({tt});
  auto *opA = addUnary(args[0], tt, 0);
  auto *opB = addUnary(opA->getResult(0), tt, 0);
  auto *opC = addBinary(opA->getResult(0), opB->getResult(0), tt, 0);
  finishFunc({opC->getResult(0)});

  auto [obs] = runMock();

  EXPECT_FALSE(mockPassFailed());
  EXPECT_EQ(countSpills(), 0u) << "graph fits comfortably; no spill expected";
  EXPECT_EQ(obs->ooms.size(), 0u);
  // Real op-model sizes were used (not the ignored fake arg).
  ASSERT_FALSE(obs->lives.empty());
  for (const auto &l : obs->lives) {
    EXPECT_EQ(l.opL1Usage, kOpL1)
        << "op-model must report the real per-core L1 size";
  }
  // All three outputs stay resident in L1 (nothing was pushed out).
  EXPECT_TRUE(resultIsL1(opA->getResult(0)));
  EXPECT_TRUE(resultIsL1(opB->getResult(0)));
  EXPECT_TRUE(resultIsL1(opC->getResult(0)));
}

//===----------------------------------------------------------------------===//
// FarthestLastUseVictimSpilled
//
// Farthest-last-use eviction driven through the allocator-backed tracker.
// Three 256 KiB producers are kept simultaneously live; a tight fixture budget
// admits two but not three, so creating the third trips OOM and the pass evicts
// the farthest-last-use victim (spill-to-DRAM). Exercises the full stateful
// machinery under eviction: liveRecords flattening into each stateful query,
// Snapshot capture, and record-restore during the eviction replay.
//===----------------------------------------------------------------------===//
class MockAllocatorSpillTest : public L1SpillMockAllocatorFixture {};

TEST_F(MockAllocatorSpillTest, FarthestLastUseVictimSpilled) {
  l1BudgetPerCore = 700 * kKiB; // ~2.7 producers: 2 fit, the 3rd trips OOM
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto tt = tensorType(shape, makeL1Sharded(shape));

  auto args = beginFunc({tt});
  auto *opA = addUnary(args[0], tt, 0);
  auto *opB = addUnary(args[0], tt, 0);
  auto *opPressure = addUnary(args[0], tt, 0);
  // Consumers in reverse order so opA has the farthest last-use -> evicted.
  auto *useB = addUnary(opB->getResult(0), tt, 0);
  auto *useA = addUnary(opA->getResult(0), tt, 0);
  finishFunc(
      {opPressure->getResult(0), useB->getResult(0), useA->getResult(0)});

  auto [obs] = runMock();

  EXPECT_FALSE(mockPassFailed());
  ASSERT_GE(obs->evictions.size(), 1u) << "3x256 KiB > 700 KiB must spill";
  EXPECT_EQ(obs->evictions.front().victim, opA)
      << "farthest-last-use victim must be opA";
  EXPECT_TRUE(wasSpilled(opA->getResult(0)));
  EXPECT_FALSE(wasSpilled(opB->getResult(0)));
  EXPECT_TRUE(resultIsL1(opB->getResult(0)));
}

//===----------------------------------------------------------------------===//
// OverSubscriptionSpills
//
// Sixteen 256 KiB producers are kept simultaneously live (4 MiB) against a
// realistic ~1.25 MiB per-core L1 budget (matching how a real compile sets the
// budget to the device L1). The stateful tracker must resolve the
// over-subscription by spilling (evict + refit), not crash and not cram
// everything into L1. Exercises the allocator-backed tracker under heavy L1
// pressure end to end.
//
// NOTE: an earlier version used a 100 MiB fixture budget to isolate the
// device-L1 gate via the old "OOM-as-backend-error -> demote" path. That path
// was removed by https://github.com/tenstorrent/tt-mlir/issues/9045 (a
// fragmentation OOM from the stateful query is now classified as OOM and taken
// through evict+refit, like the scalar tracker), so the budget is now set to a
// realistic value where base-sim and device agree, as in a real compile.
//===----------------------------------------------------------------------===//
class MockAllocatorDeviceL1Test : public L1SpillMockAllocatorFixture {};

TEST_F(MockAllocatorDeviceL1Test, OverSubscriptionSpills) {
  l1BudgetPerCore = 5 * kOpL1; // ~1.25 MiB: ~5 producers fit, rest must spill
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto tt = tensorType(shape, makeL1Sharded(shape));

  constexpr int kNumProducers = 16; // 16 x 256 KiB = 4 MiB >> budget
  auto args = beginFunc({tt});
  llvm::SmallVector<mlir::Operation *> producers;
  for (int i = 0; i < kNumProducers; ++i) {
    producers.push_back(addUnary(args[0], tt, 0));
  }
  // Consume in reverse so every producer stays live through the last creation.
  llvm::SmallVector<mlir::Value> rets;
  for (int i = kNumProducers - 1; i >= 0; --i) {
    rets.push_back(addUnary(producers[i]->getResult(0), tt, 0)->getResult(0));
  }
  finishFunc(rets);

  runMock();

  // A spilled producer keeps its L1 result type; the spill is a separate
  // ToMemoryConfig(->DRAM) op, so wasSpilled() (not resultIsL1) is the right
  // predicate for "this producer's data was pushed to DRAM".
  size_t spilledProducers = 0;
  for (auto *p : producers) {
    if (wasSpilled(p->getResult(0))) {
      ++spilledProducers;
    }
  }

  EXPECT_FALSE(mockPassFailed());
  // 4 MiB of live producers cannot fit a ~1.25 MiB budget -> the tracker must
  // spill some to DRAM.
  EXPECT_GT(countSpills(), 0u)
      << "over-subscription (4 MiB into ~1.25 MiB) must spill to DRAM";
  EXPECT_GT(spilledProducers, 0u)
      << "some producers must be spilled to DRAM under over-subscription";
}

//===----------------------------------------------------------------------===//
// SequentialChainFreesRecords
//
// Complement to the accumulation test: a long chain where each tensor dies one
// step after birth. If the tracker frees live records on tensor death
// (MockAllocatorL1Tracker::removeTensor -> liveRecords.erase), at most two
// tensors are ever live, occupancy stays ~512 KiB, and nothing leaves L1 --
// even though the chain's total footprint (10 x 256 KiB = 2.5 MiB) exceeds
// device L1. A broken free path would accumulate and force spills/demotions.
//===----------------------------------------------------------------------===//
class MockAllocatorFreeTest : public L1SpillMockAllocatorFixture {};

TEST_F(MockAllocatorFreeTest, DeadTensorsFreeRecordsNoSpill) {
  l1BudgetPerCore = 100u * kKiB * 1024u; // 100 MiB: fixture budget never binds
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto tt = tensorType(shape, makeL1Sharded(shape));

  constexpr int kChainLen = 10; // 10 x 256 KiB = 2.5 MiB total > device L1
  auto args = beginFunc({tt});
  mlir::Operation *prev = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
  for (int i = 0; i < kChainLen; ++i) {
    mlir::Value in = prev ? prev->getResult(0) : args[0];
    prev = addUnary(in, tt, 0);
    ops.push_back(prev);
  }
  finishFunc({prev->getResult(0)});

  auto [obs] = runMock();

  EXPECT_FALSE(mockPassFailed());
  EXPECT_EQ(countSpills(), 0u)
      << "sequential chain frees each tensor before the next -> no spill";
  EXPECT_TRUE(obs->demotions.empty());
  // Every op stays in L1: peak residency never exceeded device capacity because
  // dead tensors released their records.
  for (auto *op : ops) {
    EXPECT_TRUE(resultIsL1(op->getResult(0)))
        << "no op should leave L1 in a self-freeing chain";
  }
}

//===----------------------------------------------------------------------===//
// PerCoreSizeMatchesConstant
//
// Pins the calibration constant: if tt-metal changes tensor sizing, this fails
// loudly and points at the fix (update kOpL1 and re-derive the budgets above).
//===----------------------------------------------------------------------===//
class MockAllocatorCalibrationTest : public L1SpillMockAllocatorFixture {};

TEST_F(MockAllocatorCalibrationTest, PerCoreSizeMatchesConstant) {
  l1BudgetPerCore = 100u * kKiB * 1024u;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto tt = tensorType(shape, makeL1Sharded(shape));

  auto args = beginFunc({tt});
  auto *opA = addUnary(args[0], tt, 0);
  finishFunc({opA->getResult(0)});

  auto [obs] = runMock();

  ASSERT_EQ(obs->lives.size(), 1u);
  EXPECT_EQ(obs->lives.front().opL1Usage, kOpL1)
      << "mock op-model per-core L1 changed; update kOpL1 and re-derive "
         "budgets";
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
