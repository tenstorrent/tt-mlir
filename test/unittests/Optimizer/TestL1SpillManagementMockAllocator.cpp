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

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"

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

//===----------------------------------------------------------------------===//
// View-op tripwires (https://github.com/tenstorrent/tt-mlir/issues/9054)
//
// The L1 spill pass classifies certain ops (reshape/pad/repeat/permute) as
// input-aliasing views via `isAliasingViewOp` and aliases them onto their
// source's L1 slot instead of tracking a fresh allocation. tt-metal signals
// such a view by reporting the op's output buffer at the weightless input's
// address 0 under the stateful op-model query. These tripwires pin that
// contract: for each op our predicate calls a view, the mock op-model query
// MUST report an L1 output allocation at address 0. If tt-metal drifts (a
// classified-view op starts returning a real non-zero address), the tripwire
// fails -- our `isAliasingViewOp` is then stale and must be adjusted, because
// the spill pass would otherwise alias (undercount) a real allocation.
//===----------------------------------------------------------------------===//

// True if the op's stateful op-model query (empty live set = the op's own
// query) reports an output buffer placed in L1 at address 0.
static bool queryReportsL1Address0(mlir::Operation *op,
                                   mlir::tt::ttnn::TTNNLayoutAttr inLayout,
                                   mlir::tt::ttnn::TTNNLayoutAttr outLayout) {
  auto result = mlir::tt::ttnn::op_constraint_validation::validateOperation(
      op, /*inputLayouts=*/{inLayout}, mlir::tt::ttnn::OpConfig(outLayout),
      /*liveRecords=*/
      llvm::ArrayRef<mlir::tt::ttnn::op_model::OpModelAllocationRecord>{},
      /*additionalL1Usage=*/0);
  EXPECT_TRUE(result.isSuccess()) << "stateful op-model query should succeed";
  for (const auto &r : result.outputAllocations) {
    if (r.bufferType == mlir::tt::ttnn::BufferType::L1 && r.address == 0) {
      return true;
    }
  }
  return false;
}

class MockAllocatorViewTripwireTest : public L1SpillMockAllocatorFixture {};

// Pad: the FillCacheInputPadRewritePattern scrub pad from the llama_3_2_1b
// repro (seq_len 18 -> 32, TILE fast-path view).
TEST_F(MockAllocatorViewTripwireTest, PadViewReportsL1Address0) {
  llvm::SmallVector<int64_t> inShape = {1, 8, 18, 64};
  llvm::SmallVector<int64_t> outShape = {1, 8, 32, 64};
  auto inLayout = makeL1Interleaved(inShape);
  auto outLayout = makeL1Interleaved(outShape);
  auto args = beginFunc({tensorType(inShape, inLayout)});
  auto *pad = addPad(args[0], tensorType(outShape, outLayout),
                     /*padding=*/{0, 0, 0, 0, 0, 14, 0, 0});
  finishFunc({pad->getResult(0)});

  EXPECT_TRUE(mlir::tt::ttnn::canPadBeView(pad));
  EXPECT_TRUE(mlir::tt::ttnn::isAliasingViewOp(pad));
  EXPECT_TRUE(queryReportsL1Address0(pad, inLayout, outLayout))
      << "TILE-view ttnn.pad must report an L1 output at address 0";
}

// Repeat: an all-ones repetition is a strict no-op (repeat.cpp:196).
TEST_F(MockAllocatorViewTripwireTest, RepeatAllOnesViewReportsL1Address0) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 64};
  auto layout = makeL1Interleaved(shape);
  auto tt = tensorType(shape, layout);
  auto args = beginFunc({tt});
  auto *repeat = addRepeat(args[0], tt, /*repeatDims=*/{1, 1, 1, 1});
  finishFunc({repeat->getResult(0)});

  EXPECT_TRUE(mlir::tt::ttnn::canRepeatBeView(repeat));
  EXPECT_TRUE(mlir::tt::ttnn::isAliasingViewOp(repeat));
  EXPECT_TRUE(queryReportsL1Address0(repeat, layout, layout))
      << "all-ones ttnn.repeat must report an L1 output at address 0";
}

// Permute: an identity permutation with unchanged memory config is a no-op.
TEST_F(MockAllocatorViewTripwireTest, PermuteNopViewReportsL1Address0) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 64};
  auto layout = makeL1Interleaved(shape);
  auto tt = tensorType(shape, layout);
  auto args = beginFunc({tt});
  auto *permute = addPermute(args[0], tt, /*permutation=*/{0, 1, 2, 3});
  finishFunc({permute->getResult(0)});

  EXPECT_TRUE(mlir::tt::ttnn::canPermuteBeView(permute));
  EXPECT_TRUE(mlir::tt::ttnn::isAliasingViewOp(permute));
  EXPECT_TRUE(queryReportsL1Address0(permute, layout, layout))
      << "identity ttnn.permute must report an L1 output at address 0";
}

// Negative: a real (non-view) pad that grows a tile dim must NOT be classified
// a view.
TEST_F(MockAllocatorViewTripwireTest, RealPadIsNotAView) {
  llvm::SmallVector<int64_t> inShape = {1, 1, 32, 64};
  llvm::SmallVector<int64_t> outShape = {1, 1, 64, 64}; // +1 full tile row
  auto args = beginFunc({tensorType(inShape, makeL1Interleaved(inShape))});
  auto *pad = addPad(args[0], tensorType(outShape, makeL1Interleaved(outShape)),
                     /*padding=*/{0, 0, 0, 0, 0, 32, 0, 0});
  finishFunc({pad->getResult(0)});

  EXPECT_FALSE(mlir::tt::ttnn::canPadBeView(pad))
      << "a pad that adds a full tile row is a real allocation, not a view";
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MockDeviceEnvironment());
  return RUN_ALL_TESTS();
}
