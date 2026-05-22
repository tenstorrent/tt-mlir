// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "L1SpillTestFixture.h"
#include "gtest/gtest.h"

using namespace mlir::tt::ttnn::test;

//===----------------------------------------------------------------------===//
// LinearChainTest
//===----------------------------------------------------------------------===//

class LinearChainTest : public L1SpillTestFixture {};

// Two large producers (each 800 KiB) feeding a small join. The second
// producer triggers OOM; farthest-last-use eviction picks opA (only live
// tensor at that moment), the pass inserts one DRAM spill, and opJoin then
// reads opA from DRAM and opB from L1.
TEST_F(LinearChainTest, SpillsFirstProducerWhenSecondCausesOOM) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA = addUnary(args[0], tt, /*l1UsageBytes=*/800 * kKiB);
  auto *opB = addUnary(args[0], tt, /*l1UsageBytes=*/800 * kKiB);
  auto *opJoin = addBinary(opA->getResult(0), opB->getResult(0), tt,
                            /*l1UsageBytes=*/100 * kKiB);
  finishFunc({opJoin->getResult(0)});

  auto [obs] = run();

  EXPECT_EQ(countSpills(), 1u)
      << "expected one spill-to-DRAM ToMemoryConfigOp";
  EXPECT_TRUE(wasSpilled(opA->getResult(0)))
      << "opA (only live tensor at OOM) should be spilled";
  ASSERT_EQ(obs->evictions.size(), 1u);
  EXPECT_EQ(obs->evictions[0].victim, opA);
}

//===----------------------------------------------------------------------===//
// FarthestLastUseOrderingTest
//===----------------------------------------------------------------------===//

class FarthestLastUseOrderingTest : public L1SpillTestFixture {};

// Two tensors live at the OOM point. opA's last consumer is at pos 4;
// opB's at pos 3. The pass must evict opA (farther last-use position).
// opPressure consumes only the function arg, so the input-overlap accounting
// in SumL1MemoryTracker::validate() does NOT subtract A or B from the
// pressure and the OOM signal makes it through.
TEST_F(FarthestLastUseOrderingTest, EvictsFarthestNotNearest) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA        = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opB        = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opPressure = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opLastB    = addUnary(opB->getResult(0), tt, /*l1UsageBytes=*/100 * kKiB);
  auto *opLastA    = addUnary(opA->getResult(0), tt, /*l1UsageBytes=*/100 * kKiB);
  finishFunc({opPressure->getResult(0), opLastB->getResult(0),
              opLastA->getResult(0)});

  auto [obs] = run();

  ASSERT_EQ(obs->evictions.size(), 1u) << "expected exactly one eviction";
  EXPECT_EQ(obs->evictions[0].victim, opA)
      << "farthest-last-use must pick opA (lastUse=4) over opB (lastUse=3)";
  EXPECT_TRUE(wasSpilled(opA->getResult(0)));
  EXPECT_FALSE(wasSpilled(opB->getResult(0)));
}

//===----------------------------------------------------------------------===//
// SelfSpillTest
//===----------------------------------------------------------------------===//

class SelfSpillTest : public L1SpillTestFixture {};

// DISABLED until rpavlovic/remove-output-l1-usage-attr lands. On that
// branch the Stage-3 OOM path switches from spillToDram to demoteToDram
// for ops whose output alone exceeds the budget. Remove the DISABLED_
// prefix after the branch merges.
TEST_F(SelfSpillTest, DISABLED_OpTooLargeForBudgetDemotes) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 2048, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  // opA output ≈ 2 MB; budget ≈ 1.3 MB → cannot fit even with empty live set.
  auto *opA = addUnary(args[0], tt, /*l1UsageBytes=*/2000 * kKiB);
  finishFunc({opA->getResult(0)});

  auto [obs] = run();

  // No farthest-last-use evictions expected (live set was empty).
  EXPECT_TRUE(obs->evictions.empty())
      << "no eviction — nothing else was live to evict";

  // After demotion, opA's result type should have a DRAM layout.
  auto rt = mlir::cast<mlir::RankedTensorType>(opA->getResult(0).getType());
  auto layout = mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(rt.getEncoding());
  EXPECT_FALSE(layout.hasL1BufferType())
      << "opA should be demoted in place to DRAM";

  // Observer should record the demotion (success=true).
  ASSERT_FALSE(obs->demotions.empty()) << "expected at least one demotion event";
  EXPECT_TRUE(obs->demotions.front().success);
}

//===----------------------------------------------------------------------===//
// NotImplementedTest
//===----------------------------------------------------------------------===//

class NotImplementedTest : public L1SpillTestFixture {};

// When an op returns NotImplemented from validation, the pass cannot know
// its L1 requirements and must conservatively evict all live tensors before
// running that op.
TEST_F(NotImplementedTest, EvictsAllLiveTensorsBeforeNotImplementedOp) {
  l1BudgetPerCore = 4000 * kKiB; // ample budget — OOM is not the trigger here
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA = addUnary(args[0], tt, /*l1UsageBytes=*/600 * kKiB);
  auto *opB = addUnary(args[0], tt, /*l1UsageBytes=*/400 * kKiB);
  // opC is NotImplemented: pass must flush A and B before processing it.
  auto *opC = addBinary(opA->getResult(0), opB->getResult(0), tt,
                        /*l1UsageBytes=*/0);
  finishFunc({opC->getResult(0)});

  forceNotImplemented(opC);
  auto [obs] = run();

  // Both opA and opB must be spilled before opC runs.
  EXPECT_TRUE(wasSpilled(opA->getResult(0)))
      << "opA should be spilled before NotImplemented opC";
  EXPECT_TRUE(wasSpilled(opB->getResult(0)))
      << "opB should be spilled before NotImplemented opC";
  EXPECT_GE(countSpills(), 2u)
      << "at least 2 spill-to-DRAM ops expected";
  // evictAllFromL1 calls onEviction for each victim.
  EXPECT_EQ(obs->evictions.size(), 2u)
      << "evictAllFromL1 should record exactly 2 eviction events (A and B)";
}
