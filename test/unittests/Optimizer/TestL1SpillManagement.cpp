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

//===----------------------------------------------------------------------===//
// NotImplementedPostFlushTest
//===----------------------------------------------------------------------===//

class NotImplementedPostFlushTest : public L1SpillTestFixture {};

// After evictAllFromL1 resets the tracker, a subsequent allocation must
// land at a fresh top-of-budget address as if no prior tensors existed.
// This is the runtime check for Task 1.5's invariant (spillToDramBeforeTrigger
// must be paired with a full tracker reset).
TEST_F(NotImplementedPostFlushTest, PostFlushAllocationSeesEmptyTracker) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA         = addUnary(args[0], tt, /*l1UsageBytes=*/600 * kKiB);
  auto *opB         = addUnary(args[0], tt, /*l1UsageBytes=*/400 * kKiB);
  auto *opC         = addBinary(opA->getResult(0), opB->getResult(0), tt,
                                 /*l1UsageBytes=*/0);
  forceNotImplemented(opC);
  // After the NotImplemented flush, allocate 1 MiB — this is ~77% of the
  // 1.3 MiB budget. If the tracker still thinks A and B are live, this
  // would push occupied past budget and OOM. With a correct reset, it fits.
  auto *opPostFlush    = addUnary(args[0], tt, /*l1UsageBytes=*/1000 * kKiB);
  auto *usePostFlush   = addUnary(opPostFlush->getResult(0), tt,
                                   /*l1UsageBytes=*/50 * kKiB);
  finishFunc({opC->getResult(0), usePostFlush->getResult(0)});

  auto [obs] = run();

  // A and B must have been spilled by evictAllFromL1.
  EXPECT_TRUE(wasSpilled(opA->getResult(0)));
  EXPECT_TRUE(wasSpilled(opB->getResult(0)));
  // opPostFlush must NOT be spilled — the reset tracker has 1.3 MiB free.
  EXPECT_FALSE(wasSpilled(opPostFlush->getResult(0)))
      << "post-flush allocation should fit on a fully-reset tracker";
  // Exactly the 2 evict-all spills, no extras from opPostFlush.
  EXPECT_EQ(countSpills(), 2u);
  EXPECT_EQ(obs->evictions.size(), 2u);
}

//===----------------------------------------------------------------------===//
// SnapshotReplayManyAllocsTest
//===----------------------------------------------------------------------===//

class SnapshotReplayManyAllocsTest : public L1SpillTestFixture {};

// Evict opA (allocated first; pos 0). 3 subsequent allocations (B, C, D)
// must be replayed forward from the empty-tracker snapshot. If the replay
// machinery is broken, the pass crashes inside markEvictedAndRebuild.
TEST_F(SnapshotReplayManyAllocsTest, EvictEarliestTensorReplaysAllSuccessors) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA = addUnary(args[0], tt, /*l1UsageBytes=*/200 * kKiB);
  auto *opB = addUnary(args[0], tt, /*l1UsageBytes=*/200 * kKiB);
  auto *opC = addUnary(args[0], tt, /*l1UsageBytes=*/200 * kKiB);
  auto *opD = addUnary(args[0], tt, /*l1UsageBytes=*/200 * kKiB);
  auto *opPressure = addUnary(args[0], tt, /*l1UsageBytes=*/600 * kKiB);
  // useD, useC, useB, useA in reverse order so opA's lastUse is the
  // farthest — eviction picks opA deterministically.
  auto *useD = addUnary(opD->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  auto *useC = addUnary(opC->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  auto *useB = addUnary(opB->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  auto *useA = addUnary(opA->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  finishFunc({opPressure->getResult(0), useD->getResult(0),
              useC->getResult(0), useB->getResult(0), useA->getResult(0)});

  auto [obs] = run();

  ASSERT_EQ(obs->evictions.size(), 1u)
      << "expected exactly one eviction (opA, allocated earliest)";
  EXPECT_EQ(obs->evictions[0].victim, opA);
  EXPECT_TRUE(wasSpilled(opA->getResult(0)));
  // B, C, D must NOT be spilled — replay reproduced their addresses correctly.
  EXPECT_FALSE(wasSpilled(opB->getResult(0)));
  EXPECT_FALSE(wasSpilled(opC->getResult(0)));
  EXPECT_FALSE(wasSpilled(opD->getResult(0)));
}

//===----------------------------------------------------------------------===//
// SnapshotReplayCrossEvictionTest
//===----------------------------------------------------------------------===//

class SnapshotReplayCrossEvictionTest : public L1SpillTestFixture {};

// Two evictions at consecutive positions. Eviction 2 restores from a
// snapshot that was UPDATED by eviction 1's replay loop. If the
// implementation forgot to refresh snapshots during replay, eviction 2
// produces wrong free-list state and downstream allocations either crash
// or land at wrong addresses.
//
// Critical: opP1 and opP2 each need a downstream consumer (useP1, useP2) —
// func.return is NOT in the schedule, so values consumed only by it have
// lastUse = their own creation position and die immediately. The consumers
// also dictate the eviction order at OOM #2: we want opB evicted before
// opP1, so opB.lastUse must be FURTHER than opP1.lastUse.
TEST_F(SnapshotReplayCrossEvictionTest, SecondEvictionUsesUpdatedSnapshots) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA  = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opB  = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opP1 = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opP2 = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  // Consumers in REVERSE order of producers so opA has the farthest
  // last-use (→ evicted first), then opB, then opP1, then opP2.
  auto *useP2 = addUnary(opP2->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  auto *useP1 = addUnary(opP1->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  auto *useB  = addUnary(opB->getResult(0),  tt, /*l1UsageBytes=*/50 * kKiB);
  auto *useA  = addUnary(opA->getResult(0),  tt, /*l1UsageBytes=*/50 * kKiB);
  finishFunc({useP2->getResult(0), useP1->getResult(0),
              useB->getResult(0),  useA->getResult(0)});

  auto [obs] = run();

  ASSERT_EQ(obs->evictions.size(), 2u)
      << "expected two evictions (one at each pressure op)";
  EXPECT_EQ(obs->evictions[0].victim, opA) << "first eviction = farthest = opA";
  EXPECT_EQ(obs->evictions[1].victim, opB) << "second eviction = opB";
  EXPECT_TRUE(wasSpilled(opA->getResult(0)));
  EXPECT_TRUE(wasSpilled(opB->getResult(0)));
}

//===----------------------------------------------------------------------===//
// SnapshotReplayNonEmptyAndCascadeTest
//===----------------------------------------------------------------------===//

class SnapshotReplayNonEmptyAndCascadeTest : public L1SpillTestFixture {};

// Two evictions exercise both halves of the snapshot mechanism:
//   1. First eviction restores a NON-EMPTY snapshot (opA and opB exist
//      before opVictim).
//   2. Second eviction (opB) is allocated BEFORE opVictim, so its
//      pre-alloc snapshot is even smaller. The replay forward from B's
//      alloc index must walk past opVictim's already-skipped events
//      (preserving them as skipped) and re-allocate Trigger1 and Post.
TEST_F(SnapshotReplayNonEmptyAndCascadeTest,
       NonEmptySnapshotPlusCascadeStaysConsistent) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA        = addUnary(args[0], tt, /*l1UsageBytes=*/300 * kKiB);
  auto *opB        = addUnary(args[0], tt, /*l1UsageBytes=*/300 * kKiB);
  auto *opVictim   = addUnary(args[0], tt, /*l1UsageBytes=*/300 * kKiB);
  auto *opTrigger1 = addUnary(args[0], tt, /*l1UsageBytes=*/600 * kKiB);
  auto *opPost     = addUnary(args[0], tt, /*l1UsageBytes=*/100 * kKiB);
  auto *opTrigger2 = addUnary(args[0], tt, /*l1UsageBytes=*/200 * kKiB);
  // Consumers ordered so last-use is:
  //   Trigger2(6) < Trigger1(7) < Post(8) < A(9) < B(10) < Victim(11).
  // → at OOM #1, farthest is opVictim; at OOM #2, farthest among live is opB.
  auto *useTrigger2 = addUnary(opTrigger2->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  auto *useTrigger1 = addUnary(opTrigger1->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  auto *usePost     = addUnary(opPost->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  auto *useA        = addUnary(opA->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  auto *useB        = addUnary(opB->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  auto *useVictim   = addUnary(opVictim->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  finishFunc({useTrigger2->getResult(0), useTrigger1->getResult(0),
              usePost->getResult(0),     useA->getResult(0),
              useB->getResult(0),        useVictim->getResult(0)});

  auto [obs] = run();

  ASSERT_EQ(obs->evictions.size(), 2u)
      << "expected two evictions (Victim at OOM #1, B at OOM #2)";
  EXPECT_EQ(obs->evictions[0].victim, opVictim)
      << "OOM #1: farthest-last-use is opVictim";
  EXPECT_EQ(obs->evictions[1].victim, opB)
      << "OOM #2: farthest-last-use among live ops is opB";
  EXPECT_TRUE(wasSpilled(opVictim->getResult(0)));
  EXPECT_TRUE(wasSpilled(opB->getResult(0)));
  // opA must survive both evictions and stay in L1.
  EXPECT_FALSE(wasSpilled(opA->getResult(0)));
  // Trigger1 and Post must survive eviction #2 (they were re-allocated by
  // the replay forward from B's snapshot at correct addresses).
  EXPECT_FALSE(wasSpilled(opTrigger1->getResult(0)));
  EXPECT_FALSE(wasSpilled(opPost->getResult(0)));
  EXPECT_FALSE(wasSpilled(opTrigger2->getResult(0)));
}
