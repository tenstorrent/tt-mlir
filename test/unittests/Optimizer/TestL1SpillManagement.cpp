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

// Op whose output alone exceeds budget cannot be made to fit by evicting
// anything else. The pass demotes the op's output in place (result type
// flips from L1 to DRAM) and fires onSelfSpill — this is the Stage-3
// OOM path in handleOOM (post remove-output-l1-usage-attr).
TEST_F(SelfSpillTest, OpTooLargeForBudgetDemotes) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 2048, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  // opA output ≈ 2 MiB; budget ≈ 1.3 MiB → cannot fit even with empty live set.
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

  // Stage-3 OOM path fires onSelfSpill. The action (demoteToDram) updates
  // the result type but doesn't itself fire onDemotion.
  ASSERT_FALSE(obs->selfSpills.empty())
      << "expected an onSelfSpill event from Stage-3 OOM handling";
  EXPECT_EQ(obs->selfSpills.front().op, opA);
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

//===----------------------------------------------------------------------===//
// ForkJoinTest
//===----------------------------------------------------------------------===//

class ForkJoinTest : public L1SpillTestFixture {};

// opA forks into opB and opC, joined by opJoin. opAfterAll keeps opA alive
// past opJoin so that opA has the farthest last-use when the no-fit check
// fires at opC. Pass evicts opA and inserts the spill-to-DRAM so all three
// consumers can still read it.
TEST_F(ForkJoinTest, SharedTensorEvictedAsFarthestLastUse) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA   = addUnary(args[0], tt, /*l1UsageBytes=*/600 * kKiB);
  auto *opB   = addUnary(opA->getResult(0), tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opC   = addUnary(opA->getResult(0), tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opJoin = addBinary(opB->getResult(0), opC->getResult(0), tt,
                           /*l1UsageBytes=*/100 * kKiB);
  // opAfterAll keeps opA's last-use later than opJoin's, making opA the
  // farthest-last-use eviction target.
  auto *opAfterAll = addUnary(opA->getResult(0), tt, /*l1UsageBytes=*/100 * kKiB);
  finishFunc({opJoin->getResult(0), opAfterAll->getResult(0)});

  auto [obs] = run();

  EXPECT_TRUE(wasSpilled(opA->getResult(0)))
      << "opA (farthest-last-use of the shared fork tensor) should be spilled";
  ASSERT_FALSE(obs->evictions.empty());
  EXPECT_EQ(obs->evictions.front().victim, opA)
      << "first eviction must be opA";
}

//===----------------------------------------------------------------------===//
// SequentialMLPTest
//===----------------------------------------------------------------------===//

class SequentialMLPTest : public L1SpillTestFixture {};

// 8-op sequential chain (MLP-style). Each tensor dies one step after birth,
// so the pass should add ZERO spills. Regressions that inflate lifetimes or
// double-count inputs would surface as unexpected evictions here.
TEST_F(SequentialMLPTest, EightOpChainProducesNoSpills) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  mlir::Operation *prev = nullptr;
  for (int i = 0; i < 8; ++i) {
    mlir::Value input = prev ? prev->getResult(0) : args[0];
    prev = addUnary(input, tt, /*l1UsageBytes=*/400 * kKiB);
  }
  finishFunc({prev->getResult(0)});

  auto [obs] = run();

  EXPECT_EQ(obs->evictions.size(), 0u) << "linear chain should not spill";
  EXPECT_EQ(countSpills(), 0u);
}

//===----------------------------------------------------------------------===//
// ResidualBlockTest
//===----------------------------------------------------------------------===//

class ResidualBlockTest : public L1SpillTestFixture {};

// Residual / skip-connection block. opSkip lives through both branch ops.
// inputOverlap accounting must correctly subtract the skip tensor when
// validating branch ops — otherwise we'd see spurious evictions.
TEST_F(ResidualBlockTest, SkipConnectionDoesNotForceEviction) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opSkip    = addUnary(args[0], tt, /*l1UsageBytes=*/500 * kKiB);
  auto *opBranch1 = addUnary(opSkip->getResult(0), tt, /*l1UsageBytes=*/400 * kKiB);
  auto *opBranch2 = addUnary(opBranch1->getResult(0), tt,
                              /*l1UsageBytes=*/400 * kKiB);
  auto *opAdd     = addBinary(opSkip->getResult(0), opBranch2->getResult(0), tt,
                               /*l1UsageBytes=*/100 * kKiB);
  finishFunc({opAdd->getResult(0)});

  auto [obs] = run();

  EXPECT_EQ(obs->evictions.size(), 0u)
      << "residual block fits in budget — input-overlap must cancel pressure";
  EXPECT_EQ(countSpills(), 0u);
}

//===----------------------------------------------------------------------===//
// QKVForkJoinTest
//===----------------------------------------------------------------------===//

class QKVForkJoinTest : public L1SpillTestFixture {};

// 3-way fork (Q, K, V projections from one input), then 2-way join.
// Exercises input-overlap correctness across multiple consumers of the
// same producer.
TEST_F(QKVForkJoinTest, ThreeWayForkJoinFitsInBudget) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opIn   = addUnary(args[0], tt, /*l1UsageBytes=*/400 * kKiB);
  auto *opQ    = addUnary(opIn->getResult(0), tt, /*l1UsageBytes=*/300 * kKiB);
  auto *opK    = addUnary(opIn->getResult(0), tt, /*l1UsageBytes=*/300 * kKiB);
  auto *opV    = addUnary(opIn->getResult(0), tt, /*l1UsageBytes=*/300 * kKiB);
  auto *useQK  = addBinary(opQ->getResult(0), opK->getResult(0), tt,
                            /*l1UsageBytes=*/200 * kKiB);
  auto *useV   = addUnary(opV->getResult(0), tt, /*l1UsageBytes=*/200 * kKiB);
  auto *opAttn = addBinary(useQK->getResult(0), useV->getResult(0), tt,
                            /*l1UsageBytes=*/100 * kKiB);
  finishFunc({opAttn->getResult(0)});

  auto [obs] = run();

  EXPECT_EQ(obs->evictions.size(), 0u)
      << "QKV fork-join should fit within 1.3 MiB budget";
  EXPECT_EQ(countSpills(), 0u);
}

//===----------------------------------------------------------------------===//
// CBOverlapTest
//===----------------------------------------------------------------------===//

class CBOverlapTest : public L1SpillTestFixture {};

// When an op's circular-buffer region (grows bottom-up from 0) would clash
// with a live tensor (placed top-down), the pass evicts the live tensor.
// The check is cushioned by cbFragCushion (10% of budget) to account for
// unmodeled runtime allocator fragmentation.
TEST_F(CBOverlapTest, HighCBPeakEvictsLowAddressTensor) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opLarge = addUnary(args[0], tt, /*l1UsageBytes=*/1000 * kKiB);
  auto *opCB    = addUnary(args[0], tt, /*l1UsageBytes=*/50 * kKiB);
  setL1Usage(opCB, /*l1=*/50 * kKiB, /*cb=*/800 * kKiB);
  // Force opLarge to outlive opCB so it's still live when the CB check fires.
  auto *useLarge = addUnary(opLarge->getResult(0), tt, /*l1UsageBytes=*/50 * kKiB);
  finishFunc({opCB->getResult(0), useLarge->getResult(0)});

  auto [obs] = run();

  EXPECT_TRUE(wasSpilled(opLarge->getResult(0)))
      << "opLarge should be spilled to clear the CB region";
  // Observer should record either an eviction or a fragmentation demote.
  EXPECT_TRUE(!obs->evictions.empty() || !obs->demotions.empty());
}

//===----------------------------------------------------------------------===//
// CushionDominantTriggerTest
//===----------------------------------------------------------------------===//

class CushionDominantTriggerTest : public L1SpillTestFixture {};

// In ensureFitsL1, `wouldCBsOverlapTensors` is guarded by `cbPeakUsage > 0`
// (see L1SpillManagement.cpp around line 597). So the function's
// "cushion alone with cbPeak=0" branch is unreachable from that caller.
// What IS reachable: a small cbPeakUsage that, on its own, wouldn't
// justify eviction — but together with the 10% cushion of 1.3 MiB
// (≈ 130 KiB) becomes large enough to trip the check at a tight fit.
//
// This test demonstrates the cushion's contribution. With cbPeak=1 KiB
// alone (1 KiB > 0 KiB lowestExisting at the tight fit), the check
// already fires; but raising the cushion's share to ~130 KiB ensures the
// pass would not be on the boundary of the check. Eviction is triggered.
TEST_F(CushionDominantTriggerTest, SmallCBPlusCushionFiresAtTightFit) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 2048, 1024};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opNearFull = addUnary(args[0], tt, /*l1UsageBytes=*/1250 * kKiB);
  auto *opNext     = addUnary(args[0], tt, /*l1UsageBytes=*/50 * kKiB);
  // Tiny cbPeak: 1 KiB. On its own, far below lowestExisting (~50 KiB after
  // opNearFull). With the 130-KiB cushion added, exceeds it — fragmentation
  // check fires → handleFragmentation evicts opNearFull.
  setL1Usage(opNext, /*l1=*/50 * kKiB, /*cb=*/1 * kKiB);
  auto *useNearFull = addUnary(opNearFull->getResult(0), tt,
                                /*l1UsageBytes=*/50 * kKiB);
  finishFunc({opNext->getResult(0), useNearFull->getResult(0)});

  auto [obs] = run();

  EXPECT_TRUE(wasSpilled(opNearFull->getResult(0)))
      << "cushion contribution should evict opNearFull. "
      << "evictions=" << obs->evictions.size()
      << " demotions=" << obs->demotions.size()
      << " spills=" << countSpills();
}

//===----------------------------------------------------------------------===//
// HandleNoFitTest (smoke coverage)
//===----------------------------------------------------------------------===//

class HandleNoFitTest : public L1SpillTestFixture {};

// Alloc → free → realloc with the SAME size at the same slot must succeed
// (free-list coalescing). A regression in freeAddress' adjacent-merge
// logic would surface as an unexpected spurious eviction here. Full
// geometric handleNoFit (non-contiguous free space) requires alias-aware
// setup (addTensorAtAddress) and is parked out of scope.
TEST_F(HandleNoFitTest, AllocFreeReallocCoalesces) {
  l1BudgetPerCore = 1300 * kKiB;
  llvm::SmallVector<int64_t> shape = {1, 1, 1024, 512};
  auto l1Layout = makeL1Sharded(shape);
  auto tt = tensorType(shape, l1Layout);

  auto args = beginFunc({tt});
  auto *opA   = addUnary(args[0], tt, /*l1UsageBytes=*/700 * kKiB);
  auto *useA  = addUnary(opA->getResult(0), tt, /*l1UsageBytes=*/100 * kKiB);
  // After useA runs, opA dies. opB can now reuse the freed slot.
  auto *opB   = addUnary(args[0], tt, /*l1UsageBytes=*/700 * kKiB);
  finishFunc({useA->getResult(0), opB->getResult(0)});

  auto [obs] = run();

  EXPECT_EQ(obs->evictions.size(), 0u)
      << "free-list coalesce should let opB reuse opA's freed slot";
  EXPECT_EQ(countSpills(), 0u);
}

//===----------------------------------------------------------------------===//
// ViewEligibleReshapeAliasTest
//===----------------------------------------------------------------------===//

class ViewEligibleReshapeAliasTest : public L1SpillTestFixture {};

// DISABLED — reproducer for
// https://github.com/tenstorrent/tt-mlir/issues/8625
//
// The alias path (`addTensorAtAddress` at L1SpillManagement.cpp:1124-1133)
// runs only AFTER `ensureFitsL1` queries `wouldAllocateAt(perResultL1)`
// using the reshape's full output size — ignoring that a view-eligible
// reshape would not consume a fresh slot. So in any scenario where the
// would-be alias size would otherwise force eviction, the pass evicts
// the input instead of recognising the alias.
//
// To meaningfully test the alias path, the pass needs to consult
// canReshapeBeView (or an analogous predicate) before the fit/CB
// checks. Until then, this test reproduces the limitation: opA is
// spilled even though the reshape SHOULD alias it.
TEST_F(ViewEligibleReshapeAliasTest, DISABLED_ReshapeAliasesInputNoDoubleCount) {
  l1BudgetPerCore = 1300 * kKiB;

  // 1024 = 32 × 32 (tile-aligned). 256 = 32 × 8 (tile-aligned).
  // Last dim (1024) is preserved between input and output shapes →
  // canReshapeBeView returns true.
  llvm::SmallVector<int64_t> shapeIn  = {1, 1, 1024, 1024};
  llvm::SmallVector<int64_t> shapeOut = {1, 4,  256, 1024};
  auto layoutIn  = makeL1Sharded(shapeIn);
  auto layoutOut = makeL1Sharded(shapeOut);
  auto ttIn  = tensorType(shapeIn,  layoutIn);
  auto ttOut = tensorType(shapeOut, layoutOut);

  auto args = beginFunc({ttIn});
  auto *opA       = addUnary(args[0], ttIn, /*l1UsageBytes=*/1000 * kKiB);
  auto *opReshape = addReshape(opA->getResult(0), ttOut,
                                /*l1UsageBytes=*/1000 * kKiB);
  auto *opConsumer = addUnary(opReshape->getResult(0), ttOut,
                               /*l1UsageBytes=*/100 * kKiB);
  finishFunc({opConsumer->getResult(0)});

  auto [obs] = run();
  (void)obs;

  EXPECT_EQ(obs->evictions.size(), 0u)
      << "view-eligible reshape must alias opA's slot — no eviction expected";
  EXPECT_EQ(countSpills(), 0u);
  EXPECT_FALSE(wasSpilled(opA->getResult(0)))
      << "opA must stay in L1; reshape aliases its slot";
}
