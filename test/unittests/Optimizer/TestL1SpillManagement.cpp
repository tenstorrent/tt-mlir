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
