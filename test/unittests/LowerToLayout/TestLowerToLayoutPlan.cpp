// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/LowerToLayoutPlan.h"

#include "gtest/gtest.h"

namespace mlir::tt::d2m {

// Sanity tests for the Plan scaffolding.

TEST(PlanScaffoldTest, StepVariantHoldsKind) {
  Step s = TilizeStep{{32, 32}};
  ASSERT_TRUE(std::holds_alternative<TilizeStep>(s));
  EXPECT_EQ(std::get<TilizeStep>(s).tileShape,
            (llvm::SmallVector<int64_t>{32, 32}));
}

TEST(PlanScaffoldTest, EmptyPlanMinimizesToEmpty) {
  Plan empty;
  EXPECT_TRUE(minimize(std::move(empty)).empty());
}

TEST(PlanScaffoldTest, CanonicalizeStubReturnsEmpty) {
  PlanState src{}, tgt{};
  EXPECT_TRUE(canonicalize(src, tgt, /*targetGridShape=*/{}).empty());
}

// Cancellation rules.

TEST(MinimizerCancelTest, TilizeUntilize) {
  Plan p{TilizeStep{{32, 32}}, UntilizeStep{}};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, UntilizeTilize) {
  Plan p{UntilizeStep{}, TilizeStep{{32, 32}}};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, NoCancelWithDifferentKindBetween) {
  // Tilize; Mask; Untilize is not a cancellation candidate; Mask blocks it.
  Plan p{TilizeStep{{32, 32}}, MaskStep{ttcore::OOBVal::Zero, {4, 4}},
         UntilizeStep{}};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 3u);
}

TEST(MinimizerCancelTest, ChainedPairsCollapse) {
  // (Tilize; Untilize); (Tilize; Untilize) → ∅
  Plan p{TilizeStep{{32, 32}}, UntilizeStep{}, TilizeStep{{32, 32}},
         UntilizeStep{}};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, NestedCancellationViaAdjacency) {
  // Tilize; (Untilize; Tilize); Untilize → cancellations from the inside out.
  Plan p{TilizeStep{{32, 32}}, UntilizeStep{}, TilizeStep{{32, 32}},
         UntilizeStep{}};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, ReCheckAfterEraseCollapsesOuterPair) {
  // Tilize; Tilize; Untilize; Untilize — the inner (Tilize;Untilize) cancels
  // first, leaving (Tilize;Untilize) as the new adjacency that also cancels.
  // Exercises the left-neighbor re-check after erase.
  Plan p{TilizeStep{{32, 32}}, TilizeStep{{32, 32}}, UntilizeStep{},
         UntilizeStep{}};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, NoCancelAcrossReshardGap) {
  // Tilize; Reshard; Untilize — the Reshard blocks the cancel pair.
  Plan p{TilizeStep{{32, 32}}, ReshardStep{{2, 2}, {32, 32}, {}},
         UntilizeStep{}};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 3u);
}

TEST(MinimizerCancelTest, SingleStepStaysSingle) {
  Plan p{TilizeStep{{32, 32}}};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 1u);
  EXPECT_TRUE(std::holds_alternative<TilizeStep>(result[0]));
}

// Fusion rules.

TEST(MinimizerFuseTest, F3_ReshardMerge) {
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      ReshardStep{{4, 4}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(result[0]));
  EXPECT_EQ(std::get<ReshardStep>(result[0]).gridShape,
            (llvm::SmallVector<int64_t>{4, 4}));
}

TEST(MinimizerFuseTest, F3_ReshardMergeKeepsLast) {
  // Three reshards collapse to the final one.
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      ReshardStep{{4, 4}, {32, 32}, {}},
      ReshardStep{{8, 8}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(std::get<ReshardStep>(result[0]).gridShape,
            (llvm::SmallVector<int64_t>{8, 8}));
}

TEST(MinimizerFuseTest, F6_MaskMerge) {
  Plan p{
      MaskStep{ttcore::OOBVal::Zero, {4, 4}},
      MaskStep{ttcore::OOBVal::NegInf, {4, 4}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<MaskStep>(result[0]));
  EXPECT_EQ(std::get<MaskStep>(result[0]).oobVal, ttcore::OOBVal::NegInf);
}

TEST(MinimizerFuseTest, NoFuseBetweenUnrelatedKinds) {
  Plan p{TilizeStep{{32, 32}}, MaskStep{ttcore::OOBVal::Zero, {4, 4}}};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 2u);
}

TEST(MinimizerFuseTest, NoFuseReshardAcrossTilize) {
  // Reshard; Tilize; Reshard — the Tilize between prevents the Reshards from
  // fusing. (Commutation rules may let them merge later, but not at this pass.)
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      TilizeStep{{32, 32}},
      ReshardStep{{4, 4}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 3u);
}

TEST(MinimizerFuseTest, F6_MaskMergeKeepsLastLogicalShape) {
  // When two masks fuse, the second's payload wins (oobVal + logicalShape).
  Plan p{
      MaskStep{ttcore::OOBVal::Zero, {4, 4}},
      MaskStep{ttcore::OOBVal::NegInf, {8, 8}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 1u);
  auto &m = std::get<MaskStep>(result[0]);
  EXPECT_EQ(m.oobVal, ttcore::OOBVal::NegInf);
  EXPECT_EQ(m.logicalShape, (llvm::SmallVector<int64_t>{8, 8}));
}

// Commutation rules (only applied when they enable further simplification).

TEST(MinimizerCommuteTest, MaskAcrossReshardEnablesMaskFusion) {
  // Mask; Reshard; Mask → (commute) → Reshard; Mask; Mask → (fuse F6) →
  // Reshard; Mask.
  Plan p{
      MaskStep{ttcore::OOBVal::Zero, {4, 4}},
      ReshardStep{{2, 2}, {32, 32}, {}},
      MaskStep{ttcore::OOBVal::NegInf, {4, 4}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<ReshardStep>(result[0]));
  ASSERT_TRUE(std::holds_alternative<MaskStep>(result[1]));
  EXPECT_EQ(std::get<MaskStep>(result[1]).oobVal, ttcore::OOBVal::NegInf);
}

TEST(MinimizerCommuteTest, ReshardAcrossMaskEnablesReshardFusion) {
  // Reshard; Mask; Reshard → (commute) → Mask; Reshard; Reshard → (fuse F3) →
  // Mask; Reshard.
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      MaskStep{ttcore::OOBVal::Zero, {4, 4}},
      ReshardStep{{4, 4}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<MaskStep>(result[0]));
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(result[1]));
  EXPECT_EQ(std::get<ReshardStep>(result[1]).gridShape,
            (llvm::SmallVector<int64_t>{4, 4}));
}

TEST(MinimizerCommuteTest, NoCommuteWithoutSimplificationBenefit) {
  // Mask; Reshard on its own: commutation is legal but would not enable any
  // cancel or fuse. The gate prevents unmotivated swaps (and infinite flip).
  Plan p{MaskStep{ttcore::OOBVal::Zero, {4, 4}},
         ReshardStep{{2, 2}, {32, 32}, {}}};
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<MaskStep>(result[0]));
  EXPECT_TRUE(std::holds_alternative<ReshardStep>(result[1]));
}

TEST(MinimizerCommuteTest, InterleavedFuseAndCommuteFullyReduce) {
  // Mask; Reshard; Reshard; Mask — F3 collapses the Reshards, then commutation
  // enables F6 mask fusion. End state: Reshard; Mask.
  Plan p{
      MaskStep{ttcore::OOBVal::Zero, {4, 4}},
      ReshardStep{{2, 2}, {32, 32}, {}},
      ReshardStep{{4, 4}, {32, 32}, {}},
      MaskStep{ttcore::OOBVal::NegInf, {4, 4}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 2u);
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(result[0]));
  ASSERT_TRUE(std::holds_alternative<MaskStep>(result[1]));
  EXPECT_EQ(std::get<ReshardStep>(result[0]).gridShape,
            (llvm::SmallVector<int64_t>{4, 4}));
  EXPECT_EQ(std::get<MaskStep>(result[1]).oobVal, ttcore::OOBVal::NegInf);
}

// Fixpoint driver.

TEST(MinimizerFixpointTest, CancelEnablesFusion) {
  // Reshard; Tilize; Untilize; Reshard → Reshard; Reshard (after cancel) →
  // Reshard (after fuse).
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      TilizeStep{{32, 32}},
      UntilizeStep{},
      ReshardStep{{4, 4}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(result[0]));
  EXPECT_EQ(std::get<ReshardStep>(result[0]).gridShape,
            (llvm::SmallVector<int64_t>{4, 4}));
}

} // namespace mlir::tt::d2m
