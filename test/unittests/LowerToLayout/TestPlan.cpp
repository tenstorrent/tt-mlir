// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/LowerToLayout/Plan.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace mlir::tt::d2m {

namespace {

DenseIntElementsAttr makeCollapsedIntervals(MLIRContext &context,
                                            ArrayRef<int64_t> flattenedValues) {
  assert(flattenedValues.size() % 2 == 0);
  auto intervalType = RankedTensorType::get(
      {static_cast<int64_t>(flattenedValues.size() / 2), 2},
      IntegerType::get(&context, 64));
  return DenseIntElementsAttr::get(intervalType, flattenedValues);
}

RankedTensorType
makeTensorType(MLIRContext &context, ArrayRef<int64_t> logicalShape,
               ArrayRef<int64_t> dimAlignments,
               DenseIntElementsAttr collapsedIntervals,
               ArrayRef<int64_t> gridShape, ttcore::OOBVal oobVal,
               ttcore::MemorySpace memorySpace,
               ttcore::TensorMemoryLayout memoryLayout, Type elementType) {
  auto layout = ttcore::MetalLayoutAttr::get(&context, logicalShape,
                                             dimAlignments, collapsedIntervals,
                                             oobVal, memorySpace, memoryLayout);
  ArrayRef<int64_t> tileShape;
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = tileType.getShape();
  }
  return RankedTensorType::get(layout.getDeviceShape(gridShape, tileShape),
                               elementType, layout);
}

struct FormatRoundTripTypes {
  MLIRContext context;
  RankedTensorType scalarType;
  RankedTensorType tiledType;

  FormatRoundTripTypes() {
    context.loadDialect<ttcore::TTCoreDialect>();
    Type f32 = Float32Type::get(&context);
    scalarType = RankedTensorType::get({32, 32}, f32);
    tiledType = RankedTensorType::get({1, 1}, ttcore::TileType::get(f32));
  }
};

TilizeStep makeTilizeStep(RankedTensorType inputType,
                          RankedTensorType outputType) {
  return TilizeStep{{32, 32}, inputType, OutputBufferSpec{outputType}};
}

UntilizeStep makeUntilizeStep(RankedTensorType inputType,
                              RankedTensorType outputType) {
  return UntilizeStep{inputType, OutputBufferSpec{outputType}};
}

struct CanonicalizeTest : public ::testing::Test {
  MLIRContext context;

  CanonicalizeTest() { context.loadDialect<ttcore::TTCoreDialect>(); }
};

} // namespace

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

// Cancellation rules.

TEST(MinimizerCancelTest, TilizeUntilize) {
  FormatRoundTripTypes types;
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         makeUntilizeStep(types.tiledType, types.scalarType)};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, UntilizeTilize) {
  FormatRoundTripTypes types;
  Plan p{makeUntilizeStep(types.tiledType, types.scalarType),
         makeTilizeStep(types.scalarType, types.tiledType)};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, NoCancelWhenRoundTripChangesScalarType) {
  MLIRContext context;
  context.loadDialect<ttcore::TTCoreDialect>();
  auto f32Type = RankedTensorType::get({32, 32}, Float32Type::get(&context));
  auto bf16Type = RankedTensorType::get({32, 32}, BFloat16Type::get(&context));
  auto bfpTile =
      ttcore::TileType::get(&context, {32, 32}, ttcore::DataType::BFP_BFloat8);
  auto bfpTiledType = RankedTensorType::get({1, 1}, bfpTile);

  Plan p{
      makeTilizeStep(f32Type, bfpTiledType),
      makeUntilizeStep(bfpTiledType, bf16Type),
  };
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<TilizeStep>(result[0]));
  EXPECT_TRUE(std::holds_alternative<UntilizeStep>(result[1]));
}

TEST(MinimizerCancelTest, NoCancelWithDifferentKindBetween) {
  FormatRoundTripTypes types;
  // Tilize; Mask; Untilize is not a cancellation candidate; Mask blocks it.
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         MaskStep{ttcore::OOBVal::Zero, {4, 4}},
         makeUntilizeStep(types.tiledType, types.scalarType)};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 3u);
}

TEST(MinimizerCancelTest, ChainedPairsCollapse) {
  FormatRoundTripTypes types;
  // (Tilize; Untilize); (Tilize; Untilize) → ∅
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         makeUntilizeStep(types.tiledType, types.scalarType),
         makeTilizeStep(types.scalarType, types.tiledType),
         makeUntilizeStep(types.tiledType, types.scalarType)};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, NestedCancellationViaAdjacency) {
  FormatRoundTripTypes types;
  // Tilize; (Untilize; Tilize); Untilize → cancellations from the inside out.
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         makeUntilizeStep(types.tiledType, types.scalarType),
         makeTilizeStep(types.scalarType, types.tiledType),
         makeUntilizeStep(types.tiledType, types.scalarType)};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, ReCheckAfterEraseCollapsesOuterPair) {
  FormatRoundTripTypes types;
  // Tilize; Tilize; Untilize; Untilize — the inner (Tilize;Untilize) cancels
  // first, leaving (Tilize;Untilize) as the new adjacency that also cancels.
  // Exercises the left-neighbor re-check after erase.
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         makeTilizeStep(types.scalarType, types.tiledType),
         makeUntilizeStep(types.tiledType, types.scalarType),
         makeUntilizeStep(types.tiledType, types.scalarType)};
  EXPECT_TRUE(minimize(std::move(p)).empty());
}

TEST(MinimizerCancelTest, NoCancelAcrossReshardGap) {
  FormatRoundTripTypes types;
  // Tilize; Reshard; Untilize — the Reshard blocks the cancel pair.
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         ReshardStep{{2, 2}, {32, 32}, {}},
         makeUntilizeStep(types.tiledType, types.scalarType)};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 3u);
}

TEST(MinimizerCancelTest, SingleStepStaysSingle) {
  FormatRoundTripTypes types;
  Plan p{makeTilizeStep(types.scalarType, types.tiledType)};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 1u);
  EXPECT_TRUE(std::holds_alternative<TilizeStep>(result[0]));
}

// Fusion rules.

TEST(MinimizerFuseTest, F3ReshardMerge) {
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

TEST(MinimizerFuseTest, F3ReshardMergeKeepsLast) {
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

TEST(MinimizerFuseTest, F6MaskMerge) {
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
  FormatRoundTripTypes types;
  Plan p{makeTilizeStep(types.scalarType, types.tiledType),
         MaskStep{ttcore::OOBVal::Zero, {4, 4}}};
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 2u);
}

TEST(MinimizerFuseTest, NoFuseReshardAcrossTilize) {
  FormatRoundTripTypes types;
  // Reshard; Tilize; Reshard — the Tilize between prevents the Reshards from
  // fusing. (Commutation rules may let them merge later, but not at this pass.)
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      makeTilizeStep(types.scalarType, types.tiledType),
      ReshardStep{{4, 4}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  EXPECT_EQ(result.size(), 3u);
}

TEST(MinimizerFuseTest, F6MaskMergeKeepsLastLogicalShape) {
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
  FormatRoundTripTypes types;
  // Reshard; Tilize; Untilize; Reshard → Reshard; Reshard (after cancel) →
  // Reshard (after fuse).
  Plan p{
      ReshardStep{{2, 2}, {32, 32}, {}},
      makeTilizeStep(types.scalarType, types.tiledType),
      makeUntilizeStep(types.tiledType, types.scalarType),
      ReshardStep{{4, 4}, {32, 32}, {}},
  };
  auto result = minimize(std::move(p));
  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(result[0]));
  EXPECT_EQ(std::get<ReshardStep>(result[0]).gridShape,
            (llvm::SmallVector<int64_t>{4, 4}));
}

TEST_F(CanonicalizeTest, DetectsCollapsedIntervalOnlyMappingChange) {
  auto collapsedA = makeCollapsedIntervals(context, {0, 2, 2, 4});
  auto collapsedB = makeCollapsedIntervals(context, {0, 1, 1, 4});
  Type elemType = Float32Type::get(&context);

  RankedTensorType srcType = makeTensorType(
      context, {1, 40, 32, 128}, {1, 1, 32, 32}, collapsedA, {1, 1},
      ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
      ttcore::TensorMemoryLayout::Sharded, elemType);
  RankedTensorType tgtType = makeTensorType(
      context, {1, 40, 32, 128}, {1, 1, 32, 32}, collapsedB, {1, 1},
      ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
      ttcore::TensorMemoryLayout::Sharded, elemType);

  Plan plan =
      canonicalize(PlanState{srcType}, PlanState{tgtType}, {8, 8}, &context);
  ASSERT_EQ(plan.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(plan[0]));
  EXPECT_EQ(std::get<ReshardStep>(plan[0]).collapsedIntervals, collapsedB);
}

TEST_F(CanonicalizeTest, BfpBridgeRoundTripDoesNotCancel) {
  auto collapsed = makeCollapsedIntervals(context, {0, 1, 1, 2});
  RankedTensorType srcType = makeTensorType(
      context, {256, 256}, {256, 256}, collapsed, {8, 8}, ttcore::OOBVal::Undef,
      ttcore::MemorySpace::DeviceL1, ttcore::TensorMemoryLayout::Sharded,
      Float32Type::get(&context));
  Type bfpTile =
      ttcore::TileType::get(&context, {32, 32}, ttcore::DataType::BFP_BFloat8);
  RankedTensorType tgtType =
      makeTensorType(context, {256, 256}, {128, 128}, collapsed, {4, 4},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, bfpTile);

  Plan plan = minimize(
      canonicalize(PlanState{srcType}, PlanState{tgtType}, {8, 8}, &context));
  ASSERT_EQ(plan.size(), 4u);
  EXPECT_TRUE(std::holds_alternative<TilizeStep>(plan[0]));
  ASSERT_TRUE(std::holds_alternative<UntilizeStep>(plan[1]));
  EXPECT_EQ(std::get<UntilizeStep>(plan[1]).output.type.getElementType(),
            BFloat16Type::get(&context));
  EXPECT_TRUE(std::holds_alternative<ReshardStep>(plan[2]));
  EXPECT_TRUE(std::holds_alternative<TilizeStep>(plan[3]));
}

TEST_F(CanonicalizeTest, CollapseOnlyPhysicalNoopErasesToLayout) {
  auto collapsed = ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
      &context, /*rank=*/2);
  auto uncollapsed = makeCollapsedIntervals(context, {});
  Type elemType = ttcore::TileType::get(Float32Type::get(&context));

  RankedTensorType srcType =
      makeTensorType(context, {32, 32}, {32, 32}, collapsed, {1, 1},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, elemType);
  RankedTensorType tgtType =
      makeTensorType(context, {32, 32}, {32, 32}, uncollapsed, {1, 1},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, elemType);

  ASSERT_EQ(srcType.getShape(), tgtType.getShape());
  Plan plan =
      canonicalize(PlanState{srcType}, PlanState{tgtType}, {8, 8}, &context);
  EXPECT_TRUE(plan.empty());
}

TEST_F(CanonicalizeTest, HostToDeviceVirtualBounceUsesGridAwareAlignments) {
  auto collapsed = ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
      &context, /*rank=*/2);
  Type tiledElemType = ttcore::TileType::get(Float32Type::get(&context));
  RankedTensorType systemType =
      RankedTensorType::get({32, 5120}, tiledElemType);
  RankedTensorType targetType =
      makeTensorType(context, {32, 5120}, {32, 256}, collapsed, {1, 16},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, tiledElemType);

  Plan plan = canonicalize(PlanState{systemType}, PlanState{targetType}, {8, 8},
                           &context);
  ASSERT_FALSE(plan.empty());
  auto *step = std::get_if<HostToBounceBufferStep>(&plan.front());
  ASSERT_NE(step, nullptr);

  auto bouncedLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(step->output.type.getEncoding());
  auto expectedAlignments =
      ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
          bouncedLayout.getLogicalShape(), {8, 8},
          ttcore::MetalLayoutAttr::normalizeAndFlattenIntervals(
              collapsed, bouncedLayout.getLogicalShape().size()));

  EXPECT_EQ(bouncedLayout.getMemorySpace(), ttcore::MemorySpace::DeviceDRAM);
  EXPECT_EQ(bouncedLayout.getMemoryLayout(),
            ttcore::TensorMemoryLayout::Interleaved);
  EXPECT_EQ(bouncedLayout.getDimAlignments(),
            ArrayRef<int64_t>(expectedAlignments));
  EXPECT_FALSE(static_cast<bool>(step->output.vgmForward));
  EXPECT_FALSE(static_cast<bool>(step->output.vgmInverse));
}

TEST_F(CanonicalizeTest, OobOnlyChangeProducesReinterpretThenMask) {
  auto collapsed = ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
      &context, /*rank=*/2);
  Type elemType = ttcore::TileType::get(Float32Type::get(&context));
  RankedTensorType srcType =
      makeTensorType(context, {32, 33}, {32, 32}, collapsed, {1, 1},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, elemType);
  RankedTensorType tgtType =
      makeTensorType(context, {32, 33}, {32, 32}, collapsed, {1, 1},
                     ttcore::OOBVal::Zero, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, elemType);

  Plan plan =
      canonicalize(PlanState{srcType}, PlanState{tgtType}, {8, 8}, &context);
  ASSERT_EQ(plan.size(), 2u);
  ASSERT_TRUE(std::holds_alternative<ReinterpretLayoutStep>(plan[0]));
  ASSERT_TRUE(std::holds_alternative<MaskStep>(plan[1]));
  EXPECT_EQ(std::get<MaskStep>(plan[1]).oobVal, ttcore::OOBVal::Zero);
}

TEST_F(CanonicalizeTest, RemapOnlyChangeProducesExplicitRemapStep) {
  auto collapsed = ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
      &context, /*rank=*/2);
  Type elemType = Float32Type::get(&context);
  RankedTensorType type =
      makeTensorType(context, {64, 64}, {32, 32}, collapsed, {2, 2},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, elemType);

  AffineExpr d0 = getAffineDimExpr(0, &context);
  AffineExpr d1 = getAffineDimExpr(1, &context);
  AffineMap remap = AffineMap::get(2, 0, {d1, d0}, &context);

  Plan plan =
      canonicalize(PlanState{type}, PlanState{type, remap}, {8, 8}, &context);
  ASSERT_EQ(plan.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<RemapStep>(plan[0]));
  EXPECT_EQ(std::get<RemapStep>(plan[0]).remapping, remap);
}

TEST_F(CanonicalizeTest, VgmOnlyChangeProducesRebufferStep) {
  auto collapsed = ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
      &context, /*rank=*/2);
  Type elemType = Float32Type::get(&context);
  RankedTensorType type =
      makeTensorType(context, {64, 64}, {32, 32}, collapsed, {2, 2},
                     ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
                     ttcore::TensorMemoryLayout::Sharded, elemType);

  AffineExpr d0 = getAffineDimExpr(0, &context);
  AffineExpr d1 = getAffineDimExpr(1, &context);
  AffineMap vgm = AffineMap::get(2, 0, {d0, d1}, &context);

  Plan plan = canonicalize(PlanState{type}, PlanState{type, {}, vgm, vgm},
                           {8, 8}, &context);
  ASSERT_EQ(plan.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<RebufferStep>(plan[0]));
  EXPECT_EQ(std::get<RebufferStep>(plan[0]).output.vgmForward, vgm);
  EXPECT_EQ(std::get<RebufferStep>(plan[0]).output.vgmInverse, vgm);
}

TEST_F(CanonicalizeTest, HostReturnDefersVgmClearUntilGridCollapse) {
  auto uncollapsed = makeCollapsedIntervals(context, {});
  Type elemType = Float32Type::get(&context);
  RankedTensorType srcType = makeTensorType(
      context, {1, 136, 2048}, {1, 32, 256}, uncollapsed, {1, 1, 64},
      ttcore::OOBVal::Undef, ttcore::MemorySpace::DeviceL1,
      ttcore::TensorMemoryLayout::Sharded, elemType);
  RankedTensorType tgtType = RankedTensorType::get({1, 136, 2048}, elemType);

  AffineExpr d0 = getAffineDimExpr(0, &context);
  AffineExpr d1 = getAffineDimExpr(1, &context);
  AffineExpr d2 = getAffineDimExpr(2, &context);
  AffineExpr d3 = getAffineDimExpr(3, &context);
  AffineExpr d4 = getAffineDimExpr(4, &context);
  AffineExpr d5 = getAffineDimExpr(5, &context);
  AffineMap vgm = AffineMap::get(3, 0, {d0, d1, d2}, &context);
  AffineMap remap = AffineMap::get(6, 0, {d0, d1, d2, d3, d4, d5}, &context);

  Plan plan = canonicalize(PlanState{srcType, remap, vgm, vgm},
                           PlanState{tgtType}, {8, 8}, &context);
  ASSERT_EQ(plan.size(), 2u);
  ASSERT_TRUE(std::holds_alternative<ReshardStep>(plan[0]));
  ASSERT_TRUE(std::holds_alternative<DeviceToHostStep>(plan[1]));
}

} // namespace mlir::tt::d2m
