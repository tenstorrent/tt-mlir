// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DataflowPlanning.h"

#include <gtest/gtest.h>

namespace mlir::tt::d2m {
namespace {

DataflowKernelVariant
makeVariant(DataflowCandidateId member, llvm::SmallVector<int64_t> grid,
            KernelResourceEstimate resources, uint64_t computeCycles,
            uint64_t dataMovementCycles, float confidence) {
  KernelCostEstimate cost;
  cost.computeCycles = computeCycles;
  cost.dataMovementCycles = dataMovementCycles;
  cost.confidence = confidence;
  return DataflowKernelVariant(/*variantId=*/0, {member}, std::move(grid), {},
                               resources, cost);
}

TEST(DataflowPlanningTest, AggregatesTemporalPlanCost) {
  DataflowCandidateGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  kernels.push_back(
      {makeVariant(/*member=*/0, {1, 2},
                   {/*l1BytesPerCore=*/100, /*dramBytes=*/10,
                    /*nocBytes=*/20, /*cbCount=*/2, /*dstTiles=*/1},
                   /*computeCycles=*/100, /*dataMovementCycles=*/50,
                   /*confidence=*/0.8F),
       {}});
  kernels.push_back(
      {makeVariant(/*member=*/1, {2, 2},
                   {/*l1BytesPerCore=*/200, /*dramBytes=*/30,
                    /*nocBytes=*/40, /*cbCount=*/3, /*dstTiles=*/2},
                   /*computeCycles=*/60, /*dataMovementCycles=*/80,
                   /*confidence=*/0.6F),
       {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(kernels),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_EQ(cost.latencyCycles, 180u);
  EXPECT_FALSE(cost.initiationIntervalCycles.has_value());
  EXPECT_EQ(cost.dramBytes, 40u);
  EXPECT_EQ(cost.nocBytes, 60u);
  EXPECT_EQ(cost.peakL1BytesPerCore, 200u);
  EXPECT_EQ(cost.occupiedCores, 4u);
  EXPECT_EQ(cost.programCount, 2u);
  EXPECT_FLOAT_EQ(cost.confidence, 0.6F);
}

TEST(DataflowPlanningTest, AggregatesSpatialPlanInitiationInterval) {
  DataflowCandidateGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  kernels.push_back(
      {makeVariant(/*member=*/0, {1, 2}, {}, /*computeCycles=*/100,
                   /*dataMovementCycles=*/50, /*confidence=*/0.8F),
       {0, 0}});
  kernels.push_back(
      {makeVariant(/*member=*/1, {2, 2}, {}, /*computeCycles=*/60,
                   /*dataMovementCycles=*/80, /*confidence=*/0.6F),
       {1, 0}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Spatial, std::move(kernels),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_FALSE(cost.latencyCycles.has_value());
  EXPECT_EQ(cost.initiationIntervalCycles, 100u);
  EXPECT_EQ(cost.occupiedCores, 6u);
  EXPECT_FLOAT_EQ(cost.confidence, 0.6F);
}

TEST(DataflowPlanningTest, LeavesCyclesUnknownWithoutKernelEstimate) {
  DataflowCandidateGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  kernels.push_back(
      {DataflowKernelVariant(/*variantId=*/0, {/*members=*/0}, {1, 1}, {}),
       {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(kernels),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_FALSE(cost.latencyCycles.has_value());
  EXPECT_FALSE(cost.initiationIntervalCycles.has_value());
  EXPECT_FLOAT_EQ(cost.confidence, 0.0F);
}

TEST(DataflowPlanningTest, RejectsUnknownCandidateReference) {
  DataflowCandidateGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  kernels.push_back(
      {DataflowKernelVariant(/*variantId=*/0, {/*members=*/0}, {1, 1}, {}),
       {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(kernels),
                           {});

  DataflowFeasibilityResult result =
      StructuralDataflowFeasibilityModel().evaluate(graph, plan);

  EXPECT_FALSE(result.feasible);
  EXPECT_EQ(result.rejectionKind, DataflowRejectionKind::InvalidVariant);
}

TEST(DataflowPlanningTest, RejectsInvalidCandidateDependency) {
  DataflowCandidateGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {{0, 1}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, {}, {});

  DataflowFeasibilityResult result =
      StructuralDataflowFeasibilityModel().evaluate(graph, plan);

  EXPECT_FALSE(result.feasible);
  EXPECT_EQ(result.rejectionKind, DataflowRejectionKind::InvalidGraph);
}

TEST(DataflowPlanningTest, RejectsReorderedTemporalDependency) {
  llvm::SmallVector<GenericOp> operations(2);
  DataflowCandidateGraph graph({}, nullptr, /*scopeOrdinal=*/0,
                               std::move(operations), {{0, 1}});
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  kernels.push_back(
      {DataflowKernelVariant(/*variantId=*/0, {/*members=*/1}, {1, 1}, {}),
       {}});
  kernels.push_back(
      {DataflowKernelVariant(/*variantId=*/0, {/*members=*/0}, {1, 1}, {}),
       {}});
  DataflowMappingPlan plan(
      {}, nullptr, /*scopeOrdinal=*/0, DataflowPlanStrategy::Temporal,
      std::move(kernels),
      {{/*producerKernel=*/1, /*consumerKernel=*/0,
        DataflowConnectionKind::Materialized, /*bufferDepth=*/1}});

  DataflowFeasibilityResult result =
      StructuralDataflowFeasibilityModel().evaluate(graph, plan);

  EXPECT_FALSE(result.feasible);
  EXPECT_EQ(result.rejectionKind, DataflowRejectionKind::InvalidGraph);
}

} // namespace
} // namespace mlir::tt::d2m
