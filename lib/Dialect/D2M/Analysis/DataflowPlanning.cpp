// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DataflowPlanning.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace mlir::tt::d2m {

DataflowCandidateGraph::DataflowCandidateGraph(
    func::FuncOp function, Block *block, unsigned scopeOrdinal,
    llvm::SmallVector<GenericOp> operations,
    llvm::SmallVector<DataflowCandidateEdge> edges)
    : function(function), block(block), scopeOrdinal(scopeOrdinal),
      operations(std::move(operations)), edges(std::move(edges)) {}

static void collectDefiningCandidates(
    Value value, const llvm::DenseMap<Operation *, DataflowCandidateId> &ids,
    llvm::DenseSet<Value> &visited,
    llvm::DenseSet<DataflowCandidateId> &producers) {
  if (!visited.insert(value).second) {
    return;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return;
  }

  auto candidate = ids.find(definingOp);
  if (candidate != ids.end()) {
    producers.insert(candidate->second);
    return;
  }

  // Do not infer dependencies through the body of another kernel candidate.
  if (definingOp->getParentOfType<GenericOp>()) {
    return;
  }

  for (Value operand : definingOp->getOperands()) {
    collectDefiningCandidates(operand, ids, visited, producers);
  }
}

static llvm::SmallVector<DataflowCandidateEdge>
buildCandidateEdges(llvm::ArrayRef<GenericOp> operations) {
  llvm::DenseMap<Operation *, DataflowCandidateId> ids;
  for (auto item : llvm::enumerate(operations)) {
    GenericOp operation = item.value();
    ids.try_emplace(operation.getOperation(),
                    static_cast<DataflowCandidateId>(item.index()));
  }

  llvm::SmallVector<DataflowCandidateEdge> edges;
  for (auto item : llvm::enumerate(operations)) {
    DataflowCandidateId consumerId =
        static_cast<DataflowCandidateId>(item.index());
    GenericOp consumer = item.value();
    llvm::DenseSet<DataflowCandidateId> producers;
    for (Value input : consumer.getInputs()) {
      llvm::DenseSet<Value> visited;
      collectDefiningCandidates(input, ids, visited, producers);
    }
    for (DataflowCandidateId producerId : producers) {
      if (producerId != consumerId) {
        edges.push_back({producerId, consumerId});
      }
    }
  }

  llvm::sort(edges, [](const DataflowCandidateEdge &lhs,
                       const DataflowCandidateEdge &rhs) {
    return std::tie(lhs.producer, lhs.consumer) <
           std::tie(rhs.producer, rhs.consumer);
  });
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](const DataflowCandidateEdge &lhs,
                             const DataflowCandidateEdge &rhs) {
                            return lhs.producer == rhs.producer &&
                                   lhs.consumer == rhs.consumer;
                          }),
              edges.end());
  return edges;
}

llvm::SmallVector<DataflowCandidateGraph>
buildDataflowCandidateGraphs(ModuleOp module) {
  llvm::SmallVector<DataflowCandidateGraph> graphs;

  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    llvm::DenseMap<Block *, unsigned> blockToScope;
    llvm::SmallVector<Block *> blocks;
    llvm::SmallVector<llvm::SmallVector<GenericOp>> operationsByBlock;

    function.walk([&](GenericOp genericOp) {
      if (genericOp->getParentOfType<SpatialOp>() ||
          genericOp->getParentOfType<GenericOp>()) {
        return;
      }

      Block *block = genericOp->getBlock();
      auto [it, inserted] = blockToScope.try_emplace(
          block, static_cast<unsigned>(operationsByBlock.size()));
      if (inserted) {
        blocks.push_back(block);
        operationsByBlock.emplace_back();
      }
      operationsByBlock[it->second].push_back(genericOp);
    });

    for (size_t scope = 0; scope < operationsByBlock.size(); ++scope) {
      llvm::SmallVector<GenericOp> operations =
          std::move(operationsByBlock[scope]);
      llvm::SmallVector<DataflowCandidateEdge> edges =
          buildCandidateEdges(operations);
      graphs.emplace_back(function, blocks[scope], static_cast<unsigned>(scope),
                          std::move(operations), std::move(edges));
    }
  }

  return graphs;
}

DataflowKernelVariant::DataflowKernelVariant(
    unsigned variantId, llvm::SmallVector<DataflowCandidateId> members,
    llvm::SmallVector<int64_t> gridShape,
    llvm::SmallVector<int64_t> blockFactors, KernelResourceEstimate resources,
    KernelCostEstimate cost)
    : variantId(variantId), members(std::move(members)),
      gridShape(std::move(gridShape)), blockFactors(std::move(blockFactors)),
      resources(resources), cost(cost) {}

llvm::StringRef stringifyDataflowConnectionKind(DataflowConnectionKind kind) {
  switch (kind) {
  case DataflowConnectionKind::Materialized:
    return "materialized";
  case DataflowConnectionKind::L1Stream:
    return "l1-stream";
  case DataflowConnectionKind::NocStream:
    return "noc-stream";
  case DataflowConnectionKind::Dram:
    return "dram";
  }
  llvm_unreachable("unknown dataflow connection kind");
}

llvm::StringRef stringifyDataflowPlanStrategy(DataflowPlanStrategy strategy) {
  switch (strategy) {
  case DataflowPlanStrategy::Temporal:
    return "temporal-fallback";
  case DataflowPlanStrategy::Fused:
    return "fused";
  case DataflowPlanStrategy::Spatial:
    return "spatial";
  }
  llvm_unreachable("unknown dataflow plan strategy");
}

DataflowMappingPlan::DataflowMappingPlan(
    func::FuncOp function, Block *block, unsigned scopeOrdinal,
    DataflowPlanStrategy strategy,
    llvm::SmallVector<DataflowPlannedKernel, 0> kernels,
    llvm::SmallVector<DataflowPlannedConnection> connections)
    : function(function), block(block), scopeOrdinal(scopeOrdinal),
      strategy(strategy), kernels(std::move(kernels)),
      connections(std::move(connections)) {}

llvm::StringRef stringifyDataflowRejectionKind(DataflowRejectionKind kind) {
  switch (kind) {
  case DataflowRejectionKind::None:
    return "none";
  case DataflowRejectionKind::InvalidGraph:
    return "invalid-graph";
  case DataflowRejectionKind::InvalidVariant:
    return "invalid-variant";
  case DataflowRejectionKind::ResourceLimit:
    return "resource-limit";
  case DataflowRejectionKind::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown dataflow rejection kind");
}

DataflowFeasibilityResult DataflowFeasibilityResult::success() {
  return {true, DataflowRejectionKind::None, {}};
}

DataflowFeasibilityResult
DataflowFeasibilityResult::reject(DataflowRejectionKind kind,
                                  llvm::StringRef message) {
  return {false, kind, message.str()};
}

DataflowFeasibilityResult StructuralDataflowFeasibilityModel::evaluate(
    const DataflowCandidateGraph &graph,
    const DataflowMappingPlan &plan) const {
  if (plan.getFunction() != graph.getFunction() ||
      plan.getBlock() != graph.getBlock() ||
      plan.getScopeOrdinal() != graph.getScopeOrdinal()) {
    return DataflowFeasibilityResult::reject(
        DataflowRejectionKind::InvalidGraph,
        "plan identity does not match its candidate graph");
  }

  llvm::SmallVector<std::optional<unsigned>> candidateToKernel(
      graph.getOperations().size());
  for (auto [kernelIndex, kernel] : llvm::enumerate(plan.getKernels())) {
    const DataflowKernelVariant &variant = kernel.variant;
    if (variant.getMembers().empty() || variant.getGridShape().empty() ||
        llvm::any_of(variant.getGridShape(),
                     [](int64_t dim) { return dim <= 0; })) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidVariant,
          "kernel variant must have members and a positive grid shape");
    }

    for (DataflowCandidateId member : variant.getMembers()) {
      if (member >= candidateToKernel.size()) {
        return DataflowFeasibilityResult::reject(
            DataflowRejectionKind::InvalidVariant,
            "kernel variant references an unknown candidate");
      }
      if (candidateToKernel[member]) {
        return DataflowFeasibilityResult::reject(
            DataflowRejectionKind::InvalidVariant,
            "candidate is covered by more than one kernel variant");
      }
      candidateToKernel[member] = static_cast<unsigned>(kernelIndex);
    }
  }

  if (llvm::any_of(candidateToKernel,
                   [](const std::optional<unsigned> &kernel) {
                     return !kernel.has_value();
                   })) {
    return DataflowFeasibilityResult::reject(
        DataflowRejectionKind::InvalidGraph,
        "mapping plan does not cover every graph candidate");
  }

  llvm::DenseSet<std::pair<unsigned, unsigned>> connections;
  for (const DataflowPlannedConnection &connection : plan.getConnections()) {
    if (connection.producerKernel >= plan.getKernels().size() ||
        connection.consumerKernel >= plan.getKernels().size() ||
        connection.producerKernel == connection.consumerKernel ||
        connection.bufferDepth == 0) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "plan contains an invalid kernel connection");
    }
    if (!connections
             .insert({connection.producerKernel, connection.consumerKernel})
             .second) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "plan contains a duplicate kernel connection");
    }
  }

  for (const DataflowCandidateEdge &edge : graph.getEdges()) {
    if (edge.producer >= candidateToKernel.size() ||
        edge.consumer >= candidateToKernel.size()) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "candidate graph contains an invalid dependency");
    }
    unsigned producerKernel = *candidateToKernel[edge.producer];
    unsigned consumerKernel = *candidateToKernel[edge.consumer];
    if (producerKernel == consumerKernel) {
      continue;
    }
    if (!connections.contains({producerKernel, consumerKernel})) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "plan does not preserve a producer-consumer dependency");
    }
    if (plan.getStrategy() == DataflowPlanStrategy::Temporal &&
        producerKernel >= consumerKernel) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "temporal plan does not preserve dependency order");
    }
  }

  return DataflowFeasibilityResult::success();
}

static std::optional<uint64_t>
getKernelInitiationInterval(const DataflowKernelVariant &variant) {
  const KernelCostEstimate &cost = variant.getCost();
  if (cost.initiationIntervalCycles) {
    return cost.initiationIntervalCycles;
  }
  if (!cost.computeCycles || !cost.dataMovementCycles) {
    return std::nullopt;
  }
  return std::max(*cost.computeCycles, *cost.dataMovementCycles);
}

static uint32_t getCoreCount(llvm::ArrayRef<int64_t> gridShape) {
  uint64_t count = 1;
  for (int64_t dim : gridShape) {
    count *= static_cast<uint64_t>(dim);
  }
  return static_cast<uint32_t>(
      std::min<uint64_t>(count, std::numeric_limits<uint32_t>::max()));
}

DataflowPlanCost
AnalyticalDataflowCostModel::evaluate(const DataflowCandidateGraph &,
                                      const DataflowMappingPlan &plan) const {
  DataflowPlanCost result;
  result.programCount = static_cast<uint32_t>(plan.getKernels().size());
  result.confidence = 1.0F;

  bool allCyclesKnown = true;
  uint64_t cycleSum = 0;
  uint64_t maxInitiationInterval = 0;
  for (const DataflowPlannedKernel &kernel : plan.getKernels()) {
    const DataflowKernelVariant &variant = kernel.variant;
    const KernelResourceEstimate &resources = variant.getResources();
    result.dramBytes += resources.dramBytes;
    result.nocBytes += resources.nocBytes;
    result.peakL1BytesPerCore =
        std::max(result.peakL1BytesPerCore, resources.l1BytesPerCore);

    uint32_t coreCount = getCoreCount(variant.getGridShape());
    if (plan.getStrategy() == DataflowPlanStrategy::Spatial) {
      result.occupiedCores += coreCount;
    } else {
      result.occupiedCores = std::max(result.occupiedCores, coreCount);
    }

    std::optional<uint64_t> initiationInterval =
        getKernelInitiationInterval(variant);
    if (!initiationInterval) {
      allCyclesKnown = false;
      result.confidence = 0.0F;
      continue;
    }
    cycleSum += *initiationInterval;
    maxInitiationInterval =
        std::max(maxInitiationInterval, *initiationInterval);
    result.confidence =
        std::min(result.confidence, variant.getCost().confidence);
  }

  if (allCyclesKnown) {
    if (plan.getStrategy() == DataflowPlanStrategy::Spatial) {
      result.initiationIntervalCycles = maxInitiationInterval;
    } else {
      result.latencyCycles = cycleSum;
    }
  }

  return result;
}

DataflowMappingPlan
buildTemporalFallbackPlan(const DataflowCandidateGraph &graph) {
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  kernels.reserve(graph.getOperations().size());
  for (auto item : llvm::enumerate(graph.getOperations())) {
    DataflowCandidateId candidateId =
        static_cast<DataflowCandidateId>(item.index());
    GenericOp operation = item.value();
    kernels.push_back({DataflowKernelVariant(
                           /*variantId=*/0, {candidateId},
                           llvm::to_vector(operation.getGrid().getShape()),
                           operation.getBlockFactorsValue()),
                       /*coreOffset=*/{}});
  }

  llvm::SmallVector<DataflowPlannedConnection> connections;
  connections.reserve(graph.getEdges().size());
  for (const DataflowCandidateEdge &edge : graph.getEdges()) {
    connections.push_back({edge.producer, edge.consumer,
                           DataflowConnectionKind::Materialized,
                           /*bufferDepth=*/1});
  }

  return DataflowMappingPlan(graph.getFunction(), graph.getBlock(),
                             graph.getScopeOrdinal(),
                             DataflowPlanStrategy::Temporal, std::move(kernels),
                             std::move(connections));
}

static void printOptionalCycles(llvm::raw_ostream &os,
                                std::optional<uint64_t> cycles) {
  if (cycles) {
    os << *cycles;
  } else {
    os << "unknown";
  }
}

void printDataflowPlan(llvm::raw_ostream &os,
                       const DataflowCandidateGraph &graph,
                       const DataflowMappingPlan &plan,
                       const DataflowPlanCost &cost) {
  os << "d2m-dataflow-plan function=@" << plan.getFunction().getSymName()
     << " scope=" << plan.getScopeOrdinal()
     << " strategy=" << stringifyDataflowPlanStrategy(plan.getStrategy())
     << " candidates=" << graph.getOperations().size()
     << " dependencies=" << graph.getEdges().size()
     << " kernels=" << plan.getKernels().size()
     << " connections=" << plan.getConnections().size() << " latency=";
  printOptionalCycles(os, cost.latencyCycles);
  os << " ii=";
  printOptionalCycles(os, cost.initiationIntervalCycles);
  os << "\n";
}

LogicalResult TemporalDataflowPlanMaterializer::materialize(
    const DataflowCandidateGraph &, const DataflowMappingPlan &plan) const {
  return success(plan.getStrategy() == DataflowPlanStrategy::Temporal);
}

} // namespace mlir::tt::d2m
