// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWPLANNING_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWPLANNING_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace mlir::tt::d2m {

using DataflowCandidateId = unsigned;

struct DataflowCandidateEdge {
  DataflowCandidateId producer;
  DataflowCandidateId consumer;
};

/// A stable, read-only view of the top-level d2m.generic operations in one
/// function block. Candidate IDs are indices into operations.
class DataflowCandidateGraph {
public:
  DataflowCandidateGraph(func::FuncOp function, Block *block,
                         unsigned scopeOrdinal,
                         llvm::SmallVector<GenericOp> operations,
                         llvm::SmallVector<DataflowCandidateEdge> edges);

  func::FuncOp getFunction() const { return function; }
  Block *getBlock() const { return block; }
  unsigned getScopeOrdinal() const { return scopeOrdinal; }
  llvm::ArrayRef<GenericOp> getOperations() const { return operations; }
  llvm::ArrayRef<DataflowCandidateEdge> getEdges() const { return edges; }

private:
  func::FuncOp function;
  Block *block;
  unsigned scopeOrdinal;
  llvm::SmallVector<GenericOp> operations;
  llvm::SmallVector<DataflowCandidateEdge> edges;
};

llvm::SmallVector<DataflowCandidateGraph>
buildDataflowCandidateGraphs(ModuleOp module);

struct KernelResourceEstimate {
  uint64_t l1BytesPerCore = 0;
  uint64_t dramBytes = 0;
  uint64_t nocBytes = 0;
  uint32_t cbCount = 0;
  uint32_t dstTiles = 0;
};

struct KernelCostEstimate {
  std::optional<uint64_t> computeCycles;
  std::optional<uint64_t> dataMovementCycles;
  std::optional<uint64_t> initiationIntervalCycles;
  float confidence = 0.0F;
};

/// One possible implementation of one or more candidate computations. A
/// multi-member variant represents fusion. Future staged variants may appear
/// more than once in a plan after the feasibility model learns their contract.
class DataflowKernelVariant {
public:
  DataflowKernelVariant(unsigned variantId,
                        llvm::SmallVector<DataflowCandidateId> members,
                        llvm::SmallVector<int64_t> gridShape,
                        llvm::SmallVector<int64_t> blockFactors,
                        KernelResourceEstimate resources = {},
                        KernelCostEstimate cost = {});

  unsigned getVariantId() const { return variantId; }
  llvm::ArrayRef<DataflowCandidateId> getMembers() const { return members; }
  llvm::ArrayRef<int64_t> getGridShape() const { return gridShape; }
  llvm::ArrayRef<int64_t> getBlockFactors() const { return blockFactors; }
  const KernelResourceEstimate &getResources() const { return resources; }
  const KernelCostEstimate &getCost() const { return cost; }

private:
  unsigned variantId;
  llvm::SmallVector<DataflowCandidateId> members;
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> blockFactors;
  KernelResourceEstimate resources;
  KernelCostEstimate cost;
};

struct DataflowPlannedKernel {
  DataflowKernelVariant variant;
  llvm::SmallVector<int64_t> coreOffset;
};

enum class DataflowConnectionKind {
  Materialized,
  L1Stream,
  NocStream,
  Dram,
};

llvm::StringRef stringifyDataflowConnectionKind(DataflowConnectionKind kind);

struct DataflowPlannedConnection {
  unsigned producerKernel;
  unsigned consumerKernel;
  DataflowConnectionKind kind = DataflowConnectionKind::Materialized;
  uint32_t bufferDepth = 1;
};

enum class DataflowPlanStrategy {
  Temporal,
  Fused,
  Spatial,
};

llvm::StringRef stringifyDataflowPlanStrategy(DataflowPlanStrategy strategy);

/// An immutable external-dataflow decision. Search implementations construct
/// plans out of band and materialize only the selected plan.
class DataflowMappingPlan {
public:
  DataflowMappingPlan(func::FuncOp function, Block *block,
                      unsigned scopeOrdinal, DataflowPlanStrategy strategy,
                      llvm::SmallVector<DataflowPlannedKernel, 0> kernels,
                      llvm::SmallVector<DataflowPlannedConnection> connections);

  func::FuncOp getFunction() const { return function; }
  Block *getBlock() const { return block; }
  unsigned getScopeOrdinal() const { return scopeOrdinal; }
  DataflowPlanStrategy getStrategy() const { return strategy; }
  llvm::ArrayRef<DataflowPlannedKernel> getKernels() const { return kernels; }
  llvm::ArrayRef<DataflowPlannedConnection> getConnections() const {
    return connections;
  }

private:
  func::FuncOp function;
  Block *block;
  unsigned scopeOrdinal;
  DataflowPlanStrategy strategy;
  llvm::SmallVector<DataflowPlannedKernel, 0> kernels;
  llvm::SmallVector<DataflowPlannedConnection> connections;
};

enum class DataflowRejectionKind {
  None,
  InvalidGraph,
  InvalidVariant,
  ResourceLimit,
  Unsupported,
};

llvm::StringRef stringifyDataflowRejectionKind(DataflowRejectionKind kind);

struct DataflowFeasibilityResult {
  bool feasible;
  DataflowRejectionKind rejectionKind;
  std::string message;

  static DataflowFeasibilityResult success();
  static DataflowFeasibilityResult reject(DataflowRejectionKind kind,
                                          llvm::StringRef message);
};

class DataflowFeasibilityModel {
public:
  virtual ~DataflowFeasibilityModel() = default;

  virtual DataflowFeasibilityResult
  evaluate(const DataflowCandidateGraph &graph,
           const DataflowMappingPlan &plan) const = 0;
};

/// Enforces graph coverage, variant shape, and connection integrity. Hardware
/// resource models can compose stricter checks on top of this baseline.
class StructuralDataflowFeasibilityModel final
    : public DataflowFeasibilityModel {
public:
  DataflowFeasibilityResult
  evaluate(const DataflowCandidateGraph &graph,
           const DataflowMappingPlan &plan) const override;
};

struct DataflowPlanCost {
  std::optional<uint64_t> latencyCycles;
  std::optional<uint64_t> initiationIntervalCycles;
  uint64_t dramBytes = 0;
  uint64_t nocBytes = 0;
  uint64_t peakL1BytesPerCore = 0;
  uint32_t occupiedCores = 0;
  uint32_t spillCount = 0;
  uint32_t programCount = 0;
  float confidence = 0.0F;
};

class DataflowCostModel {
public:
  virtual ~DataflowCostModel() = default;

  virtual DataflowPlanCost evaluate(const DataflowCandidateGraph &graph,
                                    const DataflowMappingPlan &plan) const = 0;
};

/// Aggregates optional per-kernel estimates without inventing unavailable
/// cycle data. Temporal latency and spatial initiation interval are reported
/// only when every participating kernel has a cycle estimate.
class AnalyticalDataflowCostModel final : public DataflowCostModel {
public:
  DataflowPlanCost evaluate(const DataflowCandidateGraph &graph,
                            const DataflowMappingPlan &plan) const override;
};

DataflowMappingPlan
buildTemporalFallbackPlan(const DataflowCandidateGraph &graph);

void printDataflowPlan(llvm::raw_ostream &os,
                       const DataflowCandidateGraph &graph,
                       const DataflowMappingPlan &plan,
                       const DataflowPlanCost &cost);

class DataflowPlanMaterializer {
public:
  virtual ~DataflowPlanMaterializer() = default;

  virtual LogicalResult materialize(const DataflowCandidateGraph &graph,
                                    const DataflowMappingPlan &plan) const = 0;
};

/// The temporal fallback already matches the input IR, so its materializer is
/// intentionally a no-op. Fused and spatial strategies provide their own
/// implementations without changing the planning pass boundary.
class TemporalDataflowPlanMaterializer final : public DataflowPlanMaterializer {
public:
  LogicalResult materialize(const DataflowCandidateGraph &graph,
                            const DataflowMappingPlan &plan) const override;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWPLANNING_H
