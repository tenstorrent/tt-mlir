// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <utility>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDATAFLOWPLANNING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// This is the stable input boundary for future dependency and legality
// analyses. Keeping collection separate from planning prevents search from
// repeatedly cloning or mutating the payload IR.
class CandidateGraph {
public:
  CandidateGraph(func::FuncOp function, Block *block, unsigned scopeOrdinal,
                 llvm::SmallVector<GenericOp> operations)
      : function(function), block(block), scopeOrdinal(scopeOrdinal),
        operations(std::move(operations)) {}

  func::FuncOp getFunction() const { return function; }

  Block *getBlock() const { return block; }

  unsigned getScopeOrdinal() const { return scopeOrdinal; }

  llvm::ArrayRef<GenericOp> getOperations() const { return operations; }

private:
  func::FuncOp function;
  Block *block;
  unsigned scopeOrdinal;
  llvm::SmallVector<GenericOp> operations;
};

// MappingPlan intentionally exposes no mutation API. Search implementations
// should build candidate plans out of band and materialize only the selected
// plan.
class MappingPlan {
public:
  MappingPlan(func::FuncOp function, Block *block, unsigned scopeOrdinal,
              llvm::SmallVector<GenericOp> temporalOrder)
      : function(function), block(block), scopeOrdinal(scopeOrdinal),
        temporalOrder(std::move(temporalOrder)) {}

  func::FuncOp getFunction() const { return function; }

  Block *getBlock() const { return block; }

  unsigned getScopeOrdinal() const { return scopeOrdinal; }

  llvm::ArrayRef<GenericOp> getTemporalOrder() const { return temporalOrder; }

private:
  func::FuncOp function;
  Block *block;
  unsigned scopeOrdinal;
  llvm::SmallVector<GenericOp> temporalOrder;
};

static llvm::SmallVector<CandidateGraph> buildCandidateGraphs(ModuleOp module) {
  llvm::SmallVector<CandidateGraph> graphs;

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

    for (std::size_t scope = 0; scope < operationsByBlock.size(); ++scope) {
      graphs.emplace_back(function, blocks[scope], static_cast<unsigned>(scope),
                          std::move(operationsByBlock[scope]));
    }
  }

  return graphs;
}

// The first policy is the always-legal temporal fallback: retain the existing
// d2m.generic order and let downstream passes make their current decisions.
static MappingPlan buildTemporalFallbackPlan(const CandidateGraph &graph) {
  return MappingPlan(graph.getFunction(), graph.getBlock(),
                     graph.getScopeOrdinal(),
                     llvm::SmallVector<GenericOp>(graph.getOperations()));
}

static LogicalResult verifyMappingPlan(const MappingPlan &plan) {
  llvm::DenseSet<Operation *> seen;
  Operation *previous = nullptr;
  for (GenericOp genericOp : plan.getTemporalOrder()) {
    if (!genericOp || genericOp->getBlock() != plan.getBlock() ||
        genericOp->getParentOfType<func::FuncOp>() != plan.getFunction() ||
        genericOp->getParentOfType<SpatialOp>() ||
        genericOp->getParentOfType<GenericOp>() ||
        !seen.insert(genericOp.getOperation()).second ||
        (previous && !previous->isBeforeInBlock(genericOp))) {
      return failure();
    }
    previous = genericOp;
  }
  return success();
}

static void dumpMappingPlan(const MappingPlan &plan) {
  llvm::errs() << "d2m-dataflow-plan function=@"
               << plan.getFunction().getSymName()
               << " scope=" << plan.getScopeOrdinal()
               << " strategy=temporal-fallback generics="
               << plan.getTemporalOrder().size() << "\n";
}

static LogicalResult materializeMappingPlan(const MappingPlan &) {
  // The fallback is already represented by the current IR, so materialization
  // is intentionally a no-op. Spatial and fused plans will be emitted here.
  return success();
}

class D2MDataflowPlanningPass final
    : public impl::D2MDataflowPlanningBase<D2MDataflowPlanningPass> {
public:
  using impl::D2MDataflowPlanningBase<
      D2MDataflowPlanningPass>::D2MDataflowPlanningBase;

  void runOnOperation() override {
    llvm::SmallVector<CandidateGraph> graphs =
        buildCandidateGraphs(getOperation());
    for (const CandidateGraph &graph : graphs) {
      MappingPlan plan = buildTemporalFallbackPlan(graph);

      if (failed(verifyMappingPlan(plan))) {
        getOperation().emitError(
            "dataflow planner produced an invalid mapping plan");
        signalPassFailure();
        return;
      }

      if (this->dumpPlan) {
        dumpMappingPlan(plan);
      }

      if (failed(materializeMappingPlan(plan))) {
        getOperation().emitError("failed to materialize the selected dataflow "
                                 "mapping plan");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
