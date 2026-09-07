// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Analysis/DataflowPlanning.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDATAFLOWPLANNING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MDataflowPlanningPass final
    : public impl::D2MDataflowPlanningBase<D2MDataflowPlanningPass> {
public:
  using impl::D2MDataflowPlanningBase<
      D2MDataflowPlanningPass>::D2MDataflowPlanningBase;

  void runOnOperation() override {
    StructuralDataflowFeasibilityModel feasibilityModel;
    AnalyticalDataflowCostModel costModel;
    TemporalDataflowPlanMaterializer materializer;

    for (const DataflowCandidateGraph &graph :
         buildDataflowCandidateGraphs(getOperation())) {
      DataflowMappingPlan plan = buildTemporalFallbackPlan(graph);
      DataflowFeasibilityResult feasibility =
          feasibilityModel.evaluate(graph, plan);
      if (!feasibility.feasible) {
        getOperation().emitError()
            << "dataflow planner rejected the selected mapping plan ("
            << stringifyDataflowRejectionKind(feasibility.rejectionKind)
            << "): " << feasibility.message;
        signalPassFailure();
        return;
      }

      DataflowPlanCost cost = costModel.evaluate(graph, plan);
      if (this->dumpPlan) {
        printDataflowPlan(llvm::errs(), graph, plan, cost);
      }

      if (failed(materializer.materialize(graph, plan))) {
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
