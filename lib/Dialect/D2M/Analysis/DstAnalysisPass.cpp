//===- DstAnalysisPass.cpp - DST Analysis --------------------------------===//
//
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisBasic.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisGraphColoring.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDSTREQUIREMENTANALYSIS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct D2MDstRequirementAnalysisPass
    : public impl::D2MDstRequirementAnalysisBase<
          D2MDstRequirementAnalysisPass> {

  D2MDstRequirementAnalysisPass() = default;
  D2MDstRequirementAnalysisPass(const D2MDstRequirementAnalysisPass &pass) =
      default;
  D2MDstRequirementAnalysisPass(const D2MDstRequirementAnalysisOptions &options)
      : D2MDstRequirementAnalysisBase(options) {}

  void runOnOperation() override {
    auto func = getOperation();

    // Create analysis strategy based on option
    std::unique_ptr<DstAnalysis> analysis;
    std::string strategyName = strategy.getValue();

    if (strategyName == "basic") {
      analysis = createBasicDstAnalysis();
    } else if (strategyName == "greedy") {
      analysis = createGreedyDstAnalysis();
    } else if (strategyName == "graph-coloring") {
      // Default to Chaitin-Briggs graph coloring
      analysis = createChaitinBriggsDstAnalysis();
    } else {
      func.emitError() << "Unknown DST analysis strategy: " << strategyName
                       << ". Valid options: basic, graph-coloring, greedy";
      return signalPassFailure();
    }

    // Run analysis
    DstAnalysisResult result = analysis->analyze(func);

    // Emit diagnostics if requested
    if (emitDiagnostics) {
      if (result.isValid) {
        func.emitRemark() << "DST analysis (" << analysis->getStrategyName()
                          << "): " << result.numSlicesRequired
                          << " slices required";
      } else {
        func.emitWarning() << "DST analysis failed: "
                           << result.failureReason.value_or("unknown reason");
      }
    }

    // Store result as pass attribute for other passes to query
    func->setAttr(
        "dst_slices_required",
        mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32),
                               result.numSlicesRequired));
  }
};

} // namespace
} // namespace mlir::tt::d2m
