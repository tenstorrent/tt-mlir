// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/CompileTimeStatsObserver.h"
#include "ttmlir/Dialect/TTNN/Analysis/DecisionTrace.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNGREEDYMEMORYLAYOUTPROPAGATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNGreedyMemoryLayoutPropagation
    : public impl::TTNNGreedyMemoryLayoutPropagationBase<
          TTNNGreedyMemoryLayoutPropagation> {
public:
  using impl::TTNNGreedyMemoryLayoutPropagationBase<
      TTNNGreedyMemoryLayoutPropagation>::TTNNGreedyMemoryLayoutPropagationBase;

  // Custom copy constructor: Pass::Option members are non-copyable, so we
  // delegate to the base copy constructor and default-initialize them.
  TTNNGreedyMemoryLayoutPropagation(
      const TTNNGreedyMemoryLayoutPropagation &other)
      : TTNNGreedyMemoryLayoutPropagationBase(other) {}

  // Pipeline constructor: accepts complex options beyond what tablegen handles.
  TTNNGreedyMemoryLayoutPropagation(
      TTNNGreedyMemoryLayoutPropagationPipelineOptions opts)
      : TTNNGreedyMemoryLayoutPropagationBase() {
    maxLegalLayouts = opts.maxLegalLayouts;
    rowMajorEnabled = opts.rowMajorEnabled;
    beamWidth = opts.beamWidth;
    maxInputCandidatesPerOperand = opts.maxInputCandidatesPerOperand;
    maxReshardCandidates = opts.maxReshardCandidates;
    enableL1ShardingLayouts = opts.enableL1ShardingLayouts;
    enableDecisionTrace = opts.enableDecisionTrace;
    decisionTraceDir = std::move(opts.decisionTraceDir);
    enableCompileTimeStats = opts.enableCompileTimeStats;
    overrideOutputLayout = std::move(opts.overrideOutputLayout);
    overrideConv2dConfig = std::move(opts.overrideConv2dConfig);
  }

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNGreedyMemoryLayoutPropagation pass requires OpModel support to be "
        "enabled.");
#else
    ModuleOp moduleOp = getOperation();
    op_model::ScopedSingletonDeviceGuard deviceGuard(moduleOp);

    // Set default L1Full slice config on Conv2d ops before validation.
    applyConvSliceConfig(moduleOp);

    // Get the max grid size from the system description.
    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();

    llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;

    // Step 1: Run ScalarDataTypeAnalysis to collect all scalar types used in
    // the graph.
    ScalarDataTypeAnalysis scalarDataTypeAnalysis =
        getAnalysis<ScalarDataTypeAnalysis>();
    scalarDataTypeAnalysis.init(
        ScalarDataTypeAnalysisInput(&overrideOutputLayout));
    auto scalarTypes = scalarDataTypeAnalysis.getResult();

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "ScalarDataTypeAnalysis found {0} unique scalar types.",
                 scalarTypes.size());

    // Step 2: Run LegalTensorLayoutAnalysis to generate layouts for all tensor
    // types.
    LegalTensorLayoutAnalysis legalTensorLayoutAnalysis =
        getAnalysis<LegalTensorLayoutAnalysis>();
    legalTensorLayoutAnalysis.init(LegalTensorLayoutAnalysisInput(
        deviceGrid, &scalarTypes, rowMajorEnabled));
    TensorTypeLayoutsMap tensorTypePossibleLayouts =
        legalTensorLayoutAnalysis.getResult();

    if (!enableL1ShardingLayouts) {
      clearShardedLayouts(tensorTypePossibleLayouts);
    }

    // Step 3: Walk operations and run per-op analyses.
    moduleOp->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      func->walk([&](Operation *op) {
        if (!LegalOpLayoutAnalysis::isValidAnalysisTarget(op)) {
          return;
        }
        // Skip ops that don't implement the OpModel interface (e.g.,
        // ttcore.load_cached). These ops cannot be validated by the backend.
        if (!mlir::dyn_cast<OpModel>(op)) {
          return;
        }
        // Skip ToLayoutOp -- their layouts are set by other passes.
        if (isa<ToLayoutOp>(op)) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());

        auto tensorLayouts = tensorTypePossibleLayouts.find(tensorType);
        assert(tensorLayouts != tensorTypePossibleLayouts.end() &&
               "No layouts found for tensor type");

        LegalOpLayoutAnalysis legalOpLayoutAnalysis =
            getChildAnalysis<LegalOpLayoutAnalysis>(op);
        legalOpLayoutAnalysis.init(LegalOpLayoutAnalysisInput(
            &tensorLayouts->getSecond(), maxLegalLayouts, &overrideOutputLayout,
            rowMajorEnabled));

        LegalOpConfigAnalysis legalOpConfigAnalysis =
            getChildAnalysis<LegalOpConfigAnalysis>(op);
        legalOpConfigAnalysis.init(LegalOpConfigAnalysisInput(
            legalOpLayoutAnalysis.getResult(), &overrideConv2dConfig));
        legalConfigs[op] = legalOpConfigAnalysis.getResult();
      });
    });

    // Step 4: Run layout propagation for each forward device function.
    moduleOp->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "Running greedy layout propagation on func {0} with "
                   "{1} legal op configs.",
                   func.getName(), legalConfigs.size());

      std::unique_ptr<MemoryLayoutPropagationObserver> observer;
      if (enableCompileTimeStats) {
        observer = std::make_unique<CompileTimeStatsObserver>();
      } else if (enableDecisionTrace) {
        observer = std::make_unique<DecisionTraceObserver>();
      }

      MemoryLayoutPropagation propagation(
          func, deviceGrid, legalConfigs, &tensorTypePossibleLayouts,
          static_cast<size_t>(beamWidth),
          static_cast<size_t>(maxInputCandidatesPerOperand),
          static_cast<size_t>(maxReshardCandidates), std::move(observer));
      propagation.run();

      // Write decision trace JSON if enabled.
      if (enableDecisionTrace) {
        if (const DecisionTrace *dt =
                propagation.getObserver()->getDecisionTrace()) {
          if (DecisionTrace::writeTraceForFunc(decisionTraceDir, func.getName(),
                                               *dt)) {
            TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                         "Decision trace written to {0}/{1}", decisionTraceDir,
                         func.getName());
          }
        }
      }
    });
#endif
  }

protected:
  ::mlir::Pass::Option<llvm::StringMap<OutputLayoutOverrideParams>,
                       OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, OptionNames::overrideOutputLayout,
          ::llvm::cl::desc("Override output tensor layout for specific ops."),
          ::llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};
  ::mlir::Pass::Option<llvm::StringMap<Conv2dConfigOverrideParams>,
                       Conv2dConfigOverrideParser>
      overrideConv2dConfig{
          *this, OptionNames::overrideConv2dConfig,
          ::llvm::cl::desc("Override Conv2d configuration for specific ops."),
          ::llvm::cl::init(llvm::StringMap<Conv2dConfigOverrideParams>())};
};

// Pipeline create function.
std::unique_ptr<::mlir::Pass> createTTNNGreedyMemoryLayoutPropagation(
    TTNNGreedyMemoryLayoutPropagationPipelineOptions options) {
  return std::make_unique<TTNNGreedyMemoryLayoutPropagation>(
      std::move(options));
}

} // namespace mlir::tt::ttnn
