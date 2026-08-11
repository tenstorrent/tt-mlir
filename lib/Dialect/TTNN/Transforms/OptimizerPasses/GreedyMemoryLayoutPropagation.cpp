// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/Diagnostics/CompileTimeStatsObserver.h"
#include "ttmlir/Dialect/TTNN/Diagnostics/DecisionTrace.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/D2MOptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
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
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdio>
#include <string>

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
    maxReshardCandidatesPerType = opts.maxReshardCandidatesPerType;
    enableL1ShardingLayouts = opts.enableL1ShardingLayouts;
    enableDecisionTrace = opts.enableDecisionTrace;
    decisionTraceDir = std::move(opts.decisionTraceDir);
    enableCompileTimeStats = opts.enableCompileTimeStats;
    overrideOutputLayout = std::move(opts.overrideOutputLayout);
    overrideConv2dConfig = std::move(opts.overrideConv2dConfig);
    overrideConv3dConfig = std::move(opts.overrideConv3dConfig);
    enableConv2dSearchExtensions = opts.enableConv2dSearchExtensions;
  }

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNGreedyMemoryLayoutPropagation pass requires OpModel support to be "
        "enabled.");
#else
    auto _tMLA = std::chrono::steady_clock::now();
    fprintf(stderr, "[mla-timing] GreedyMemoryLayoutPropagation START\n");

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
    fprintf(stderr, "[mla-timing] Step1: ScalarDataTypeAnalysis ...\n");
    auto _tStep1 = std::chrono::steady_clock::now();
    ScalarDataTypeAnalysis scalarDataTypeAnalysis =
        getAnalysis<ScalarDataTypeAnalysis>();
    scalarDataTypeAnalysis.init(
        ScalarDataTypeAnalysisInput(&overrideOutputLayout));
    auto scalarTypes = scalarDataTypeAnalysis.getResult();
    fprintf(stderr, "[mla-timing] Step1 done in %lld ms  (%zu scalar types)\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - _tStep1)
                .count(),
            scalarTypes.size());

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "ScalarDataTypeAnalysis found {0} unique scalar types.",
                 scalarTypes.size());

    // Step 2: Run LegalTensorLayoutAnalysis to generate layouts for all tensor
    // types.
    fprintf(stderr, "[mla-timing] Step2: LegalTensorLayoutAnalysis ...\n");
    auto _tStep2 = std::chrono::steady_clock::now();
    LegalTensorLayoutAnalysis legalTensorLayoutAnalysis =
        getAnalysis<LegalTensorLayoutAnalysis>();
    legalTensorLayoutAnalysis.init(LegalTensorLayoutAnalysisInput(
        deviceGrid, &scalarTypes, rowMajorEnabled));
    TensorTypeLayoutsMap tensorTypePossibleLayouts =
        legalTensorLayoutAnalysis.getResult();
    size_t _numTensorTypes = tensorTypePossibleLayouts.size();
    fprintf(stderr,
            "[mla-timing] Step2 done in %lld ms  (%zu tensor types)\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - _tStep2)
                .count(),
            _numTensorTypes);

    if (!enableL1ShardingLayouts) {
      clearShardedLayouts(tensorTypePossibleLayouts);

      // Clear interleaved layouts that have L1 buffer type too, as we dont want
      // to consider any L1 layouts if L1 sharding layouts are disabled.
      clearL1InterleavedLayouts(tensorTypePossibleLayouts);
    }

    // Step 3: Walk operations and run per-op analyses.
    fprintf(stderr,
            "[mla-timing] Step3: per-op LegalOpLayout/Config analysis ...\n");
    auto _tStep3 = std::chrono::steady_clock::now();
    size_t _step3OpIdx = 0;
    moduleOp->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      func->walk([&](Operation *op) {
        if (!optimizer_utils::opHasTensorResult(op)) {
          // Constraint sinks (FillCacheOp, PagedUpdateCacheOp,
          // PagedFillCacheOp) have no tensor result but still need input layout
          // validation. Register a null-output OpConfig so processOp can
          // validate their input combinations and drive upstream reshards.
          if (mlir::dyn_cast<OpModel>(op) && optimizer_utils::isSinkOp(op)) {
            legalConfigs[op] = {OpConfig{TTNNLayoutAttr()}};
          }
          return;
        }
        // Skip ops that don't implement the OpModel interface (e.g.,
        // ttcore.load_cached). These ops cannot be validated by the backend.
        if (!mlir::dyn_cast<OpModel>(op)) {
          return;
        }
        // Skip ToTensorSpecOp -- their layouts are set by other passes.
        if (isa<ToTensorSpecOp>(op)) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());

        // Build loc string for timing print
        std::string _locStr;
        {
          llvm::raw_string_ostream _ss(_locStr);
          op->getLoc().print(_ss);
        }
        fprintf(stderr,
                "[mla-timing]   op[%zu] %-36s  shape=[",
                _step3OpIdx,
                op->getName().getStringRef().str().c_str());
        for (size_t _i = 0; _i < tensorType.getShape().size(); ++_i) {
          if (_i > 0) fprintf(stderr, ",");
          fprintf(stderr, "%lld", (long long)tensorType.getShape()[_i]);
        }
        fprintf(stderr, "]  loc=%s\n", _locStr.c_str());
        auto _tOp = std::chrono::steady_clock::now();

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
            legalOpLayoutAnalysis.getResult(), &overrideConv2dConfig,
            &overrideConv3dConfig, enableConv2dSearchExtensions));
        legalConfigs[op] = legalOpConfigAnalysis.getResult();

        fprintf(stderr,
                "[mla-timing]   op[%zu] done  %lld ms  layouts=%zu  configs=%zu\n",
                _step3OpIdx,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - _tOp)
                    .count(),
                legalOpLayoutAnalysis.getResult().size(),
                legalConfigs[op].size());
        ++_step3OpIdx;
      });
    });
    fprintf(stderr, "[mla-timing] Step3 done in %lld ms  (%zu ops)\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - _tStep3)
                .count(),
            _step3OpIdx);

    // Step 4: Run layout propagation for each forward device function.
    fprintf(stderr,
            "[mla-timing] Step4: MemoryLayoutPropagation  legalConfigs=%zu ...\n",
            legalConfigs.size());
    auto _tStep4 = std::chrono::steady_clock::now();
    moduleOp->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "Running greedy layout propagation on func {0} with "
                   "{1} legal op configs.",
                   func.getName(), legalConfigs.size());

      std::unique_ptr<LayoutPropagationObserver> observer;
      if (enableDecisionTrace) {
        if (enableCompileTimeStats) {
          TTMLIR_TRACE(
              ttmlir::LogComponent::GreedyOptimizer,
              "Both decision-trace and compile-time-stats enabled; "
              "using decision trace (options are mutually exclusive).");
        }
        observer = std::make_unique<DecisionTraceObserver>();
      } else if (enableCompileTimeStats) {
        observer = std::make_unique<CompileTimeStatsObserver>();
      }

      fprintf(stderr, "[mla-timing]   propagation.run() for func '%s' ...\n",
              func.getName().str().c_str());
      auto _tProp = std::chrono::steady_clock::now();
      MemoryLayoutPropagation propagation(
          func, legalConfigs, &tensorTypePossibleLayouts,
          static_cast<size_t>(beamWidth),
          static_cast<size_t>(maxInputCandidatesPerOperand),
          static_cast<size_t>(maxReshardCandidatesPerType),
          std::move(observer));
      propagation.run();
      fprintf(stderr, "[mla-timing]   propagation.run() done  %lld ms\n",
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - _tProp)
                  .count());

      // Sync D2M subgraph function types to match dispatch op's current inputs
      // (e.g. after reshard insertion, operand types may have changed).
      d2m_optimizer_utils::syncAllD2MFuncTypes(func);

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
    fprintf(stderr, "[mla-timing] Step4 done in %lld ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - _tStep4)
                .count());
    fprintf(stderr, "[mla-timing] GreedyMemoryLayoutPropagation TOTAL %lld ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - _tMLA)
                .count());
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
  ::mlir::Pass::Option<llvm::StringMap<Conv3dConfigOverrideParams>,
                       Conv3dConfigOverrideParser>
      overrideConv3dConfig{
          *this, OptionNames::overrideConv3dConfig,
          ::llvm::cl::desc("Override Conv3d configuration for specific ops."),
          ::llvm::cl::init(llvm::StringMap<Conv3dConfigOverrideParams>())};
  ::mlir::Pass::Option<bool> enableConv2dSearchExtensions{
      *this, "enable-conv2d-search-extensions",
      ::llvm::cl::desc("Enable extended Conv2d config search space "
                       "(actBlockH 384, double-buffer, reshardIfNotOptimal)."),
      ::llvm::cl::init(false)};

};

// Pipeline create function.
std::unique_ptr<::mlir::Pass> createTTNNGreedyMemoryLayoutPropagation(
    TTNNGreedyMemoryLayoutPropagationPipelineOptions options) {
  return std::make_unique<TTNNGreedyMemoryLayoutPropagation>(
      std::move(options));
}

} // namespace mlir::tt::ttnn
