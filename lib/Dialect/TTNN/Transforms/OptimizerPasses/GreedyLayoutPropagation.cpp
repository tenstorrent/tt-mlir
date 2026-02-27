// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/GreedyLayoutPropagation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/DecisionTrace.h"
#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
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

namespace impl {

std::unique_ptr<::mlir::Pass> createTTNNGreedyLayoutPropagation();
std::unique_ptr<::mlir::Pass>
createTTNNGreedyLayoutPropagation(TTNNGreedyLayoutPropagationOptions options);

template <typename DerivedT>
class TTNNGreedyLayoutPropagationBase
    : public ::mlir::OperationPass<::mlir::ModuleOp> {
public:
  using Base = TTNNGreedyLayoutPropagationBase;

  TTNNGreedyLayoutPropagationBase()
      : ::mlir::OperationPass<::mlir::ModuleOp>(
            ::mlir::TypeID::get<DerivedT>()) {}
  TTNNGreedyLayoutPropagationBase(const TTNNGreedyLayoutPropagationBase &other)
      : ::mlir::OperationPass<::mlir::ModuleOp>(other) {}
  TTNNGreedyLayoutPropagationBase &
  operator=(const TTNNGreedyLayoutPropagationBase &) = delete;
  TTNNGreedyLayoutPropagationBase(TTNNGreedyLayoutPropagationBase &&) = delete;
  TTNNGreedyLayoutPropagationBase &
  operator=(TTNNGreedyLayoutPropagationBase &&) = delete;
  ~TTNNGreedyLayoutPropagationBase() override = default;

  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("ttnn-greedy-layout-propagation");
  }
  ::llvm::StringRef getArgument() const override {
    return "ttnn-greedy-layout-propagation";
  }

  ::llvm::StringRef getDescription() const override {
    return "Greedy edge-based layout propagation for all operands.";
  }

  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("TTNNGreedyLayoutPropagation");
  }
  ::llvm::StringRef getName() const override {
    return "TTNNGreedyLayoutPropagation";
  }

  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TTNNGreedyLayoutPropagationBase<DerivedT>)

  TTNNGreedyLayoutPropagationBase(TTNNGreedyLayoutPropagationOptions options)
      : TTNNGreedyLayoutPropagationBase() {
    maxLegalLayouts = std::move(options.maxLegalLayouts);
    rowMajorEnabled = std::move(options.rowMajorEnabled);
    beamWidth = std::move(options.beamWidth);
    insertMemReconfig = std::move(options.insertMemReconfig);
    overrideOutputLayout = std::move(options.overrideOutputLayout);
    overrideConv2dConfig = std::move(options.overrideConv2dConfig);
    enableDecisionTrace = std::move(options.enableDecisionTrace);
    decisionTraceDir = std::move(options.decisionTraceDir);
  }

protected:
  ::mlir::Pass::Option<int64_t> maxLegalLayouts{
      *this, OptionNames::maxLegalLayouts,
      ::llvm::cl::desc("Override maximum number of sharded layouts for legal "
                       "layout analysis."),
      ::llvm::cl::init(64)};
  ::mlir::Pass::Option<bool> rowMajorEnabled{
      *this, "row-major-enabled",
      ::llvm::cl::desc(
          "Enable row major layout generation in legal layout analysis."),
      ::llvm::cl::init(false)};
  ::mlir::Pass::Option<int64_t> beamWidth{
      *this, "beam-width",
      ::llvm::cl::desc(
          "Beam width for layout propagation (1=greedy, >1=beam search)."),
      ::llvm::cl::init(8)};
  ::mlir::Pass::Option<llvm::StringMap<InsertMemReconfigParams>,
                       mlir::tt::ttnn::InsertMemReconfigParser>
      insertMemReconfig{
          *this, OptionNames::insertMemReconfig,
          ::llvm::cl::desc(
              "Manually insert memory reconfig op for specific op's operand."),
          ::llvm::cl::init(llvm::StringMap<InsertMemReconfigParams>())};
  ::mlir::Pass::Option<llvm::StringMap<OutputLayoutOverrideParams>,
                       mlir::tt::ttnn::OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, OptionNames::overrideOutputLayout,
          ::llvm::cl::desc("Override output tensor layout for specific ops."),
          ::llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};
  ::mlir::Pass::Option<llvm::StringMap<Conv2dConfigOverrideParams>,
                       mlir::tt::ttnn::Conv2dConfigOverrideParser>
      overrideConv2dConfig{
          *this, OptionNames::overrideConv2dConfig,
          ::llvm::cl::desc("Override Conv2d configuration for specific ops."),
          ::llvm::cl::init(llvm::StringMap<Conv2dConfigOverrideParams>())};
  ::mlir::Pass::Option<bool> enableDecisionTrace{
      *this, "enable-decision-trace",
      ::llvm::cl::desc("Enable decision trace JSON output."),
      ::llvm::cl::init(false)};
  ::mlir::Pass::Option<std::string> decisionTraceDir{
      *this, "decision-trace-dir",
      ::llvm::cl::desc("Directory for decision trace JSON output."),
      ::llvm::cl::init("ttrt-artifacts/decision_trace")};

private:
  friend std::unique_ptr<::mlir::Pass> createTTNNGreedyLayoutPropagation() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass> createTTNNGreedyLayoutPropagation(
      TTNNGreedyLayoutPropagationOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createTTNNGreedyLayoutPropagation() {
  return impl::createTTNNGreedyLayoutPropagation();
}

std::unique_ptr<::mlir::Pass>
createTTNNGreedyLayoutPropagation(TTNNGreedyLayoutPropagationOptions options) {
  return impl::createTTNNGreedyLayoutPropagation(std::move(options));
}

class TTNNGreedyLayoutPropagation
    : public impl::TTNNGreedyLayoutPropagationBase<
          TTNNGreedyLayoutPropagation> {
public:
  using impl::TTNNGreedyLayoutPropagationBase<
      TTNNGreedyLayoutPropagation>::TTNNGreedyLayoutPropagationBase;

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNGreedyLayoutPropagation pass requires OpModel support to be "
        "enabled.");
#else
    ModuleOp moduleOp = getOperation();
    op_model::ScopedSingletonDeviceGuard deviceGuard(moduleOp);

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

      std::unique_ptr<LayoutPropagationObserver> observer;
      if (enableDecisionTrace) {
        observer = std::make_unique<DecisionTraceObserver>();
      }

      LayoutPropagation propagation(func, deviceGrid, legalConfigs,
                                    &tensorTypePossibleLayouts,
                                    static_cast<size_t>(beamWidth),
                                    std::move(observer));
      propagation.run();

      // Write decision trace JSON if enabled.
      if (enableDecisionTrace) {
        if (const DecisionTrace *dt =
                propagation.getObserver()->getDecisionTrace()) {
          if (DecisionTrace::writeTraceForFunc(decisionTraceDir,
                                               func.getName(), *dt)) {
            TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                         "Decision trace written to {0}/{1}", decisionTraceDir,
                         func.getName());
          }
        }
      }
    });
#endif
  }
};

} // namespace mlir::tt::ttnn
