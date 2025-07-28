// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemReconfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNTraits.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

namespace impl {

std::unique_ptr<::mlir::Pass> createTTNNOptimizer();
std::unique_ptr<::mlir::Pass> createTTNNOptimizer(TTNNOptimizerOptions options);

// Go through the ops, set sharding specs for each op based on sharding
// analysis, by updating layout attribute of each op.
//
template <typename DerivedT>
class TTNNOptimizerBase : public ::mlir::OperationPass<::mlir::ModuleOp> {
public:
  using Base = TTNNOptimizerBase;

  TTNNOptimizerBase()
      : ::mlir::OperationPass<::mlir::ModuleOp>(
            ::mlir::TypeID::get<DerivedT>()) {}
  TTNNOptimizerBase(const TTNNOptimizerBase &other)
      : ::mlir::OperationPass<::mlir::ModuleOp>(other) {}
  TTNNOptimizerBase &operator=(const TTNNOptimizerBase &) = delete;
  TTNNOptimizerBase(TTNNOptimizerBase &&) = delete;
  TTNNOptimizerBase &operator=(TTNNOptimizerBase &&) = delete;
  ~TTNNOptimizerBase() override = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("ttnn-optimizer");
  }
  ::llvm::StringRef getArgument() const override { return "ttnn-optimizer"; }

  ::llvm::StringRef getDescription() const override {
    return "Determine op configurations for maximum performance.";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("TTNNOptimizer");
  }
  ::llvm::StringRef getName() const override { return "TTNNOptimizer"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTNNOptimizerBase<DerivedT>)

  TTNNOptimizerBase(TTNNOptimizerOptions options) : TTNNOptimizerBase() {
    insertMemReconfig = std::move(options.insertMemReconfig);
    overrideOutputLayout = std::move(options.overrideOutputLayout);
    overrideConv2dConfig = std::move(options.overrideConv2dConfig);
    memoryLayoutAnalysisEnabled =
        std::move(options.memoryLayoutAnalysisEnabled);
    l1InterleavedAnalysisEnabled =
        std::move(options.l1InterleavedAnalysisEnabled);
    memReconfigEnabled = std::move(options.memReconfigEnabled);
    memoryLayoutAnalysisPolicy = std::move(options.memoryLayoutAnalysisPolicy);
    maxLegalLayouts = std::move(options.maxLegalLayouts);
    rowMajorEnabled = std::move(options.rowMajorEnabled);
  }

protected:
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
  ::mlir::Pass::Option<bool> memoryLayoutAnalysisEnabled{
      *this, OptionNames::memoryLayoutAnalysisEnabled,
      ::llvm::cl::desc("Enable memory layout optimization."),
      ::llvm::cl::init(false)};
  ::mlir::Pass::Option<bool> l1InterleavedAnalysisEnabled{
      *this, OptionNames::l1InterleavedAnalysisEnabled,
      ::llvm::cl::desc("Enable L1 interleaved optimization."),
      ::llvm::cl::init(false)};
  ::mlir::Pass::Option<bool> memReconfigEnabled{
      *this, OptionNames::memReconfigEnabled,
      ::llvm::cl::desc("Memory layout reconfiguration pass."),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<mlir::tt::MemoryLayoutAnalysisPolicyType,
                       mlir::tt::MemoryLayoutAnalysisPolicyTypeParser>
      memoryLayoutAnalysisPolicy{
          *this, OptionNames::memoryLayoutAnalysisPolicy,
          llvm::cl::desc("Specify policy for memory layout analysis."),
          llvm::cl::init(MemoryLayoutAnalysisPolicyType::DFSharding)};
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

private:
  friend std::unique_ptr<::mlir::Pass> createTTNNOptimizer() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass>
  createTTNNOptimizer(TTNNOptimizerOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createTTNNOptimizer() {
  return impl::createTTNNOptimizer();
}

std::unique_ptr<::mlir::Pass>
createTTNNOptimizer(TTNNOptimizerOptions options) {
  return impl::createTTNNOptimizer(std::move(options));
}

class TTNNOptimizer : public impl::TTNNOptimizerBase<TTNNOptimizer> {
public:
  using impl::TTNNOptimizerBase<TTNNOptimizer>::TTNNOptimizerBase;
  void runOnOperation() final {
    // Generate legal OP configuration candidates.
    // Perform memory layout analysis.
    // Perform final configuration analysis.
    // Apply graph transformations based on analysis results.
    //
    assertOverridesValid();

    ModuleOp moduleOp = getOperation();

    // Get the max grid size from the system description.
    //
    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();

    ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
        moduleOp->getAttr(ttcore::SystemDescAttr::name));
    ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
    llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
    // Map to store only L1 Interleaved legal configs for L1InterleavedAnalysis
    llvm::DenseMap<Operation *, std::vector<OpConfig>>
        l1InterleavedLegalConfigs;

    // Step 1: Run ScalarDataTypeAnalysis to collect all scalar types used in
    // the graph
    ScalarDataTypeAnalysis scalarDataTypeAnalysis =
        getAnalysis<ScalarDataTypeAnalysis>();
    scalarDataTypeAnalysis.init(
        ScalarDataTypeAnalysisInput(&overrideOutputLayout));
    auto scalarTypes = scalarDataTypeAnalysis.getResult();

    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "ScalarDataTypeAnalysis found {0} unique scalar types",
                 scalarTypes.size());

    // Step 2: Run LegalTensorLayoutAnalysis to generate layouts for all tensor
    // types
    mlir::tt::ttnn::LegalTensorLayoutAnalysis legalTensorLayoutAnalysis =
        getAnalysis<mlir::tt::ttnn::LegalTensorLayoutAnalysis>();
    legalTensorLayoutAnalysis.init(
        mlir::tt::ttnn::LegalTensorLayoutAnalysisInput(deviceGrid, &scalarTypes,
                                                       rowMajorEnabled));
    TensorTypeLayoutsMap tensorTypePossibleLayouts =
        legalTensorLayoutAnalysis.getResult();

    tracePossibleLayouts(tensorTypePossibleLayouts);

    moduleOp->walk([&](func::FuncOp func) {
      // Filter out all const-eval functions.
      if (ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      func->walk([&](Operation *op) {
        if (!LegalOpLayoutAnalysis::isValidAnalysisTarget(op)) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());

        // Get all possible layouts for this tensor type
        // Use layouts from the global analysis instead of regenerating per-op
        auto tensorLayouts = tensorTypePossibleLayouts.find(tensorType);
        bool hasLayoutsForTensorType =
            (tensorLayouts != tensorTypePossibleLayouts.end());

        assert(hasLayoutsForTensorType && "No layouts found for tensor type");

        // Run legal layout analysis to select the best layouts
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

        // Save only L1 Interleaved legal configs in a separate map for
        // L1InterleavedAnalysis later
        if (l1InterleavedAnalysisEnabled) {
          std::vector<OpConfig> l1InterleavedConfigs;
          for (const auto &config : legalConfigs[op]) {
            auto layoutAttr = config.outputLayout;
            if (layoutAttr.getBufferType() == BufferType::L1 &&
                layoutAttr.getMemLayout().getValue() ==
                    TensorMemoryLayout::Interleaved) {
              l1InterleavedConfigs.push_back(config);
            }
          }
          l1InterleavedLegalConfigs[op] = std::move(l1InterleavedConfigs);
        }
      });
    });

    llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> opSchedule;
    llvm::DenseMap<Edge, MemReconfigEntry> memReconfigEntryMap;
    std::vector<Operation *> spillToDramOps;

    // Extract override resharding edges
    //
    llvm::DenseSet<Edge> overrideReshardEdges;
    extractReshardEdges(moduleOp, overrideReshardEdges);

    if (memoryLayoutAnalysisEnabled) {
      // Perform memory layout analysis.
      //
      MemoryLayoutAnalysis memoryLayoutAnalysis =
          getAnalysis<MemoryLayoutAnalysis>();
      memoryLayoutAnalysis.init(MemoryLayoutAnalysisInput(
          &tensorTypePossibleLayouts, legalConfigs, chipDesc.getUsableL1Size(),
          overrideReshardEdges, memoryLayoutAnalysisPolicy));
      legalConfigs = memoryLayoutAnalysis.getResult().legalConfigs;
      opSchedule = memoryLayoutAnalysis.getResult().schedule;
      memReconfigEntryMap =
          memoryLayoutAnalysis.getResult().memReconfigEntryMap;
      spillToDramOps = memoryLayoutAnalysis.getResult().spillToDramOps;
    }

    // Manually overriden resharding edges should be added to the
    // memReconfigEntryMap.
    //
    for (const auto &edge : overrideReshardEdges) {
      if (memReconfigEntryMap.count(edge) == 0) {
        auto newEntry = MemReconfigEntry();
        newEntry.setOverridenReconfig(true);
        memReconfigEntryMap.try_emplace(edge, newEntry);
      }
    }

    // Pick optimal op configuration.
    //
    OpConfigAnalysis opConfigAnalysis = getAnalysis<OpConfigAnalysis>();
    opConfigAnalysis.init(OpConfigAnalysisInput(std::move(legalConfigs)));

    // Pure application of determined grid sizes to the operations.
    // No further analysis.
    //
    moduleOp->walk([&](func::FuncOp func) {
      if (ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      // If schedule is set, apply order of operations to func.
      //
      if (opSchedule[func].size() > 1) {

        for (size_t i = 0; i < opSchedule[func].size() - 1; i++) {
          Operation *op = opSchedule[func][i];

          Operation *nextOp = opSchedule[func][i + 1];
          nextOp->moveAfter(op);

          // Move DPS operand with the op.
          //
          if (isa<mlir::DestinationStyleOpInterface>(nextOp)) {
            nextOp->getOperands().back().getDefiningOp()->moveBefore(nextOp);
          }
        }
      }

      func->walk([&](Operation *op) {
        if (op->getNumResults() == 0) {
          // Skip ops with no results.
          return;
        }

        // Skip empty ops. Handled via DPS op output operand update.
        //
        if (isa<EmptyOp>(op)) {
          return;
        }

        if (!isa<RankedTensorType>(op->getResult(0).getType())) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());
        llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

        // Update the output layout attribute with the new one.
        //
        if (opConfigAnalysis.getResult().contains(op)) {
          RankedTensorType newTensorType = RankedTensorType::get(
              tensorShape,
              opConfigAnalysis.getResult()
                  .at(op)
                  .outputLayout.getScalarElementType(),
              opConfigAnalysis.getResult().at(op).outputLayout);

          // Update the memory space and layout of the op.
          //
          TTNNLayoutAttr layoutAttr =
              mlir::cast<TTNNLayoutAttr>(newTensorType.getEncoding());

          op->getResult(0).setType(newTensorType);

          // Update output data type for ops that have output data type
          // attribute.
          if (op->hasTrait<HasOutputDTypeTrait>()) {
            ttcore::DataTypeAttr newDataTypeAttr = ttcore::DataTypeAttr::get(
                op->getContext(), layoutAttr.getDataType());
            op->setAttr(HasOutputDTypeTraitBase::getOutputDTypeAttributeName(),
                        newDataTypeAttr);
          }

          // Update DPS operand layout as well.
          //
          if (isa<mlir::DestinationStyleOpInterface>(op)) {
            BufferType bufferType = layoutAttr.getBufferType();
            TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
                layoutAttr.getMemLayout();

            op->getOperands().back().setType(newTensorType);
            EmptyOp emptyOp =
                mlir::cast<EmptyOp>(op->getOperands().back().getDefiningOp());

            emptyOp.setDtype(layoutAttr.getDataType());
            if (layoutAttr.isTiled()) {
              emptyOp.setLayout(ttnn::Layout::Tile);
            } else {
              emptyOp.setLayout(ttnn::Layout::RowMajor);
            }

            emptyOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
                op->getContext(), tensorMemoryLayoutAttr,
                BufferTypeAttr::get(op->getContext(), bufferType),
                utils::createShardSpecIfNeeded(layoutAttr, deviceGrid)));
          }
          // TODO(mtopalovic): Temp workaround for generic ToLayoutOp. Allign
          // MemoryConfigAttr with layout attribute of its output tensor. This
          // redundant info should be removed or made consistent as part of temp
          // ToLayoutOp decomposition pass.
          //
          else if (isa<ttnn::ToLayoutOp>(op)) {
            BufferType bufferType = layoutAttr.getBufferType();
            TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
                layoutAttr.getMemLayout();

            // Update the device op with the new tensor type.
            //
            ttnn::ToLayoutOp toLayoutOp = llvm::cast<ttnn::ToLayoutOp>(op);
            toLayoutOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
                op->getContext(), tensorMemoryLayoutAttr,
                ttnn::BufferTypeAttr::get(op->getContext(), bufferType),
                utils::createShardSpecIfNeeded(layoutAttr, deviceGrid)));
          }

          // Set specific Conv2d Op configuration if it is exists.
          //

          if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(op)) {
            auto opAttributes = opConfigAnalysis.getResult().at(op);
            if (std::holds_alternative<ttnn::Conv2dAttrs>(
                    opAttributes.opSpecificAttrs)) {
              ttnn::Conv2dAttrs conv2dAttrs =
                  std::get<ttnn::Conv2dAttrs>(opAttributes.opSpecificAttrs);
              if (conv2dAttrs.conv2dConfig.has_value()) {
                conv2dOp.setConv2dConfigAttr(conv2dAttrs.conv2dConfig.value());
              }
              if (conv2dAttrs.deviceComputeKernelConfig.has_value()) {
                conv2dOp.setComputeConfigAttr(
                    conv2dAttrs.deviceComputeKernelConfig.value());
              }
            }
          }
        }
      });

      llvm::DenseMap<Operation *, Operation *> insertedMemoryReconfigOps;
      if (memReconfigEnabled) {
        insertedMemoryReconfigOps =
            processMemReconfigEdges(memReconfigEntryMap, deviceGrid);
      }

      processSpillOps(spillToDramOps, deviceGrid, insertedMemoryReconfigOps);

      // Try finding ops that can be upgraded from DRAM to L1 interleaved
      // layout.
      if (l1InterleavedAnalysisEnabled) {
        L1InterleavedAnalysis l1InterleavedAnalysis =
            getAnalysis<L1InterleavedAnalysis>();
        l1InterleavedAnalysis.init(L1InterleavedAnalysisInput(
            l1InterleavedLegalConfigs, opConfigAnalysis.getResult(), func,
            chipDesc.getUsableL1Size()));
        auto l1InterleavedOpConfigs =
            l1InterleavedAnalysis.getResult().upgradedConfigs;

        // Apply L1 interleaved layout changes
        for (const auto &[op, config] : l1InterleavedOpConfigs) {
          RankedTensorType tensorType =
              mlir::cast<RankedTensorType>(op->getResult(0).getType());
          TTNNLayoutAttr currentLayout =
              mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

          assert(currentLayout.getBufferType() == BufferType::DRAM &&
                 "Operation should have DRAM layout before upgrade to L1 "
                 "interleaved");

          TTNNLayoutAttr newLayout = config.outputLayout;
          assert(newLayout.getBufferType() == BufferType::L1 &&
                 newLayout.getMemLayout().getValue() ==
                     TensorMemoryLayout::Interleaved &&
                 "New layout must have L1 interleaved memory layout");

          RankedTensorType newTensorType = RankedTensorType::get(
              tensorType.getShape(), tensorType.getElementType(), newLayout);

          op->getResult(0).setType(newTensorType);
        }
      }

      insertRowMajorLayouts(func, chipDesc.getUsableL1Size());

      SmallVector<Type> funcResultTypes;

      // Pick up return op result types and update func type.
      func->walk([&](Operation *op) {
        if (op->getNumResults() == 0) {
          func::ReturnOp funcReturn = dyn_cast<func::ReturnOp>(op);
          if (funcReturn) {
            funcResultTypes.append(funcReturn.getOperandTypes().begin(),
                                   funcReturn.getOperandTypes().end());
          }
          return;
        }
      });

      // Update the function type to reflect the updated return operation's
      // result types.
      //
      FunctionType funcType = func.getFunctionType();
      FunctionType newFuncType = FunctionType::get(
          func.getContext(), funcType.getInputs(), funcResultTypes);
      func.setType(newFuncType);
    });
  }

private:
  void assertOverridesValid() {
    // Check if each overriden op exists in the graph.
    // Check if each conv2d config override is applied only to conv2d op.
    //
    llvm::StringMap<bool> overridenOpExists;
    llvm::StringMap<bool> overrideConv2dOp;
    for (const auto &[opLoc, _] : overrideOutputLayout) {
      overridenOpExists[opLoc] = false;
    }
    for (const auto &[opLoc, _] : insertMemReconfig) {
      overridenOpExists[opLoc] = false;
    }
    for (const auto &[opLoc, _] : overrideConv2dConfig) {
      overridenOpExists[opLoc] = false;
      overrideConv2dOp[opLoc] = false;
    }

    ModuleOp moduleOp = getOperation();
    moduleOp->walk([&](Operation *op) {
      if (not isa<NameLoc>(op->getLoc())) {
        return;
      }

      StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
      if (overridenOpExists.contains(opLocName)) {
        overridenOpExists[opLocName] = true;
      }
      if (!isa<ttnn::Conv2dOp>(op) && overrideConv2dOp.contains(opLocName)) {
        op->emitRemark() << "Trying to override non-conv2d op: '"
                         << op->getName()
                         << "' with conv2d config. Skipping...";
        return;
      }
    });

    for (const auto &[opLoc, opOverridenAndExists] : overridenOpExists) {
      if (!opOverridenAndExists) {
        llvm::report_fatal_error("Trying to override non-existing op: " +
                                 opLoc + ". Check logs for details");
      }
    }
  }

  void extractReshardEdges(ModuleOp &moduleOp,
                           llvm::DenseSet<Edge> &overrideReshardEdges) {
    moduleOp->walk([&](Operation *op) {
      if (isa<ToLayoutOp>(op)) {
        return;
      }

      // Skip ops without location
      //
      if (!isa<NameLoc>(op->getLoc())) {
        return;
      }

      StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
      auto opInputOverride = insertMemReconfig.find(opLocName);

      if (opInputOverride == insertMemReconfig.end()) {
        return;
      }

      SmallVector<int64_t> operandIndexes =
          opInputOverride->getValue().operandIdxes;
      for (int64_t operandIndex : operandIndexes) {
        Value operand = op->getOperand(operandIndex);
        Operation *operandOp = operand.getDefiningOp();
        overrideReshardEdges.insert(Edge(operandOp, op, operandIndex));
      }
    });

    // Check for non-existing ops in override
    //
    assert(insertMemReconfig.size() == overrideReshardEdges.size());
  }

  static llvm::DenseMap<Operation *, Operation *> processMemReconfigEdges(
      const llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap,
      ttcore::GridAttr deviceGrid) {

    // Mapping from producer op to inserted memory reconfig op.
    llvm::DenseMap<Operation *, Operation *> insertedMemoryReconfigOps;

    // Insert memory reconfig ops here based on results of memory layout
    // analysis.
    //
    for (const auto &[edge, memReconfigEntry] : memReconfigEntryMap) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "Processing mem reconfig edge: {}", edge);

      Operation *producerOp = edge.producerOp;
      Operation *consumerOp = edge.consumerOp;

      const bool overridenReconfig = memReconfigEntry.hasOverridenReconfig();
      assert(overridenReconfig ||
             memReconfigEntry.hasSelectedReshardOutputConfigBitIndex());

      // If there's no producer, tensor is defined by the consumer op input
      // operand.
      RankedTensorType producerOpTensorType =
          producerOp
              ? mlir::cast<RankedTensorType>(producerOp->getResult(0).getType())
              : mlir::cast<RankedTensorType>(
                    consumerOp->getOperand(edge.operandIndex).getType());

      llvm::ArrayRef<int64_t> producerOpTensorShape =
          producerOpTensorType.getShape();

      // Pick first layout from the list of layouts or consumer output layout if
      // this is the case of an override.
      TTNNLayoutAttr producerOpLayout =
          overridenReconfig
              ? mlir::cast<TTNNLayoutAttr>(
                    mlir::cast<RankedTensorType>(
                        consumerOp->getResult(0).getType())
                        .getEncoding())
              : memReconfigEntry.reshardOutputConfigMap
                    .at(memReconfigEntry
                            .getSelectedReshardOutputConfigBitIndex())
                    .front()
                    .outputLayout;

      // TODO(nobradovic): Match memory space and layout of consumer op.
      // This actually needs to be properly resolved based on op type, output
      // layout and other inputs.
      //
      RankedTensorType newTensorType = RankedTensorType::get(
          producerOpTensorShape, producerOpTensorType.getElementType(),
          producerOpLayout);

      MemoryConfigAttr outputMemConfigAttr = MemoryConfigAttr::get(
          consumerOp->getContext(), producerOpLayout.getMemLayout(),
          BufferTypeAttr::get(consumerOp->getContext(),
                              producerOpLayout.getBufferType()),
          utils::createShardSpecIfNeeded(producerOpLayout, deviceGrid));

      // If producerOp is a toLayoutOp, adjust its output layout(update
      // inplace) to reflect consumerOp's output layout. If producerOp is not a
      // toLayoutOp, insert a toLayoutOp in between producerOp
      // and consumerOp.
      //
      if (isa_and_nonnull<ToLayoutOp>(producerOp)) {
        ToLayoutOp toLayoutOp = llvm::cast<ToLayoutOp>(producerOp);
        toLayoutOp.setLayout(producerOpLayout.getLayout());
        toLayoutOp.setMemoryConfigAttr(outputMemConfigAttr);
        toLayoutOp.getResult().setType(newTensorType);
      } else {
        OpBuilder builder(consumerOp);
        Location loc = ttmlir::utils::appendLocationSuffix(consumerOp->getLoc(),
                                                           "_mem_reconfig");
        Operation *memoryReconfigOp = builder.create<ToLayoutOp>(
            loc,
            newTensorType,                             // output type
            consumerOp->getOperand(edge.operandIndex), // input value
            LayoutAttr::get(consumerOp->getContext(),
                            producerOpLayout.getLayout()),
            ttcore::DataTypeAttr::get(consumerOp->getContext(),
                                      producerOpLayout.getDataType()),
            outputMemConfigAttr);

        consumerOp->setOperand(edge.operandIndex,
                               memoryReconfigOp->getResult(0));
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Inserted memory reconfig op: {}",
                     mlir::cast<ToLayoutOp>(memoryReconfigOp));
        if (producerOp) {
          insertedMemoryReconfigOps[producerOp] = memoryReconfigOp;
        }
      }
    }
    return insertedMemoryReconfigOps;
  }

  void processSpillOps(const std::vector<Operation *> &spillToDramOps,
                       ttcore::GridAttr deviceGrid,
                       const llvm::DenseMap<Operation *, Operation *>
                           &insertedMemoryReconfigOps) {

    for (Operation *spilledOp : spillToDramOps) {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "Processing spilled op: {} {} @{}", spilledOp,
                   spilledOp->getName(), spilledOp->getLoc());

      RankedTensorType tensorType =
          mlir::cast<RankedTensorType>(spilledOp->getResult(0).getType());
      llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();
      TTNNLayoutAttr layoutAttr =
          mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

      // Create a new tensor type with DRAM layout.
      TTNNLayoutAttr dramLayout =
          layoutAttr.withBufferType(BufferType::DRAM)
              .withMemoryLayout(TensorMemoryLayout::Interleaved);
      RankedTensorType newTensorType = RankedTensorType::get(
          tensorShape, tensorType.getElementType(), dramLayout);

      // Create a ToLayoutOp with the new DRAM layout.
      OpBuilder builder(spilledOp->getContext());
      ttcore::DataTypeAttr dataType = ttcore::DataTypeAttr::get(
          spilledOp->getContext(), dramLayout.getDataType());
      LayoutAttr newLayout =
          LayoutAttr::get(spilledOp->getContext(), dramLayout.getLayout());

      MemoryConfigAttr memConfigAttr = MemoryConfigAttr::get(
          spilledOp->getContext(), dramLayout.getMemLayout(),
          BufferTypeAttr::get(spilledOp->getContext(), BufferType::DRAM),
          utils::createShardSpecIfNeeded(dramLayout, deviceGrid));

      builder.setInsertionPointAfter(spilledOp);
      Location loc =
          ttmlir::utils::appendLocationSuffix(spilledOp->getLoc(), "_spill");

      // Step 1: Save all uses as (Operation*, operand index).
      llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
      for (auto &use : spilledOp->getResult(0).getUses()) {
        uses.emplace_back(use.getOwner(), use.getOperandNumber());
      }

      // Step 2: Insert spilling to DRAM.
      Operation *spillToDRAMOp = builder.create<ToLayoutOp>(
          loc, newTensorType, spilledOp->getResult(0), newLayout, dataType,
          memConfigAttr);

      // Step 3: Reconnect uses.
      for (auto &use : uses) {
        Operation *useOp = use.first;
        useOp->setOperand(use.second, spillToDRAMOp->getResult(0));
      }

      // TODO(rpavlovicTT): It's possible that spilled op was followed by
      // memory reconfig ops. It's also possible that spilled op is a fork and
      // has memory reconfig ops on more than one branch. In that case we should
      // wire through each memory reconfig op and choose one which can inherit
      // L1 tensor. Rest of the users must read from tensor spilled to DRAM.
      // Right now we save only last memory reconfig op and try to skip spill
      // to DRAM on that branch.
      if (insertedMemoryReconfigOps.count(spilledOp)) {
        // There is a memory reconfig op inserted on a branch outgoing from
        // spilled op. We will avoid spill to DRAM on that branch.
        Operation *memoryReconfigOp = insertedMemoryReconfigOps.at(spilledOp);
        auto spilledOpResultLayout = mlir::cast<TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(spilledOp->getResult(0).getType())
                .getEncoding());
        assert(spilledOpResultLayout &&
               "Expected spilled op result to have layout");
        auto memoryReconfigOpLayout = mlir::cast<TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(
                memoryReconfigOp->getResult(0).getType())
                .getEncoding());
        assert(memoryReconfigOpLayout &&
               "Expected memory reconfig op result to have layout");

        if (memoryReconfigOpLayout == spilledOpResultLayout) {
          // Memory reconfig op has same layout as spilled op. We will avoid
          // memory reconfig op on that branch. Data will flow directly from
          // spilled op to consumer of memory reconfig op and will not leave L1.
          assert(memoryReconfigOp->hasOneUse() &&
                 "Expected memory reconfig op to have one use");
          auto memoryReconfigOpConsumerOperand =
              memoryReconfigOp->getResult(0).getUses().begin();

          Operation *memoryReconfigOpConsumerOp =
              memoryReconfigOpConsumerOperand->getOwner();
          memoryReconfigOpConsumerOp->setOperand(
              memoryReconfigOpConsumerOperand->getOperandNumber(),
              spilledOp->getResult(0));

          // Ensure memory reconfig op consumer is after spilling to DRAM to
          // avoid losing data. This is a sanity check. Schedule should
          // guarantee valid order.
          assert(
              spillToDRAMOp->isBeforeInBlock(memoryReconfigOpConsumerOp) &&
              "Memory reconfig op consumer should be after spilling to DRAM");

          memoryReconfigOp->erase();
          TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                       "Erased memory reconfig op: {}@{}",
                       memoryReconfigOp->getName(), memoryReconfigOp->getLoc());
        } else {
          // Memory reconfig op does not have the same layout as spilled op.
          // We can't get rid of memory reconfig op but we can avoid reading
          // from DRAM on that branch. MemoryReconfigOp has only one operand and
          // we will wire it to spilled op's result.
          memoryReconfigOp->setOperand(0, spilledOp->getResult(0));
        }
      }
    }
  }

  // Check if the op can be executed with row major layout on the input. Returns
  // expected output layout if constraints are satisfied, otherwise returns
  // `nullptr`.
  // TODO(rpavlovicTT) https://github.com/tenstorrent/tt-mlir/issues/3972
  TTNNLayoutAttr checkOpConstraints(Operation *op,
                                    std::vector<TTNNLayoutAttr> inputLayouts,
                                    size_t l1CacheSize,
                                    bool convertInputToRowMajor) {

    if (convertInputToRowMajor) {
      inputLayouts[0] = utils::convertTTNNLayoutToRowMajor(
          op->getContext(), inputLayouts[0],
          mlir::cast<RankedTensorType>(op->getOperand(0).getType()).getShape());
    }

    OpModel backend = mlir::dyn_cast<OpModel>(*op);
    assert(backend && "Backend constraints are not implemented for op");

    // Empty consumerConfig with conv2d config if conv2d op.
    OpConfig consumerConfig;
    if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(op)) {
      Conv2dAttrs conv2dAttrs = {conv2dOp.getConv2dConfigAttr(),
                                 conv2dOp.getComputeConfigAttr()};
      consumerConfig.opSpecificAttrs = conv2dAttrs;
    }

    auto opConstraintsResult =
        backend.getOpConstraints(inputLayouts, consumerConfig);

    if (!opConstraintsResult) {
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Failed constraints call after: {}", op->getLoc());
      op->emitWarning("Failed constraints call after: " +
                      llvm::toString(opConstraintsResult.takeError()));
      return nullptr;
    }

    auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
        opConstraintsResult.get();
    constexpr float tensorL1UsageCap = 0.8;
    bool l1UsageValid =
        (outputTensorUsage + cBUsagePeak) < tensorL1UsageCap * l1CacheSize;
    if (!l1UsageValid) {
      op->emitWarning("L1 usage exceeded with " +
                      std::to_string(outputTensorUsage + cBUsagePeak) +
                      " out of " + std::to_string(l1CacheSize) +
                      " scaled down to " +
                      std::to_string(tensorL1UsageCap * l1CacheSize));
      return nullptr;
    }

    return outputLayout;
  }

  // Surround op with memory reconfig ops that convert tensor to row major
  // layout for the op and revert the result back to tile layout. This will be
  // used as a workaround for MaxPool2d op which works with row major layout.
  void convertOpToRowMajorAndBack(Operation *op) {
    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);

    assert(op->getNumOperands() == 1 && "Expected exactly one operand");
    auto operand = op->getOperand(0);
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(operand.getType());
    TTNNLayoutAttr inputLayout =
        mlir::cast<TTNNLayoutAttr>(inputType.getEncoding());

    assert(inputLayout.hasInterleavedDRAMTensorMemoryLayout() &&
           "Expected interleaved DRAM tensor memory layout");

    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    TTNNLayoutAttr outputLayout =
        mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());

    // Make a new layout with the same shape but row major.
    TTNNLayoutAttr inputRowMajorLayout = utils::convertTTNNLayoutToRowMajor(
        op->getContext(), inputLayout, inputType.getShape());
    RankedTensorType newInputTensorType = RankedTensorType::get(
        inputType.getShape(), inputRowMajorLayout.getElementType(),
        inputRowMajorLayout);

    Location loc =
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_to_rm_before");

    // TODO(rpavlovicTT) https://github.com/tenstorrent/tt-mlir/issues/3973
    Operation *memoryReconfigOpBefore = builder.create<ttnn::ToLayoutOp>(
        loc, newInputTensorType, operand,
        LayoutAttr::get(op->getContext(), Layout::RowMajor),
        ttcore::DataTypeAttr::get(op->getContext(), inputLayout.getDataType()),
        MemoryConfigAttr::get(
            op->getContext(), inputLayout.getMemLayout(),
            BufferTypeAttr::get(op->getContext(), BufferType::DRAM),
            /*shardSpec=*/std::nullopt));
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Inserted memory reconfig before, type: {}",
                 memoryReconfigOpBefore->getResult(0).getType());

    // Update op's operand.
    op->setOperand(0, memoryReconfigOpBefore->getResult(0));

    // Create new tensor type for the op's result. Reuse element type from the
    // input.
    TTNNLayoutAttr outputRowMajorLayout = outputLayout.withElementType(
        inputRowMajorLayout.getElementType(),
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape());

    Type newTensorType = RankedTensorType::get(
        outputType.getShape(), outputRowMajorLayout.getElementType(),
        outputRowMajorLayout);

    // Replace op's encoding with row major.
    op->getResult(0).setType(newTensorType);
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Changed encoding of {}@{} to row major {}", op->getName(),
                 op->getLoc(), outputRowMajorLayout);

    // Save uses of op's result.
    llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
    for (auto &use : op->getResult(0).getUses()) {
      uses.emplace_back(use.getOwner(), use.getOperandNumber());
    }

    // Add another memory reconfig after op. This op will revert tensor back to
    // tile layout.
    builder.setInsertionPointAfter(op);
    Operation *memoryReconfigOpAfter = builder.create<ttnn::ToLayoutOp>(
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_to_rm_after"),
        outputType, op->getResult(0),
        LayoutAttr::get(op->getContext(), Layout::Tile),
        ttcore::DataTypeAttr::get(op->getContext(), outputLayout.getDataType()),
        MemoryConfigAttr::get(
            op->getContext(), outputLayout.getMemLayout(),
            BufferTypeAttr::get(op->getContext(), BufferType::DRAM),
            /*shardSpec=*/std::nullopt));

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Inserted memory reconfig after, type: {}",
                 memoryReconfigOpAfter->getResult(0).getType());

    // Update all uses of op's result.
    for (auto &use : uses) {
      Operation *useOp = use.first;
      useOp->setOperand(use.second, memoryReconfigOpAfter->getResult(0));
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Updated use: {}@{} input type: {}", useOp->getName(),
                   useOp->getLoc(), useOp->getOperand(use.second).getType());
    }
  }

  // Walks graph and for specific op types (MaxPool2d and Upsample) inserts
  // necessary memory reconfigurations to convert tensors to row major layout.
  // Note: in the long term this can be an analysis that is used by the
  // optimizer to determine if a memory reconfig is needed for non-sharded ops.
  void insertRowMajorLayouts(func::FuncOp func, unsigned l1CacheSize) {
    func->walk([&](Operation *op) {
      if (!isa<ttnn::MaxPool2dOp>(op) && !isa<ttnn::UpsampleOp>(op)) {
        return;
      }

      RankedTensorType resultType =
          mlir::cast<RankedTensorType>(op->getResult(0).getType());
      TTNNLayoutAttr resultLayout =
          mlir::cast<TTNNLayoutAttr>(resultType.getEncoding());

      if (resultLayout.hasShardedTensorMemoryLayout()) {
        return;
      }

      size_t numOperands = op->getNumOperands();
      assert(numOperands > 0 && "Expected at least one operand");
      std::vector<TTNNLayoutAttr> inputLayouts;
      for (size_t i = 0; i < numOperands; i++) {
        auto operand = op->getOperand(i);

        if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
          // Skip device type operand.
          continue;
        }

        RankedTensorType input =
            mlir::cast<RankedTensorType>(operand.getType());
        auto layout = ttnn::utils::getLayoutAttrFromTensor(input);

        assert(layout && "Input operand must have a layout");
        inputLayouts.push_back(layout);
      }
      assert(inputLayouts.size() > 0 && "Expected at least one input");

      // Both input and output layout has to be checked otherwise some
      // inconsistencies may arise:
      // https://github.com/tenstorrent/tt-mlir/issues/4051
      if ((!inputLayouts[0].isTiled() ||
           inputLayouts[0].hasShardedTensorMemoryLayout()) &&
          !resultLayout.isTiled()) {
        // Input is already in RowMajor or has sharded tensor memory layout, no
        // need to convert.
        return;
      }

      if (isa<ttnn::MaxPool2dOp>(op)) {
        // Unequivocally surround MaxPool2d with RM. Ideally we should query op
        // constraints to see if RM is supported. But issue
        // https://github.com/tenstorrent/tt-metal/issues/24358 blocks usage
        // of getOpConstraints for MaxPool2d.
        // TODO(rpavlovicTT): fix it once getOpConstraints is supported for
        // MaxPool2d.
        convertOpToRowMajorAndBack(op);
        return;
      }

      // Let's check first if the op can be executed with the current layout.
      if (auto actualOutputLayout =
              checkOpConstraints(op, inputLayouts, l1CacheSize,
                                 /*convertInputToRowMajor=*/false)) {
        // If output layout is different from the expected one, we need to
        // convert the output type to the expected one.
        if (actualOutputLayout != resultLayout) {
          op->getResult(0).setType(
              resultType.cloneWithEncoding(actualOutputLayout));
        }
        TTMLIR_DEBUG(
            ttmlir::LogComponent::Optimizer,
            "Op {} successfully passed constraints, no conversion needed",
            op->getName());
        return;
      }

      // Failed to satisfy constraints, try with row major layout for the input
      // operand.
      if (!checkOpConstraints(op, inputLayouts, l1CacheSize,
                              /*convertInputToRowMajor=*/true)) {
        op->emitOpError(
            "Failed to satisfy constraints with Tile and RM layouts");
        signalPassFailure();
        return;
      }

      // Row major input passed constraints, let's add necessary conversions.
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Op {} successfully passed constraints after inserting RM",
                   op->getName());
      convertOpToRowMajorAndBack(op);
    });
  }

  // Trace all possible layouts for debugging
  static void
  tracePossibleLayouts(const TensorTypeLayoutsMap &allPossibleLayouts) {
    if (!llvm::DebugFlag || !isLogLevelEnabled(ttmlir::LogLevel::Trace)) {
      return;
    }
    for (auto &[tensorType, layouts] : allPossibleLayouts) {
      for (auto &[scalarType, layoutsByDataLayout] : layouts) {
        for (size_t dataLayoutIdx = 0;
             dataLayoutIdx < static_cast<size_t>(TensorPageLayout::kNumValues);
             ++dataLayoutIdx) {
          // Get data layout enum name for debugging
          std::string dataLayoutName =
              dataLayoutIdx == static_cast<size_t>(TensorPageLayout::Tiled)
                  ? "Tiled"
                  : "RowMajor";

          TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "data layout {}",
                       dataLayoutName);

          for (size_t memLayoutIdx = 0;
               memLayoutIdx <
               static_cast<size_t>(TensorMemoryLayoutIndex::kNumValues);
               ++memLayoutIdx) {
            std::string memLayoutName =
                memLayoutIdx == static_cast<size_t>(
                                    TensorMemoryLayoutIndex::Interleaved)
                    ? "Interleaved"
                    : "Sharded";

            auto &layouts = layoutsByDataLayout[dataLayoutIdx][memLayoutIdx];
            for (const TTNNLayoutAttr &layout : layouts) {
              (void)layout;
              TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                           "Tensor type {0}, scalar type {1}, data layout "
                           "{2}, memory layout "
                           "{3} has layout {4}",
                           tensorType, scalarType, dataLayoutName,
                           memLayoutName, layout);
            }
          }
        }
      }
    }
  }
};

} // namespace mlir::tt::ttnn
