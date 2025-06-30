// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/AllPossibleLayoutsAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalLayoutAnalysis.h"
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
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <llvm/Support/Error.h>
#include <mlir/Support/LLVM.h>

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

    // Step 2: Run AllPossibleLayoutsAnalysis to generate layouts for all tensor
    // types
    mlir::tt::ttnn::AllPossibleLayoutsAnalysis allPossibleLayoutsAnalysis =
        getAnalysis<mlir::tt::ttnn::AllPossibleLayoutsAnalysis>();
    allPossibleLayoutsAnalysis.init(
        mlir::tt::ttnn::AllPossibleLayoutsAnalysisInput(
            deviceGrid, &scalarTypes, rowMajorEnabled));
    TensorTypeLayoutsMap tensorTypePossibleLayouts =
        allPossibleLayoutsAnalysis.getResult();

    tracePossibleLayouts(tensorTypePossibleLayouts);

    moduleOp->walk([&](func::FuncOp func) {
      // Filter out all const-eval functions.
      if (ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      func->walk([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }

        if (!isa<RankedTensorType>(op->getResult(0).getType())) {
          return;
        }

        if (llvm::isa<ttnn::EmptyOp>(op)) {
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
        LegalLayoutAnalysis legalLayoutAnalysis =
            getChildAnalysis<LegalLayoutAnalysis>(op);
        legalLayoutAnalysis.init(LegalLayoutAnalysisInput(
            &tensorLayouts->getSecond(), maxLegalLayouts, &overrideOutputLayout,
            &overrideConv2dConfig, rowMajorEnabled));
        legalConfigs[op] = legalLayoutAnalysis.getResult();
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
            if (auto conv2dConfig =
                    mlir::dyn_cast_if_present<ttnn::Conv2dConfigAttr>(
                        opConfigAnalysis.getResult().at(op).opSpecificAttr)) {
              conv2dOp.setConv2dConfigAttr(conv2dConfig);
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

  static mlir::TypedValue<DeviceType>
  getOrCreateDeviceOpValue(Operation *contextOp, OpBuilder &builder) {
    Block *block = contextOp->getBlock();
    for (auto &op : block->getOperations()) {
      if (GetDeviceOp deviceOp = dyn_cast<GetDeviceOp>(op)) {
        return deviceOp.getResult();
      }
    }

    // Device op does not exist in the block, hence we need to create it.
    ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(contextOp);
    auto currentInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPoint(block, block->begin());
    llvm::SmallVector<int64_t> meshShape{deviceAttr.getMeshShape()};
    if (meshShape.empty()) {
      meshShape = llvm::SmallVector<int64_t, 2>{1, 1};
    }
    // TODO (jnie): Currently hardcoding the mesh offset to 0x0
    // Need a proper plan to dynamically determine this.
    llvm::SmallVector<int64_t, 2> meshOffset{0, 0};

    auto deviceOp = builder.create<ttnn::GetDeviceOp>(
        contextOp->getLoc(), builder.getType<DeviceType>(),
        ttnn::MeshShapeAttr::get(contextOp->getContext(), meshShape[0],
                                 meshShape[1]),
        ttnn::MeshOffsetAttr::get(contextOp->getContext(), meshOffset[0],
                                  meshOffset[1]));
    builder.restoreInsertionPoint(currentInsertionPoint);
    return deviceOp;
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
            outputMemConfigAttr, getOrCreateDeviceOpValue(consumerOp, builder));

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
          memConfigAttr, getOrCreateDeviceOpValue(spilledOp, builder));

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
    Type elementType = utils::getElementType(op->getContext(), Layout::RowMajor,
                                             inputLayout.getDataType());

    // Tensor incoming to op will be row major.
    TTNNLayoutAttr inputRowMajorLayout =
        inputLayout.withElementType(elementType, inputType.getShape());
    RankedTensorType newInputTensorType = RankedTensorType::get(
        inputType.getShape(), elementType, inputRowMajorLayout);

    Location loc =
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_to_rm_before");
    Operation *memoryReconfigOpBefore = builder.create<ttnn::ToLayoutOp>(
        loc, newInputTensorType, operand,
        LayoutAttr::get(op->getContext(), Layout::RowMajor),
        ttcore::DataTypeAttr::get(op->getContext(), inputLayout.getDataType()),
        MemoryConfigAttr::get(
            op->getContext(), inputLayout.getMemLayout(),
            BufferTypeAttr::get(op->getContext(), BufferType::DRAM),
            /*shardSpec=*/std::nullopt),
        getOrCreateDeviceOpValue(op, builder));
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Inserted memory reconfig before, type: {}",
                 memoryReconfigOpBefore->getResult(0).getType());

    // Update op's operand.
    op->setOperand(0, memoryReconfigOpBefore->getResult(0));

    // Create new tensor type for the op's result.
    TTNNLayoutAttr outputRowMajorLayout = outputLayout.withElementType(
        elementType,
        mlir::cast<RankedTensorType>(op->getOperand(0).getType()).getShape());
    Type newTensorType = RankedTensorType::get(
        outputType.getShape(), elementType, outputRowMajorLayout);

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
            /*shardSpec=*/std::nullopt),
        getOrCreateDeviceOpValue(op, builder));

    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "Inserted memory reconfig after, type: {}",
                 memoryReconfigOpAfter->getResult(0).getType());

    // Update all uses of op's result.
    for (auto &use : uses) {
      Operation *useOp = use.first;
      useOp->setOperand(use.second, memoryReconfigOpAfter->getResult(0));
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                   "Updated use: {}@{} input type: {}", useOp->getName(),
                   useOp->getLoc(),
                   useOp->getOperand(use.second).getType());
    }
  }

  void insertRowMajorLayouts(func::FuncOp func, unsigned l1CacheSize) {
    // Walk through graph
    // Pick ops that are not sharded
    // We should do this after spillToDram to avoid interfering with shard
    // chains but we should check if producer of input 0 is spillToDram, then we
    // can change to RM along with changing memory config
    //
    // 0. Call getOpConstraints for that op, without output config
    // 1. If it succeeds, check memory,
    //      * if OOM go to step 3,
    //      * else, declare success, move on to next op
    // 2. If it fails, go to step 3
    // 3. Change its input 0 to RM
    // 4. Go to step 0

    func->walk([&](Operation *op) {
      // if not a conv2d, maxpool2d, upsample skip
      // skipping conv2d op for now
      // processing only maxpool2d and upsample
      // 1. maxpool2d insert RM always in front and after the op and change
      // output layout to RM
      // 2. do check op constraints with current layout and insert RM only if
      // constraints are satisfied
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

      if (isa<ttnn::MaxPool2dOp>(op)) {
        // Unequivocally surround with RM. Ideally we should query ttnn via
        // getOpConstraints to see if RM is supported. But issue
        // https://github.com/tenstorrent/tt-metal/issues/24358 blocks usage
        // of getOpConstraints for MaxPool2d.
        // TODO(rpavlovicTT): fix it once getOpConstraints is supported for
        // MaxPool2d.
        convertOpToRowMajorAndBack(op);
        return;
      }

      std::vector<TTNNLayoutAttr> inputLayouts;

      for (size_t i = 0; i < numOperands; i++) {
        auto operand = op->getOperand(i);

        if (mlir::isa<TypedValue<mlir::tt::ttnn::DeviceType>>(operand)) {
          // Skip device type operand.
          continue;
        }

        RankedTensorType input =
            mlir::cast<RankedTensorType>(operand.getType());
        auto layout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());

        assert(layout && "Input operand must have a layout");
        inputLayouts.push_back(layout);
      }

      OpModel backend = mlir::dyn_cast<OpModel>(op);
      assert(backend && "Backend constraints are not implemented for op");

      // Empty consumerConfig with conv2d config if conv2d op.
      OpConfig consumerConfig;
      if (auto conv2dOp = mlir::dyn_cast<mlir::tt::ttnn::Conv2dOp>(op)) {
        consumerConfig.opSpecificAttr = conv2dOp.getConv2dConfigAttr();
      }

      llvm::Expected<op_model::ttnn::OpConstraints> opConstraintsResult =
          backend.getOpConstraints(inputLayouts, consumerConfig);

      bool failed = false;
      if (!opConstraintsResult) {
        failed = true;
        op->emitWarning(llvm::toString(opConstraintsResult.takeError()));
      }

      if (!failed) {
        auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
            opConstraintsResult.get();
        constexpr float tensorL1UsageCap = 0.8;
        bool l1UsageValid =
            (outputTensorUsage + cBUsagePeak) < tensorL1UsageCap * l1CacheSize;

        if (!l1UsageValid) {
          failed = true;
          op->emitWarning("L1 usage exceeded with " +
                          std::to_string(outputTensorUsage + cBUsagePeak) +
                          " out of " + std::to_string(l1CacheSize) +
                          " scaled down to " +
                          std::to_string(tensorL1UsageCap * l1CacheSize));
        }
      }

      if (!failed) {
        // Constraints are satisfied, move on to next op.
        return;
      }

      // Failed to satisfy constraints, try with row major layout for the input
      // operand.
      TTNNLayoutAttr actLayout = inputLayouts[0];

      Type elementType = utils::getElementType(
          op->getContext(), Layout::RowMajor, actLayout.getDataType());
      [[maybe_unused]] TTNNLayoutAttr rowMajorLayout =
          actLayout.withElementType(
              elementType,
              mlir::cast<RankedTensorType>(op->getOperand(0).getType())
                  .getShape());

      inputLayouts[0] = rowMajorLayout;
      auto rmOpConstraintsResult =
          backend.getOpConstraints(inputLayouts, consumerConfig);

      if (!rmOpConstraintsResult) {
        op->emitOpError("Failed constraints call after inserting RM: " +
                        llvm::toString(rmOpConstraintsResult.takeError()));
        signalPassFailure();
        return;
      }

      auto [cBUsagePeak, tensorUsage, outputTensorUsage, outputLayout] =
          rmOpConstraintsResult.get();
      constexpr float tensorL1UsageCap = 0.8;
      bool l1UsageValid =
          (outputTensorUsage + cBUsagePeak) < tensorL1UsageCap * l1CacheSize;
      if (!l1UsageValid) {
        op->emitOpError("Failed L1 usage after inserting RM: " +
                        llvm::toString(rmOpConstraintsResult.takeError()));
        signalPassFailure();
        return;
      }

      // Row major input passed constraints, let's add necessary conversions.
      TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer, "Successfully passed constraints after inserting RM");
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
