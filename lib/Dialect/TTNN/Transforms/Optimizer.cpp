// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

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
    overrideInputLayout = std::move(options.overrideInputLayout);
    overrideOutputLayout = std::move(options.overrideOutputLayout);
    memoryLayoutAnalysisEnabled =
        std::move(options.memoryLayoutAnalysisEnabled);
    memReconfigEnabled = std::move(options.memReconfigEnabled);
    memoryLayoutAnalysisPolicy = std::move(options.memoryLayoutAnalysisPolicy);
    maxLegalLayouts = std::move(options.maxLegalLayouts);
    rowMajorEnabled = std::move(options.rowMajorEnabled);
  }

protected:
  ::mlir::Pass::Option<llvm::StringMap<InputLayoutOverrideParams>,
                       mlir::tt::ttnn::InputLayoutOverrideParser>
      overrideInputLayout{
          *this, "insert-memreconfig",
          ::llvm::cl::desc(
              "Manually insert memory reconfig op for specific op's operand."),
          ::llvm::cl::init(llvm::StringMap<InputLayoutOverrideParams>())};
  ::mlir::Pass::Option<llvm::StringMap<OutputLayoutOverrideParams>,
                       mlir::tt::ttnn::OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, "override-output-layout",
          ::llvm::cl::desc("Override output tensor layout for specific ops."),
          ::llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};
  ::mlir::Pass::Option<bool> memoryLayoutAnalysisEnabled{
      *this, "memory-layout-analysis-enabled",
      ::llvm::cl::desc("Enable memory layout optimization."),
      ::llvm::cl::init(false)};
  ::mlir::Pass::Option<bool> memReconfigEnabled{
      *this, "memreconfig-enabled",
      ::llvm::cl::desc("Memory layout reconfiguration pass."),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<mlir::tt::MemoryLayoutAnalysisPolicyType,
                       mlir::tt::MemoryLayoutAnalysisPolicyTypeParser>
      memoryLayoutAnalysisPolicy{
          *this, "memory-layout-analysis-policy",
          llvm::cl::desc("Specify policy for memory layout analysis."),
          llvm::cl::init(MemoryLayoutAnalysisPolicyType::DFSharding)};
  ::mlir::Pass::Option<int64_t> maxLegalLayouts{
      *this, "max-legal-layouts",
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
    assert(moduleOp->hasAttr(tt::DeviceAttr::name));
    GridAttr max_grid =
        mlir::cast<tt::DeviceAttr>(moduleOp->getAttr(tt::DeviceAttr::name))
            .getWorkerGrid();

    SystemDescAttr systemDesc = mlir::cast<tt::SystemDescAttr>(
        moduleOp->getAttr(tt::SystemDescAttr::name));
    ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
    llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>> legalLayouts;

    moduleOp->walk([&](Operation *op) {
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
      LegalLayoutAnalysis legalLayoutAnalysis =
          getChildAnalysis<LegalLayoutAnalysis>(op);
      legalLayoutAnalysis.init(LegalLayoutAnalysisInput(
          chipDesc, max_grid, tensorType, maxLegalLayouts,
          &overrideOutputLayout, rowMajorEnabled));
      legalLayouts[op] = legalLayoutAnalysis.getResult();
    });

    llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> opSchedule;
    std::unordered_set<Edge> memReconfigEdges;
    if (memoryLayoutAnalysisEnabled) {
      // Extract override resharding edges
      //
      std::unordered_set<Edge> overrideReshardEdges;
      extractReshardEdges(moduleOp, overrideReshardEdges);

      // Perform memory layout analysis.
      //
      MemoryLayoutAnalysis memoryLayoutAnalysis =
          getAnalysis<MemoryLayoutAnalysis>();
      memoryLayoutAnalysis.init(MemoryLayoutAnalysisInput(
          legalLayouts, chipDesc.getUsableL1Size(), overrideReshardEdges,
          memoryLayoutAnalysisPolicy));
      legalLayouts = memoryLayoutAnalysis.getResult().legalLayouts;
      opSchedule = memoryLayoutAnalysis.getResult().schedule;
      memReconfigEdges = memoryLayoutAnalysis.getResult().memReconfigEdges;
    }

    // Pick optimal op configuration.
    //
    OpConfigAnalysis opConfigAnalysis = getAnalysis<OpConfigAnalysis>();
    opConfigAnalysis.init(OpConfigAnalysisInput(std::move(legalLayouts)));

    // Pure application of determined grid sizes to the operations.
    // No further analysis.
    //
    moduleOp->walk([&](func::FuncOp func) {
      SmallVector<Type> funcResultTypes;

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
          func::ReturnOp funcReturn = dyn_cast<func::ReturnOp>(op);
          if (funcReturn) {
            funcResultTypes.append(funcReturn.getOperandTypes().begin(),
                                   funcReturn.getOperandTypes().end());
          }
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
          RankedTensorType newTensorType =
              RankedTensorType::get(tensorShape, tensorType.getElementType(),
                                    opConfigAnalysis.getResult().at(op));

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
                op->getContext(),
                BufferTypeAttr::get(op->getContext(), bufferType),
                ShardSpecAttr::get(
                    op->getContext(),
                    ShapeAttr::get(op->getContext(),
                                   layoutAttr.getMemref().getShape())),
                tensorMemoryLayoutAttr));
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
                op->getContext(),
                ttnn::BufferTypeAttr::get(op->getContext(), bufferType),
                ttnn::ShardSpecAttr::get(
                    op->getContext(),
                    ttnn::ShapeAttr::get(op->getContext(),
                                         layoutAttr.getMemref().getShape())),
                tensorMemoryLayoutAttr));
          }
        }
      });

      if (memReconfigEnabled) {
        processMemReconfigEdges(memReconfigEdges);
      }

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
    //
    llvm::StringMap<bool> overridenOpExists;
    for (auto &override : overrideOutputLayout) {
      overridenOpExists[override.first()] = false;
    }
    for (auto &override : overrideInputLayout) {
      overridenOpExists[override.first()] = false;
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
    });

    for (auto &override : overridenOpExists) {
      if (!override.second) {
        llvm::errs() << "Trying to override non-existing op: "
                     << override.first() << "\n";
        assert(false && "Trying to override non-existing op");
      }
    }
  }

  void extractReshardEdges(ModuleOp &moduleOp,
                           std::unordered_set<Edge> &overrideReshardEdges) {
    moduleOp->walk([&](Operation *op) {
      if (!isa<DestinationStyleOpInterface>(op)) {
        return;
      }

      // Skip ops without location
      //
      if (!isa<NameLoc>(op->getLoc())) {
        return;
      }

      StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
      auto opInputOverride = overrideInputLayout.find(opLocName);

      if (opInputOverride == overrideInputLayout.end()) {
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
    assert(overrideInputLayout.size() == overrideReshardEdges.size());
  }

  mlir::TypedValue<mlir::tt::DeviceType>
  getDeviceOpValue(Operation *contextOp) {
    Block *block = contextOp->getBlock();
    mlir::TypedValue<mlir::tt::DeviceType> deviceOpResult;
    for (auto &op : block->getOperations()) {
      if (GetDeviceOp deviceOp = dyn_cast<GetDeviceOp>(op)) {
        deviceOpResult = deviceOp.getResult();
        break;
      }
    }
    return deviceOpResult;
  }

  void
  processMemReconfigEdges(const std::unordered_set<Edge> &memReconfigEdges) {
    // Insert memory reconfig ops here based on results of memory layout
    // analysis.
    //
    for (const Edge &edge : memReconfigEdges) {
      Operation *producerOp = edge.producerOp;
      Operation *consumerOp = edge.consumerOp;

      TTNNLayoutAttr consumerOpOutputLayout = mlir::cast<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType())
              .getEncoding());

      RankedTensorType producerOpTensorType =
          mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
      llvm::ArrayRef<int64_t> producerOpTensorShape =
          producerOpTensorType.getShape();
      TTNNLayoutAttr producerOpLayout =
          mlir::cast<TTNNLayoutAttr>(producerOpTensorType.getEncoding());

      // TODO(nobradovic): Match memory space and layout of consumer op.
      // This actually needs to be properly resolved based on op type, output
      // layout and other inputs.
      //
      RankedTensorType newTensorType = RankedTensorType::get(
          producerOpTensorShape, producerOpTensorType.getElementType(),
          producerOpLayout
              .withElementType(consumerOp->getContext(),
                               consumerOpOutputLayout.getElementType())
              .withBufferType(consumerOp->getContext(),
                              consumerOpOutputLayout.getBufferType())
              .withMemoryLayout(consumerOp->getContext(),
                                consumerOpOutputLayout.getMemLayout())
              .withGrid(consumerOp->getContext(), producerOpTensorType,
                        consumerOpOutputLayout.getGrid()));

      BufferType outputBufferType = consumerOpOutputLayout.getBufferType();
      TensorMemoryLayoutAttr outputTensorMemoryLayoutAttr =
          consumerOpOutputLayout.getMemLayout();

      llvm::SmallVector<int64_t> shardShape =
          consumerOpOutputLayout.getShardShape();
      MemoryConfigAttr outputMemConfigAttr = MemoryConfigAttr::get(
          consumerOp->getContext(),
          BufferTypeAttr::get(consumerOp->getContext(), outputBufferType),
          ShardSpecAttr::get(
              consumerOp->getContext(),
              ShapeAttr::get(consumerOp->getContext(), shardShape)),
          outputTensorMemoryLayoutAttr);

      // If producerOp is a toLayoutOp, adjust its output layout(update
      // inplace) to reflect consumerOp's output layout. If producerOp is not a
      // toLayoutOp, insert a toLayoutOp in between producerOp
      // and consumerOp.
      //
      if (isa<ToLayoutOp>(producerOp)) {
        ToLayoutOp toLayoutOp = llvm::cast<ToLayoutOp>(producerOp);
        toLayoutOp.setMemoryConfigAttr(outputMemConfigAttr);
        toLayoutOp.getResult().setType(newTensorType);
      } else {
        OpBuilder builder(consumerOp);

        DataTypeAttr outputDataType = DataTypeAttr::get(
            consumerOp->getContext(), consumerOpOutputLayout.getDataType());
        Layout outputLayoutEnum = consumerOpOutputLayout.getLayout();
        LayoutAttr outputLayout =
            LayoutAttr::get(consumerOp->getContext(), outputLayoutEnum);
        Operation *memoryReconfigOp = builder.create<ToLayoutOp>(
            consumerOp->getLoc(), newTensorType, producerOp->getResult(0),
            outputLayout, outputDataType, outputMemConfigAttr,
            getDeviceOpValue(consumerOp));

        consumerOp->setOperand(edge.operandIndex,
                               memoryReconfigOp->getResult(0));
      }
    }
  }
};

} // namespace mlir::tt::ttnn
