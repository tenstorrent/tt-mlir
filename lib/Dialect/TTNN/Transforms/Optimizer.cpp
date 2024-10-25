// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalGridAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardingAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNOPTIMIZER
#include "ttmlir/Dialect/TTNN/Transforms/Optimizer.h"

class TTNNOptimizer : public impl::TTNNOptimizerBase<TTNNOptimizer> {
public:
  using impl::TTNNOptimizerBase<TTNNOptimizer>::TTNNOptimizerBase;
  void runOnOperation() final {
    // Generate legal OP configuration candidates.
    // Perform sharding analysis.
    // Perform final configuration analysis.
    // Apply graph transformations based on analysis results.
    //
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
    llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> legalLayouts;

    moduleOp->walk([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }

      if (!isa<RankedTensorType>(op->getResult(0).getType())) {
        return;
      }

      RankedTensorType tensorType =
          mlir::cast<RankedTensorType>(op->getResult(0).getType());
      LegalGridAnalysis legalGridAnalysis =
          getChildAnalysis<LegalGridAnalysis>(op);
      legalGridAnalysis.init(LegalGridAnalysisInput(chipDesc, max_grid,
                                                    tensorType, maxLegalLayouts,
                                                    &overrideOutputLayout));
      legalLayouts[op] = legalGridAnalysis.getResult();
    });

    llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> opSchedule;
    std::unordered_set<Edge> reshardedEdges;
    if (shardingPassEnabled) {
      // Extract override resharding edges
      //
      std::unordered_set<Edge> overrideReshardEdges;
      extractReshardEdges(moduleOp, overrideReshardEdges);

      // Perform sharding analysis.
      //
      ShardingAnalysis shardingAnalysis = getAnalysis<ShardingAnalysis>();
      shardingAnalysis.init(ShardingAnalysisInput(
          legalLayouts, chipDesc.getUsableL1Size(), overrideReshardEdges));
      legalLayouts = shardingAnalysis.getResult().legalLayouts;
      opSchedule = shardingAnalysis.getResult().schedule;
      reshardedEdges = shardingAnalysis.getResult().reshardedEdges;
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

        if (!isa<RankedTensorType>(op->getResult(0).getType())) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());
        llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

        // Update the output layout attribute with the new grid size.
        //
        if (opConfigAnalysis.getResult().contains(op)) {
          RankedTensorType newTensorType =
              RankedTensorType::get(tensorShape, tensorType.getElementType(),
                                    opConfigAnalysis.getResult().at(op));

          // Update the memory space and layout of the op.
          //
          tt::LayoutAttr ttLayoutAttr =
              mlir::cast<tt::LayoutAttr>(newTensorType.getEncoding());

          op->getResult(0).setType(newTensorType);

          // Update DPS operand layout as well.
          //
          if (isa<mlir::DestinationStyleOpInterface>(op)) {
            BufferType bufferType =
                utils::toTTNNBufferType(ttLayoutAttr.getMemorySpace());
            TensorMemoryLayout tensorMemoryLayout =
                utils::toTTNNTensorMemoryLayout(ttLayoutAttr.getMemLayout());

            op->getOperands().back().setType(newTensorType);
            EmptyOp emptyOp =
                mlir::cast<EmptyOp>(op->getOperands().back().getDefiningOp());

            emptyOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
                op->getContext(),
                TensorMemoryLayoutAttr::get(op->getContext(),
                                            tensorMemoryLayout),
                BufferTypeAttr::get(op->getContext(), bufferType),
                ShardSpecAttr::get(
                    op->getContext(),
                    ShapeAttr::get(op->getContext(),
                                   ttLayoutAttr.getMemref().getShape()))));
          }
          // TODO (nobradovic): Other memory management ops after lowering to
          // TTNN will need to be special handled as well. Depends on ttnn
          // layout attr refactor and lowering.
          //
          else if (isa<ttnn::CompositeToLayoutOp>(op)) {
            BufferType bufferType =
                utils::toTTNNBufferType(ttLayoutAttr.getMemorySpace());
            TensorMemoryLayout tensorMemoryLayout =
                utils::toTTNNTensorMemoryLayout(ttLayoutAttr.getMemLayout());
            // Update the device op with the new tensor type.
            //
            ttnn::CompositeToLayoutOp compositeLayoutOp =
                llvm::cast<ttnn::CompositeToLayoutOp>(op);
            compositeLayoutOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
                op->getContext(),
                ttnn::TensorMemoryLayoutAttr::get(op->getContext(),
                                                  tensorMemoryLayout),
                ttnn::BufferTypeAttr::get(op->getContext(), bufferType),
                ttnn::ShardSpecAttr::get(
                    op->getContext(),
                    ttnn::ShapeAttr::get(
                        op->getContext(),
                        ttLayoutAttr.getMemref().getShape()))));
          }
        }
      });

      if (reshardingEnabled) {
        processReshardedEdges(reshardedEdges);
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

  void processReshardedEdges(const std::unordered_set<Edge> &reshardedEdges) {
    // Insert reshard ops here based on results of sharding analysis.
    //
    for (const Edge &edge : reshardedEdges) {
      Operation *producerOp = edge.producerOp;
      Operation *consumerOp = edge.consumerOp;

      // If producerOp is a compositeToLayoutOp, adjust its output layout(update
      // inplace) to reflect consumerOp's output layout. If producerOp is not a
      // compositeToLayoutOp, insert a compositeToLayoutOp in between producerOp
      // and consumerOp.
      //
      if (isa<ttnn::CompositeToLayoutOp>(producerOp)) {
        ttnn::CompositeToLayoutOp compositeLayoutOp =
            llvm::cast<ttnn::CompositeToLayoutOp>(producerOp);
        tt::LayoutAttr consumerOpOutputLayout = mlir::cast<tt::LayoutAttr>(
            mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType())
                .getEncoding());

        RankedTensorType compositeLayoutOpTensorType =
            mlir::cast<RankedTensorType>(
                compositeLayoutOp.getResult().getType());
        llvm::ArrayRef<int64_t> compositeLayoutOpTensorShape =
            compositeLayoutOpTensorType.getShape();
        tt::LayoutAttr compositeLayoutOpLayout = mlir::cast<tt::LayoutAttr>(
            compositeLayoutOpTensorType.getEncoding());

        // TODO(nobradovic): Match memory space and layout of consumer op. This
        // actually needs to be properly resolved based on op type, output
        // layout and other inputs.
        //
        RankedTensorType newTensorType = RankedTensorType::get(
            compositeLayoutOpTensorShape,
            compositeLayoutOpTensorType.getElementType(),
            compositeLayoutOpLayout
                .withElementType(compositeLayoutOp->getContext(),
                                 consumerOpOutputLayout.getElementType())
                .withMemorySpace(compositeLayoutOp.getContext(),
                                 consumerOpOutputLayout.getMemorySpace())
                .withMemoryLayout(compositeLayoutOp.getContext(),
                                  consumerOpOutputLayout.getMemLayout())
                .withGrid(compositeLayoutOp.getContext(),
                          compositeLayoutOpTensorType,
                          consumerOpOutputLayout.getGrid()));

        compositeLayoutOp.getResult().setType(newTensorType);
        compositeLayoutOp.getOperands().back().setType(newTensorType);
      }
      // TODO (nobradovic): Resharding needs to be reimplemented for TTNN
      // dialect.
      //   else {
      //     tt::LayoutAttr consumerOpOutputLayout = mlir::cast<tt::LayoutAttr>(
      //         mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType())
      //             .getEncoding());

      //     RankedTensorType producerOpTensorType =
      //         mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
      //     llvm::ArrayRef<int64_t> producerOpTensorShape =
      //         producerOpTensorType.getShape();
      //     tt::LayoutAttr producerOpLayout =
      //         mlir::cast<tt::LayoutAttr>(producerOpTensorType.getEncoding());

      //     // TODO(nobradovic): Match memory space and layout of consumer op.
      //     This
      //     // actually needs to be properly resolved based on op type, output
      //     // layout and other inputs.
      //     //
      //     RankedTensorType newTensorType = RankedTensorType::get(
      //         producerOpTensorShape, producerOpTensorType.getElementType(),
      //         producerOpLayout
      //             .withElementType(consumerOp->getContext(),
      //                              consumerOpOutputLayout.getElementType())
      //             .withMemorySpace(consumerOp->getContext(),
      //                              consumerOpOutputLayout.getMemorySpace())
      //             .withMemoryLayout(consumerOp->getContext(),
      //                               consumerOpOutputLayout.getMemLayout())
      //             .withGrid(consumerOp->getContext(), producerOpTensorType,
      //                       consumerOpOutputLayout.getGrid()));

      // OpBuilder builder(consumerOp);

      // mlir::tensor::EmptyOp emptyOp = builder.create<tensor::EmptyOp>(
      //     consumerOp->getLoc(), producerOpTensorShape,
      //     producerOpTensorType.getElementType(),
      //     mlir::cast<LayoutAttr>(newTensorType.getEncoding()));

      // Operation *toLayoutOp = builder.create<ttir::ToLayoutOp>(
      //     consumerOp->getLoc(), newTensorType, producerOp->getResult(0),
      //     emptyOp);

      // consumerOp->setOperand(edge.operandIndex, toLayoutOp->getResult(0));
      //   }
    }
  }
};

} // namespace mlir::tt::ttnn
