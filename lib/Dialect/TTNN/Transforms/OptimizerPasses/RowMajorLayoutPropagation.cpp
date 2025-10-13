// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LLVM.h>
#include <queue>
#include <vector>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNROWMAJORLAYOUTPROPAGATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNRowMajorLayoutPropagation
    : public impl::TTNNRowMajorLayoutPropagationBase<
          TTNNRowMajorLayoutPropagation> {
public:
  using impl::TTNNRowMajorLayoutPropagationBase<
      TTNNRowMajorLayoutPropagation>::TTNNRowMajorLayoutPropagationBase;

  TTNNRowMajorLayoutPropagation(TTNNRowMajorLayoutPropagationOptions options)
      : impl::TTNNRowMajorLayoutPropagationBase<TTNNRowMajorLayoutPropagation>(
            std::move(options)) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp->walk([&](func::FuncOp func) {
      if (ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Running RM layout propagation on function: {}",
                   func.getName());

      llvm::DenseSet<Value> funcInputArguments;
      identifyInputArguments(func, funcInputArguments);

      // convert input arguments to RM
      llvm::SmallVector<Value> convertedInputArgs = convertInputArgsLayouts(funcInputArguments);

      llvm::DenseMap<Operation *, Layout> opLayoutConstraints;
      propagateRowMajorLayout(func, convertedInputArgs, opLayoutConstraints);
    });
  }

private:
  bool isInputArgument(BlockArgument arg, func::FuncOp func) {
    if (auto typeAttr = func.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
            arg.getArgNumber(), ttcore::ArgumentTypeAttr::name)) {
      auto argType = typeAttr.getValue();
      return argType == ttcore::ArgumentType::Default ||
             argType == ttcore::ArgumentType::Input;
    }
    return true;
  }

  void identifyInputArguments(func::FuncOp func,
                              llvm::DenseSet<Value> &rowMajorCandidates) {
    for (BlockArgument arg : func.getArguments()) {
      if (!isInputArgument(arg, func)) {
        continue;
      }

      rowMajorCandidates.insert(arg);
    }
  }

   llvm::SmallVector<Value> convertInputArgsLayouts(const llvm::DenseSet<Value> &inputArgs) {
    llvm::SmallVector<Value> convertedArgs;
    for (Value arg : inputArgs) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType());
      TTNNLayoutAttr oldLayout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
      assert(oldLayout && "Expected TTNNLayoutAttr on function argument");
      OpBuilder builder(arg.getContext());
      builder.setInsertionPoint(arg.getUses().begin()->getOwner());

      llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
      for (auto& use: arg.getUses()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                     "Converting input arg {} used in op {}",
                     arg, use.getOwner()->getName());
        uses.push_back({use.getOwner(), use.getOperandNumber()});
      }

      RankedTensorType newTensorType = RankedTensorType::get(
          tensorType.getShape(), tensorType.getElementType(),
          oldLayout.withLayout(Layout::RowMajor, tensorType.getShape()));
      Operation *toLayoutOp = builder.create<ToLayoutOp>(
          builder.getUnknownLoc(), newTensorType, arg,
          LayoutAttr::get(arg.getContext(), ::mlir::tt::ttnn::Layout::RowMajor),
          ttcore::DataTypeAttr::get(arg.getContext(), oldLayout.getDataType()),
          MemoryConfigAttr::get(
              builder.getContext(),
              TensorMemoryLayoutAttr::get(builder.getContext(),
                                          TensorMemoryLayout::Interleaved),
              BufferTypeAttr::get(builder.getContext(), BufferType::DRAM),
              /*shardSpec=*/std::nullopt));
      convertedArgs.push_back(toLayoutOp->getResult(0));

      for (auto& use: uses) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                      "Updating use in op {} to use converted arg",
                      use.first->getName());
        use.first->setOperand(use.second, toLayoutOp->getResult(0));
      }
    }
    return convertedArgs;
  }

  void
  propagateRowMajorLayout(func::FuncOp func,
                          const llvm::SmallVector<Value> &funcInputArgsConverted,
                          llvm::DenseMap<Operation *, Layout> &constraints) {

    std::queue<Value> worklist = {};

    for (Value arg : funcInputArgsConverted) {
      worklist.push(arg);
    }

    while (!worklist.empty()) {
      Value current = worklist.front();
      worklist.pop();

      for (auto &use : current.getUses()) {
        Operation *user = use.getOwner();

        llvm::Expected<TTNNLayoutAttr> rmOutputLayout =
            opStopsRowMajorPropagation(user, use.getOperandNumber());

        if (!rmOutputLayout) {
          llvm::consumeError(rmOutputLayout.takeError());
          continue;
        }

        RankedTensorType userResultType =
            mlir::cast<RankedTensorType>(user->getResult(0).getType());

        RankedTensorType newResultType = RankedTensorType::get(
            userResultType.getShape(), userResultType.getElementType(),
            rmOutputLayout.get());

        user->getResult(0).setType(newResultType);
        TTMLIR_DEBUG(
            ttmlir::LogComponent::RMPropagation,
            "Set RowMajor layout on op {} at {}, \n\t output layout: {}",
            user->getName(), user->getLoc(), rmOutputLayout.get());

        worklist.push(user->getResult(0));
      }
    }
  }

  llvm::Expected<TTNNLayoutAttr>
  opStopsRowMajorPropagation(Operation *op, unsigned operandIdx) {
    OpOperand &operand = op->getOpOperand(operandIdx);
    RankedTensorType tensorType =
        mlir::dyn_cast<RankedTensorType>(operand.get().getType());
    assert(tensorType && "Expected ranked tensor type");
    TTNNLayoutAttr layoutAttr =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    assert(layoutAttr && "Expected layout attribute");

    TTNNLayoutAttr rmLayout =
        layoutAttr.withLayout(Layout::RowMajor, tensorType.getShape());

    // Extract input layouts from the operation
    std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);
    inputLayouts[operandIdx] = rmLayout;

    OpConfig config = extractOpConfigFromIR(op);
    // Nullify output layout to let backend choose freely
    config.outputLayout = nullptr;

    llvm::Expected<op_constraint_validation::ValidationResult> result =
        op_constraint_validation::validateOperation(op, inputLayouts, config,
                                                    tensorL1UsageCap);
    if (!result) {
      llvm::consumeError(result.takeError());
      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Stopping RM propagation at op {} as it fails validation "
                   "with RM layout on operand {},\n\t input layout: {}",
                   op->getName(), operandIdx, rmLayout);
      return llvm::make_error<llvm::StringError>(
          "Stopping RM propagation", llvm::inconvertibleErrorCode());
    }

    if (result->actualOutputLayout.isTiled()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Stopping RM propagation at op {} as it returns tile "
                   "layout,\n\t output layout: {}",
                   op->getName(), result->actualOutputLayout);
      return llvm::make_error<llvm::StringError>(
          "Stopping RM propagation", llvm::inconvertibleErrorCode());
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                 "Continuing RM propagation through op {} with RM layout on "
                 "operand {},\n\t input layout: {}, \n\t output layout: {}",
                 op->getName(), operandIdx, rmLayout,
                 result->actualOutputLayout);

    return result->actualOutputLayout;
  }

  // Extract OpConfig from operation's IR
  OpConfig extractOpConfigFromIR(Operation *operation) {
    assert(operation->getNumResults() > 0 &&
           "Operation must have at least one result to extract OpConfig");

    OpConfig config;
    assert(operation->getNumResults() == 1 &&
           "Expected operation with one result");

    // Extract output layout from result type
    if (auto tensorType =
            mlir::dyn_cast<RankedTensorType>(operation->getResultTypes()[0])) {
      if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              tensorType.getEncoding())) {
        config.outputLayout = layoutAttr;
      }
    }

    // TODO(rpavlovicTT): handle ConvTranspose2d too.
    // For Conv2d operations, extract op-specific attributes
    if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(operation)) {
      config.opSpecificAttrs = Conv2dAttrs{conv2dOp.getConv2dConfigAttr(),
                                           conv2dOp.getComputeConfigAttr()};
    }

    return config;
  }
};

} // namespace mlir::tt::ttnn
