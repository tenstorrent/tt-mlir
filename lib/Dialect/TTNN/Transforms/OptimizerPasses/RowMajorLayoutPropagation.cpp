// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "SingletonDeviceContext.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <queue>
#include <utility>
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
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNRowMajorLayoutPropagation pass requires OpModel support to be"
        "enabled.");
#else
    op_model::ScopedSingletonDeviceGuard deviceGuard;

    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](func::FuncOp func) {
      if (ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Running RM layout propagation on function: {}",
                   func.getName());

      llvm::SmallVector<Value> funcInputArguments =
          identifyInputArguments(func);

      llvm::SmallVector<ttnn::ToLayoutOp> toRemoveToLayoutOps =
          findRedundantToLayoutOps(funcInputArguments);

      llvm::SmallVector<Value> rowMajorArgs =
          bypassRedundantToLayoutOps(toRemoveToLayoutOps);

      llvm::DenseMap<Operation *, Layout> opLayoutConstraints;
      propagateRowMajorLayout(func, rowMajorArgs, opLayoutConstraints);
    });
#endif // TTMLIR_ENABLE_OPMODEL
  }

private:
  // Returns true if block argument is an input argument of the function.
  bool isInputArgument(BlockArgument arg, func::FuncOp func) {
    if (auto typeAttr = func.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
            arg.getArgNumber(), ttcore::ArgumentTypeAttr::name)) {
      auto argType = typeAttr.getValue();
      return argType == ttcore::ArgumentType::Input;
    }
    return false;
  }

  // Identifies function input arguments that are candidates for RowMajor layout
  // propagation.
  llvm::SmallVector<Value> identifyInputArguments(func::FuncOp func) {
    llvm::SmallVector<Value> rowMajorCandidates;
    for (BlockArgument arg : func.getArguments()) {
      if (isInputArgument(arg, func)) {
        rowMajorCandidates.push_back(arg);
      }
    }
    return rowMajorCandidates;
  }

  // Finds ToLayoutOps that convert RowMajor layout to Tiled layout on function
  // inputs. Such ToLayoutOps are redundant and can be removed.  Returns the
  // list of ToLayoutOps to be removed.
  llvm::SmallVector<ttnn::ToLayoutOp>
  findRedundantToLayoutOps(const llvm::SmallVector<Value> &inputArgs) {
    llvm::SmallVector<ttnn::ToLayoutOp> opsToRemove;
    for (Value arg : inputArgs) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType());
      TTNNLayoutAttr argTTNNLayout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
      assert(argTTNNLayout && "Expected TTNNLayoutAttr on function argument");

      if (argTTNNLayout.isTiled()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                     "Input argument {} is not RowMajor, skipping it", arg);
        continue;
      }

      for (const auto &user : arg.getUsers()) {
        if (!mlir::isa<ttnn::ToLayoutOp>(user)) {
          continue;
        }
        ttnn::ToLayoutOp toLayoutOp = mlir::dyn_cast<ttnn::ToLayoutOp>(user);

        TTNNLayoutAttr targetLayout = mlir::cast<ttnn::TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(toLayoutOp->getResult(0).getType())
                .getEncoding());
        assert(targetLayout && "Expected TTNNLayoutAttr on ToLayoutOp result");
        if (!targetLayout.isTiled()) {
          // This toLayout keeps RM layout, so we can skip it.
          continue;
        }

        // Check if the only difference is page layout.
        TTNNLayoutAttr modifiedArgTTNNLayout =
            argTTNNLayout.withLayout(Layout::Tile, tensorType.getShape());
        if (modifiedArgTTNNLayout == targetLayout) {
          opsToRemove.push_back(toLayoutOp);
          TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                       "Arg layout {} differs from target layout {} only in "
                       "page layout, we can remove ToLayoutOp {}",
                       argTTNNLayout, targetLayout, toLayoutOp);
        }
      }
    }
    return opsToRemove;
  }

  // Bypasses given ToLayoutOps by rewiring their inputs to their users. Returns
  // the list of argument Values that were inputs to the removed ToLayoutOps.
  llvm::SmallVector<Value> bypassRedundantToLayoutOps(
      llvm::SmallVector<ttnn::ToLayoutOp> &toRemoveToLayoutOps) {
    SmallVector<Value> rmArgs;
    for (ttnn::ToLayoutOp &toLayoutOp : toRemoveToLayoutOps) {
      Value arg = toLayoutOp.getInput();

      llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
      for (auto &use : toLayoutOp->getResult(0).getUses()) {
        uses.emplace_back(use.getOwner(), use.getOperandNumber());
      }

      for (auto &[user, operandIdx] : uses) {
        user->setOperand(operandIdx, arg);
      }

      rmArgs.push_back(arg);

      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Bypassing and erasing ToLayoutOp {}", toLayoutOp);
      toLayoutOp.erase();
    }
    return rmArgs;
  }

  // Propagates RowMajor layout through the function starting from the given
  // argument Values. Updates the operation result types in-place. Stops when
  // reaching operations that return Tiled layouts.
  void
  propagateRowMajorLayout(func::FuncOp func,
                          const llvm::SmallVector<Value> &rowMajorArgs,
                          llvm::DenseMap<Operation *, Layout> &constraints) {

    std::queue<Value> worklist = {};

    for (Value arg : rowMajorArgs) {
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

  // Checks if given operation is valid and returns RowMajor layout. If
  // operation is not valid with RowMajor layout on given operand, returns an
  // error. Returns the RowMajor output layout if operation is valid.
  llvm::Expected<TTNNLayoutAttr>
  opStopsRowMajorPropagation(Operation *op, unsigned operandIdx) {
    if (auto toLayoutOp = mlir::dyn_cast<ttnn::ToLayoutOp>(op)) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Stopping RM propagation at ToLayoutOp {}", toLayoutOp);
      return llvm::make_error<llvm::StringError>(
          "Stopping RM propagation at ToLayoutOp",
          llvm::inconvertibleErrorCode());
    }

    OpOperand &operand = op->getOpOperand(operandIdx);
    RankedTensorType tensorType =
        mlir::dyn_cast<RankedTensorType>(operand.get().getType());
    assert(tensorType && "Expected ranked tensor type");
    TTNNLayoutAttr layoutAttr =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    assert(layoutAttr && "Expected layout attribute");

    // Extract input layouts from the operation
    std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);
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
                   op->getName(), operandIdx, layoutAttr);
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
                 op->getName(), operandIdx, layoutAttr,
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

    // For Conv2d operations, extract op-specific attributes
    if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(operation)) {
      config.opSpecificAttrs = Conv2dAttrs{conv2dOp.getConv2dConfigAttr(),
                                           conv2dOp.getComputeConfigAttr()};
    } else if (auto convTranspose2dOp =
                   mlir::dyn_cast<ttnn::ConvTranspose2dOp>(operation)) {
      config.opSpecificAttrs =
          Conv2dAttrs{convTranspose2dOp.getConv2dConfigAttr(),
                      /*deviceComputeConfigAttr*/ std::nullopt};
    }

    return config;
  }
};

} // namespace mlir::tt::ttnn
