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
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/FunctionTypes.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"

#include <cassert>
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
      // Apply analysis only on forward functions.
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
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

  // Returns true if the tensor has an integer element type.
  bool hasIntegerElementType(Value value) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return false;
    }
    return tensorType.getElementType().isIntOrIndex();
  }

  // Identifies function input arguments that are candidates for RowMajor layout
  // propagation. Currently restricted to integer tensor types only.
  llvm::SmallVector<Value> identifyInputArguments(func::FuncOp func) {
    llvm::SmallVector<Value> rowMajorCandidates;
    for (BlockArgument arg : func.getArguments()) {
      if (isInputArgument(arg, func) && hasIntegerElementType(arg)) {
        rowMajorCandidates.push_back(arg);
      }
    }
    return rowMajorCandidates;
  }

  // Finds ToLayoutOps that convert RowMajor layout to Tiled layout on function
  // inputs. Such ToLayoutOps are redundant and can be removed. Returns the
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

  // Handles dtype conversion when backend dtype differs from IR tensor element
  // type. Inserts a toLayout op with TILE layout (required for on-device
  // typecast) to perform the dtype conversion. Returns true if conversion was
  // inserted, indicating RM propagation should stop at this point.
  //
  // TODO(bmalesevic, #6783): Once issue is addressed (on-device typecasting
  // support for RM tensors), RM propagation should be allowed to continue after
  // dtype conversion without forcing conversion to tile layout (which currently
  // stops RM propagation).
  bool handleDtypeConversionIfNeeded(Operation *user, IRRewriter &rewriter,
                                     TTNNLayoutAttr rmOutputLayout,
                                     Type backendDataType,
                                     Type tensorElementType,
                                     RankedTensorType userResultType) {
    if (backendDataType == tensorElementType) {
      return false; // No conversion needed
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                 "Dtype mismatch detected at op {}: backend dtype {} != "
                 "tensor element type {}. Inserting toLayout op.",
                 ttmlir::opToString(user), backendDataType, tensorElementType);

    // First, set the operation's result type to match what the backend
    // actually produces (with backend's dtype in the layout)
    RankedTensorType backendResultType = RankedTensorType::get(
        userResultType.getShape(), backendDataType, rmOutputLayout);
    user->getResult(0).setType(backendResultType);

    rewriter.setInsertionPointAfter(user);

    // Create layout with TILE (required for on-device typecast) and the
    // original tensor element type. TTNNDecomposeLayouts requires TILE
    // layout for on-device dtype conversion; ROW_MAJOR would force a
    // host round-trip (from_device → typecast → to_device).
    TTNNLayoutAttr correctedLayout = rmOutputLayout.withElementType(
        tensorElementType, userResultType.getShape());
    TTNNLayoutAttr tileLayout =
        correctedLayout.withLayout(Layout::Tile, userResultType.getShape());

    // Create toLayout op using TILE layout for dtype conversion
    auto toLayoutOp = utils::createToLayoutOp(
        user,
        mlir::cast<mlir::TypedValue<RankedTensorType>>(user->getResult(0)),
        rewriter, tileLayout.getLayout(), tileLayout.getBufferType(),
        tileLayout.getMemLayout(), tileLayout.getDataType(),
        "_dtype_conversion");

    // Replace all uses (except the toLayout itself)
    user->getResult(0).replaceAllUsesExcept(toLayoutOp.getResult(), toLayoutOp);

    TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                 "Inserted toLayout op (TILE) after {} to convert {} -> "
                 "{}. Stopping RM propagation.",
                 ttmlir::opToString(user), backendDataType, tensorElementType);

    return true; // Conversion inserted, stop RM propagation
  }

  // Propagates RowMajor layout through the function starting from the given
  // argument values. Updates the operation result types in-place. Stops when
  // reaching operations that return Tiled layouts. Inserts ToLayoutOps before
  // ReturnOp when actual layout differs from expected function signature.
  void
  propagateRowMajorLayout(func::FuncOp func,
                          const llvm::SmallVector<Value> &rowMajorArgs,
                          llvm::DenseMap<Operation *, Layout> &constraints) {
    IRRewriter rewriter(func.getContext());
    FunctionType funcType = func.getFunctionType();

    std::queue<Value> worklist = {};

    for (Value arg : rowMajorArgs) {
      worklist.push(arg);
    }

    while (!worklist.empty()) {
      Value current = worklist.front();
      worklist.pop();

      for (auto &use : current.getUses()) {
        Operation *user = use.getOwner();

        if (auto returnOp = mlir::dyn_cast<func::ReturnOp>(user)) {
          handleReturnOp(rewriter, funcType, returnOp, use.getOperandNumber(),
                         current);
          continue;
        }

        llvm::Expected<TTNNLayoutAttr> rmOutputLayout =
            opStopsRowMajorPropagation(user, use.getOperandNumber());

        if (!rmOutputLayout) {
          llvm::consumeError(rmOutputLayout.takeError());
          continue;
        }

        RankedTensorType userResultType =
            mlir::cast<RankedTensorType>(user->getResult(0).getType());

        Type backendDataType = rmOutputLayout->getScalarElementType();
        Type tensorElementType = userResultType.getElementType();

        // Handle dtype conversion if backend dtype differs from IR element type
        if (handleDtypeConversionIfNeeded(user, rewriter, rmOutputLayout.get(),
                                          backendDataType, tensorElementType,
                                          userResultType)) {
          continue; // Stop RM propagation (converted to TILE)
        }

        // No dtype mismatch, continue propagating with RM layout
        RankedTensorType backendResultType = RankedTensorType::get(
            userResultType.getShape(), backendDataType, rmOutputLayout.get());
        user->getResult(0).setType(backendResultType);

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

    // Stop propagation at in-place operations (e.g., paged_update_cache).
    // In-place ops have no results and modify tensors in place, so there's
    // no output to propagate layout to.
    if (op->getNumResults() == 0) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Stopping RM propagation at in-place op {} with no results",
                   ttmlir::opToString(op));
      return llvm::make_error<llvm::StringError>(
          "Stopping RM propagation at in-place op",
          llvm::inconvertibleErrorCode());
    }

    OpOperand &operand = op->getOpOperand(operandIdx);
    RankedTensorType tensorType =
        mlir::dyn_cast<RankedTensorType>(operand.get().getType());
    assert(tensorType && "Expected ranked tensor type");
    TTNNLayoutAttr layoutAttr =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    assert(layoutAttr && "Expected layout attribute");

    // Extract input layouts from the operation.
    std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);
    OpConfig config = extractOpConfigFromIR(op);

    // Nullify output layout to let backend choose freely.
    config.outputLayout = nullptr;

    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(op, inputLayouts, config);
    if (result.isError()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Stopping RM propagation at op {} as it fails validation "
                   "with RM layout on operand {},\n\t input layout: {}",
                   ttmlir::opToString(op), operandIdx, layoutAttr);
      return llvm::make_error<llvm::StringError>(
          "Stopping RM propagation", llvm::inconvertibleErrorCode());
    }

    if (result.actualOutputLayout.isTiled()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                   "Stopping RM propagation at op {} as it returns tile "
                   "layout,\n\t output layout: {}",
                   ttmlir::opToString(op), result.actualOutputLayout);
      return llvm::make_error<llvm::StringError>(
          "Stopping RM propagation", llvm::inconvertibleErrorCode());
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                 "Continuing RM propagation through op {} with RM layout on "
                 "operand {},\n\t input layout: {}, \n\t output layout: {}",
                 ttmlir::opToString(op), operandIdx, layoutAttr,
                 result.actualOutputLayout);

    return result.actualOutputLayout;
  }

  // Handles ReturnOp during propagation. Checks if actual layout matches
  // expected function signature and inserts ToLayoutOp if needed for tilizing.
  void handleReturnOp(IRRewriter &rewriter, FunctionType funcType,
                      func::ReturnOp returnOp, unsigned operandIdx,
                      Value previousOp) {
    auto actualTensorType =
        mlir::dyn_cast<RankedTensorType>(previousOp.getType());
    assert(actualTensorType && "Expected ranked tensor type");

    auto actualLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(actualTensorType.getEncoding());
    assert(actualLayout && "Expected layout attribute");
    assert(!actualLayout.isTiled() &&
           "Expected row major as propagation result");

    // Extract expected layout from function signature and check if conversion
    // needed
    Type expectedReturnType = funcType.getResult(operandIdx);
    auto expectedTensorType =
        mlir::dyn_cast<RankedTensorType>(expectedReturnType);
    if (!expectedTensorType) {
      return;
    }
    auto expectedLayout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
        expectedTensorType.getEncoding());
    if (!expectedLayout || !expectedLayout.isTiled()) {
      return;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::RMPropagation,
                 "Inserting ToLayoutOp before return for operand {}: "
                 "actual layout {} -> expected layout {}",
                 operandIdx, actualLayout, expectedLayout);

    rewriter.setInsertionPoint(returnOp);

    auto toLayoutOp = utils::createToLayoutOp(
        returnOp, mlir::cast<mlir::TypedValue<RankedTensorType>>(previousOp),
        rewriter, expectedLayout.getLayout(), expectedLayout.getBufferType(),
        expectedLayout.getMemLayout(), expectedLayout.getDataType(),
        "_return_conversion");

    returnOp.setOperand(operandIdx, toLayoutOp.getResult());
  }

  // Extract OpConfig from operation's IR
  OpConfig extractOpConfigFromIR(Operation *operation) {
    // Precondition: operation must have at least one result.
    // In-place operations (with no results) are filtered out earlier.
    assert(operation->getNumResults() > 0 &&
           "extractOpConfigFromIR expects operation with results");

    OpConfig config;

    // Extract output layout from result type
    if (auto tensorType =
            mlir::dyn_cast<RankedTensorType>(operation->getResultTypes()[0])) {
      if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              tensorType.getEncoding())) {
        config.outputLayout = layoutAttr;
      }
    }

    // For Conv2d operations, extract op-specific attributes
    llvm::TypeSwitch<Operation *, void>(operation)
        .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&config](auto convOp) {
          config.opSpecificAttrs = Conv2dAttrs{convOp.getConv2dConfigAttr(),
                                               convOp.getComputeConfigAttr()};
        });

    return config;
  }
};

} // namespace mlir::tt::ttnn
