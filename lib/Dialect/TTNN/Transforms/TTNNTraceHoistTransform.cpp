// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNTRACEHOISTTRANSFORM
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNTraceHoistTransform
    : public impl::TTNNTraceHoistTransformBase<TTNNTraceHoistTransform> {
public:
  using impl::TTNNTraceHoistTransformBase<
      TTNNTraceHoistTransform>::TTNNTraceHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = this->getOperation();
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (failed(processFuncOp(funcOp))) {
        signalPassFailure();
      }
    });
  }

private:
  bool shouldHoistOp(Operation *op) {
    bool shouldHoist = true;
    shouldHoist &= !::mlir::isa<func::ReturnOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttcore::LoadCachedOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::TraceOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::GetDeviceOp>(op);
    return shouldHoist;
  }

  // Collect all inputs and outputs outside the operation set to hoist
  void collectFunctionBoundary(llvm::ArrayRef<Operation *> opsToHoist,
                               llvm::SmallVector<mlir::Value> &inputs,
                               llvm::SmallVector<mlir::Value> &outputs) {

    // Create set for quick lookup
    llvm::SmallPtrSet<Operation *, 16> opSet(opsToHoist.begin(),
                                             opsToHoist.end());
    llvm::SmallPtrSet<mlir::Value, 16> seenInputs;

    // Collect inputs: operands that come from outside the operation set
    for (Operation *op : opsToHoist) {
      for (auto operand : op->getOperands()) {
        if (!::mlir::isa<RankedTensorType>(operand.getType())) {
          continue;
        }
        Operation *definingOp = operand.getDefiningOp();
        if (!definingOp || !opSet.contains(definingOp)) {
          if (seenInputs.insert(operand).second) {
            inputs.push_back(operand);
          }
        }
      }
    }

    llvm::sort(inputs.begin(), inputs.end(), [](mlir::Value a, mlir::Value b) {
      // prioritize block arguments
      // this is ok now since we check that the funcOp has only 1 block
      // should be updated if we support multiple blocks in the future
      if (::mlir::isa<mlir::BlockArgument>(a) &&
          ::mlir::isa<mlir::BlockArgument>(b)) {
        return ::mlir::cast<mlir::BlockArgument>(a).getArgNumber() <
               ::mlir::cast<mlir::BlockArgument>(b).getArgNumber();
      }
      if (::mlir::isa<mlir::BlockArgument>(a)) {
        return true;
      }
      if (::mlir::isa<mlir::BlockArgument>(b)) {
        return false;
      }

      auto aResult = ::mlir::cast<mlir::OpResult>(a);
      auto bResult = ::mlir::cast<mlir::OpResult>(b);

      if (aResult.getOwner() == bResult.getOwner()) {
        return aResult.getResultNumber() < bResult.getResultNumber();
      }
      return aResult.getOwner()->isBeforeInBlock(bResult.getOwner());
    });

    // Collect outputs: results used outside the operation set
    for (Operation *op : opsToHoist) {
      for (auto result : op->getResults()) {
        for (auto &use : result.getUses()) {
          Operation *user = use.getOwner();
          if (!opSet.contains(user)) {
            outputs.push_back(result);
            break;
          }
        }
      }
    }
  }

  ::mlir::LogicalResult
  createTraceFunctionAndOp(func::FuncOp funcOp,
                           llvm::ArrayRef<Operation *> opsToHoist,
                           size_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Type> inputTypes;
    llvm::SmallVector<mlir::Value> outputs;
    llvm::SmallVector<mlir::Type> outputTypes;

    collectFunctionBoundary(opsToHoist, inputs, outputs);

    for (mlir::Value input : inputs) {
      inputTypes.push_back(input.getType());
    }
    for (mlir::Value output : outputs) {
      outputTypes.push_back(output.getType());
    }

    llvm::SmallString<32> traceFuncName(funcOp.getName());
    traceFuncName.append("_trace_");
    traceFuncName.append(std::to_string(traceFuncIndex));

    auto traceFuncType = builder.getFunctionType(inputTypes, outputTypes);

    // Create the function
    builder.setInsertionPoint(funcOp);
    auto traceFuncOp = builder.create<func::FuncOp>(
        funcOp.getLoc(), traceFuncName, traceFuncType);
    traceFuncOp->setAttr(utils::g_TTNNTraceAttrName, builder.getUnitAttr());

    // Build the body of the new function
    auto *traceFuncEntryBlock = traceFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(traceFuncEntryBlock);

    // maps original input values to trace function input
    // arguments/intermediates
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < inputs.size(); i++) {
      valueMap.insert({inputs[i], traceFuncEntryBlock->getArgument(i)});
    }

    for (Operation *op : opsToHoist) {
      // clone the operation into the trace function
      Operation *clonedOp = builder.clone(*op);

      // Update the op's operands with trace function input arguments
      for (size_t i = 0; i < clonedOp->getNumOperands(); i++) {
        auto originalOperand = op->getOperand(i);
        auto it = valueMap.find(originalOperand);
        if (it != valueMap.end()) {
          clonedOp->setOperand(i, it->second);
          continue;
        }
        // Special case where an op has a device operand.
        // In this case, we need to insert a GetDeviceOp within the trace
        // function. The verifier will ensure that this device matches the
        // device of the trace op.
        if (::mlir::isa<DeviceType>(originalOperand.getType())) {
          auto device = utils::getOrInsertDevice(rewriter, clonedOp);
          clonedOp->setOperand(i, device);
          continue;
        }
        return funcOp.emitError("Could not map operand in hoisted function");
      }

      // Update the op's results with trace function op output result
      for (size_t i = 0; i < op->getNumResults(); i++) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);
      }
    }

    // Finally, we need to add a return operation to the trace function
    llvm::SmallVector<mlir::Value> returnValues;
    for (mlir::Value output : outputs) {
      auto it = valueMap.find(output);
      if (it != valueMap.end()) {
        returnValues.push_back(it->second);
      } else {
        return funcOp.emitError(
            "Could not map output value in hoisted function");
      }
    }
    builder.create<func::ReturnOp>(funcOp.getLoc(), returnValues);

    // Function construction is done, we now need to reference it within
    // a ttnn trace op
    auto calleeAttr =
        mlir::SymbolRefAttr::get(builder.getContext(), traceFuncName);
    Operation *firstOp = opsToHoist.front();
    builder.setInsertionPoint(firstOp);
    auto device = utils::getOrInsertDevice(rewriter, firstOp);
    auto cqIdAttr = builder.getI32IntegerAttr(traceFuncIndex);
    auto blockingAttr = builder.getBoolAttr(false);
    auto traceOp = builder.create<mlir::tt::ttnn::TraceOp>(
        firstOp->getLoc(), outputTypes, device, cqIdAttr, blockingAttr,
        calleeAttr, ValueRange(inputs));

    // Replace uses of original outputs with the output of the trace op function
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].replaceAllUsesWith(traceOp->getResult(i));
    }

    // Remove the original ops in reverse order (to avoid dependency issues)
    for (auto it = opsToHoist.rbegin(); it != opsToHoist.rend(); it++) {
      rewriter.eraseOp(*it);
    }

    return mlir::success();
  }

  mlir::LogicalResult processFuncOp(func::FuncOp funcOp) {
    // skip const-eval functions
    if (ttmlir::utils::isConstEvalFunc(funcOp)) {
      return mlir::success();
    }

    // skip trace functions
    if (utils::isTTNNTraceFunc(funcOp)) {
      return mlir::success();
    }

    if (funcOp.getBlocks().size() != 1) {
      return funcOp.emitError("FuncOp should have exactly one block");
    }

    llvm::SmallVector<Operation *> opsToHoist;

    bool seenHoistableOp = false;
    mlir::Block &block = funcOp.getBlocks().front();
    for (mlir::Operation &op : block.getOperations()) {
      if (shouldHoistOp(&op)) {
        // Hoist all ops starting from this op into a new func
        seenHoistableOp = true;
        opsToHoist.push_back(&op);
        continue;
      }
      // If a non-hoistable op is found after a hoistable op, it must be a
      // return op
      if (seenHoistableOp && !::mlir::isa<func::ReturnOp>(op)) {
        return op.emitError(
            "Non-hoistable op found after seeing a hoistable op");
      }
    }

    // Create trace function and trace op if there are ops to hoist
    if (!opsToHoist.empty()) {
      return createTraceFunctionAndOp(funcOp, opsToHoist, 0);
    }

    return mlir::success();
  }
};
} // namespace mlir::tt::ttnn
