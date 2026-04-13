// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/D2MOptimizerUtils.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::d2m_optimizer_utils {

void applyChosenLayoutToD2MSubgraphOp(D2MSubgraphOp dispatchOp,
                                      RankedTensorType newTensorType,
                                      TTNNLayoutAttr layoutAttr,
                                      ttcore::GridAttr deviceGrid) {
  assert(dispatchOp.getNumResults() <= 1 &&
         "D2MSubgraphOp with multiple results not yet supported");

  for (unsigned i = 0; i < dispatchOp.getNumResults(); ++i) {
    dispatchOp.getResult(i).setType(newTensorType);
  }

  for (Value output : dispatchOp.getOutputs()) {
    if (EmptyOp emptyOp = output.getDefiningOp<EmptyOp>()) {
      emptyOp.getResult().setType(newTensorType);
      BufferType bufferType = layoutAttr.getBufferType();
      TensorMemoryLayoutAttr tensorMemoryLayoutAttr = layoutAttr.getMemLayout();
      emptyOp.setDtype(layoutAttr.getDataType());
      if (layoutAttr.isTiled()) {
        emptyOp.setLayout(ttnn::Layout::Tile);
      } else {
        emptyOp.setLayout(ttnn::Layout::RowMajor);
      }
      emptyOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
          dispatchOp.getContext(), tensorMemoryLayoutAttr,
          BufferTypeAttr::get(dispatchOp.getContext(), bufferType),
          utils::createShardSpecIfNeeded(layoutAttr, deviceGrid)));
    } else {
      llvm::report_fatal_error(
          "Expected EmptyOp for D2MSubgraphOp output buffer");
    }
  }

  if (func::FuncOp mainFunc = dispatchOp.getD2MMainFunc()) {
    Block &entryBlock = mainFunc.getBody().front();
    unsigned argIdx = 0;
    for (Value input : dispatchOp.getInputs()) {
      if (argIdx < entryBlock.getNumArguments()) {
        Type inputType = input.getType();
        if (isa<RankedTensorType>(inputType)) {
          entryBlock.getArgument(argIdx).setType(inputType);
        }
        ++argIdx;
      }
    }
    // Update the return value's type directly rather than inserting a
    // to_layout op. The D2M compilation pipeline will re-derive all internal
    // layouts; inserting to_layout would block the TTNN→TTIR conversion step
    // since there is no conversion pattern for it.
    Block &block = mainFunc.getBody().front();
    Operation *terminator = block.getTerminator();
    if (func::ReturnOp returnOp = dyn_cast<func::ReturnOp>(terminator)) {
      if (returnOp.getNumOperands() > 0) {
        Value currentResultValue = returnOp.getOperand(0);
        if (currentResultValue.getType() != newTensorType) {
          currentResultValue.setType(newTensorType);
        }
      }
    }
    SmallVector<Type> newInputTypes;
    for (Value input : dispatchOp.getInputs()) {
      newInputTypes.push_back(input.getType());
    }
    SmallVector<Type> newResultTypes(dispatchOp.getNumResults(), newTensorType);
    mainFunc.setType(FunctionType::get(dispatchOp.getContext(), newInputTypes,
                                       newResultTypes));
  }
}

void syncD2MFuncTypesToDispatchInputs(D2MSubgraphOp dispatchOp) {
  func::FuncOp mainFunc = dispatchOp.getD2MMainFunc();
  if (!mainFunc) {
    return;
  }
  Block &entryBlock = mainFunc.getBody().front();
  unsigned argIdx = 0;
  for (Value input : dispatchOp.getInputs()) {
    if (argIdx < entryBlock.getNumArguments()) {
      Type inputType = input.getType();
      if (isa<RankedTensorType>(inputType)) {
        entryBlock.getArgument(argIdx).setType(inputType);
      }
      ++argIdx;
    }
  }
  SmallVector<Type> newInputTypes;
  for (Value input : dispatchOp.getInputs()) {
    newInputTypes.push_back(input.getType());
  }
  SmallVector<Type> newResultTypes;
  for (Value result : dispatchOp.getResults()) {
    newResultTypes.push_back(result.getType());
  }
  mainFunc.setType(FunctionType::get(dispatchOp.getContext(), newInputTypes,
                                     newResultTypes));
}

void syncAllD2MFuncTypes(func::FuncOp func) {
  func->walk([&](D2MSubgraphOp dispatchOp) {
    syncD2MFuncTypesToDispatchInputs(dispatchOp);
  });
}

} // namespace mlir::tt::ttnn::d2m_optimizer_utils
