// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"

#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::wa {

// Operand workarounds factory method
TTNNOperandWorkarounds
TTNNOperandWorkarounds::createEmptyTTNNOperandWorkarounds() {
  return TTNNOperandWorkarounds();
}

// Operands workarounds factory method
TTNNOperandsWorkarounds
TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(int inputSize,
                                                            int outputSize) {
  llvm::SmallVector<TTNNOperandWorkarounds> inputOperandWorkarounds(
      inputSize, TTNNOperandWorkarounds::createEmptyTTNNOperandWorkarounds());
  llvm::SmallVector<TTNNOperandWorkarounds> outputOperandWorkarounds(
      outputSize, TTNNOperandWorkarounds::createEmptyTTNNOperandWorkarounds());
  return TTNNOperandsWorkarounds(inputOperandWorkarounds,
                                 outputOperandWorkarounds);
}

// Method to apply tensor workarounds. If the workaround is present, it
// applies the workaround, and returns both the target workaround argument and
// a flag indicating whether the workaround was applied.
WorkaroundResult applyWorkarounds(const TTNNOperandWorkarounds &workaround,
                                  const TTNNLayoutAttr &inputLayoutAttr) {
  WorkaroundResult result;
  result.targetTensorLayoutResult.first =
      workaround.tensorLayoutWorkaround.value_or(inputLayoutAttr.getLayout());
  result.targetTensorLayoutResult.second =
      result.targetTensorLayoutResult.first != inputLayoutAttr.getLayout();

  result.targetTensorBufferTypeResult.first =
      workaround.tensorBufferTypeWorkaround.value_or(
          inputLayoutAttr.getBufferType());
  result.targetTensorBufferTypeResult.second =
      result.targetTensorBufferTypeResult.first !=
      inputLayoutAttr.getBufferType();

  result.targetTensorMemoryLayoutResult.first =
      workaround.tensorMemoryLayoutWorkaround.value_or(
          inputLayoutAttr.getMemLayout());
  result.targetTensorMemoryLayoutResult.second =
      result.targetTensorMemoryLayoutResult.first !=
      inputLayoutAttr.getMemLayout();

  return result;
}

// Operands workarounds factory method.
TTNNOperandsWorkarounds
TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(Operation *op) {
  size_t tensorInputs =
      llvm::count_if(op->getOperands(), ttmlir::utils::isRankedTensor);
  size_t tensorResults =
      llvm::count_if(op->getResults(), ttmlir::utils::isRankedTensor);

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(
      tensorInputs, tensorResults);
}
} // namespace mlir::tt::ttnn::wa
