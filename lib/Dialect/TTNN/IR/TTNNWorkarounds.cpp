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

  // If the tensor memory layout workaround is present, apply it.
  // Otherwise, return the input tensor memory layout, which may be
  // nullopt if tensor is on host.
  result.targetTensorMemoryLayoutResult.first =
      workaround.tensorMemoryLayoutWorkaround.has_value()
          ? workaround.tensorMemoryLayoutWorkaround
          : inputLayoutAttr.getMemLayoutOpt();
  result.targetTensorMemoryLayoutResult.second =
      result.targetTensorMemoryLayoutResult.first !=
      inputLayoutAttr.getMemLayoutOpt();

  result.targetTensorDataTypeResult.first =
      workaround.tensorDataTypeWorkaround.value_or(
          inputLayoutAttr.getDataType());
  result.targetTensorDataTypeResult.second =
      result.targetTensorDataTypeResult.first != inputLayoutAttr.getDataType();

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

// Factory method to create a set of workarounds for embedding operation
// operands. The embedding operation expects the input to be in row-major layout
// and the weight operand to use the bf16 data type. Since the output of the
// embedding operation follows the same format as the weight operand, the same
// workaround is applied to the output operand. Metal issue for input operand
// workaround: https://github.com/tenstorrent/tt-metal/issues/14915 Metal issue
// for weight operand workaround: to be added
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createEmbeddingOpOperandsWorkarounds() {
  // Create input and weight workarounds.
  TTNNOperandWorkarounds inputWorkaround =
      TTNNOperandWorkarounds(Layout::RowMajor);
  TTNNOperandWorkarounds weightWorkaround =
      TTNNOperandWorkarounds(DataType::BFloat16);
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(0, 0)
      .addInputOperandWorkaround(inputWorkaround)
      .addInputOperandWorkaround(weightWorkaround)
      .addInputOperandWorkaround(weightWorkaround)
      .addOutputOperandWorkaround(weightWorkaround);
}
} // namespace mlir::tt::ttnn::wa
