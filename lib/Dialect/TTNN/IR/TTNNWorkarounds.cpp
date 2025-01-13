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
WorkaroundResults applyWorkarounds(const TTNNOperandWorkarounds &workaround,
                                   const TTNNLayoutAttr &inputLayoutAttr) {
  WorkaroundResults results;
  results.tensorLayoutResult.targetValue =
      workaround.tensorLayoutWorkaround.value_or(inputLayoutAttr.getLayout());
  results.tensorLayoutResult.previousValue = inputLayoutAttr.getLayout();

  results.tensorBufferTypeResult.targetValue =
      workaround.tensorBufferTypeWorkaround.value_or(
          inputLayoutAttr.getBufferType());
  results.tensorBufferTypeResult.previousValue =
      inputLayoutAttr.getBufferType();

  // If the tensor memory layout workaround is present, apply it.
  // Otherwise, return the input tensor memory layout, which may be
  // nullopt if tensor is on host.
  results.tensorMemoryLayoutResult.targetValue =
      workaround.tensorMemoryLayoutWorkaround.has_value()
          ? workaround.tensorMemoryLayoutWorkaround
          : inputLayoutAttr.getMemLayoutOpt();
  results.tensorMemoryLayoutResult.previousValue =
      inputLayoutAttr.getMemLayoutOpt();

  results.tensorDataTypeResult.targetValue =
      workaround.tensorDataTypeWorkaround.value_or(
          inputLayoutAttr.getDataType());
  results.tensorDataTypeResult.previousValue = inputLayoutAttr.getDataType();

  return results;
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

///////////////////////////////////////////////////////////////////////////////
// Factory methods to create a set of workarounds for specific operations
///////////////////////////////////////////////////////////////////////////////

// Factory method to create a set of workarounds for max pool 2d operation
// operands. The max pool 2d operation can accept input in both row-major and
// tile layout, but the output of the operation is strictly in row-major layout.
// In order to keep the output consistent with the input, the row-major
// workaround is applied to both the input and output operands.
// The input and output operands are expected to use the bf16 data type, so the
// bf16 workaround is applied to both the input and output operands.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createMaxPool2DOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds rowMajorLayoutBF16Workaround;
  rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround = DataType::BFloat16;
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addOutputOperandWorkaround(rowMajorLayoutBF16Workaround);
}

// Factory method to create a set of workarounds for embedding operation
// operands. The embedding operation expects the input to be in row-major layout
// and the weight operand to use the bf16 data type. Since the output of the
// embedding operation follows the same format as the weight operand, the same
// workaround is applied to the output operand.
//
// Metal issue for input operand workaround:
// https://github.com/tenstorrent/tt-metal/issues/14915
//
// Metal issue weight operand workaround:
// TBD
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createEmbeddingOpOperandsWorkarounds() {
  // Create input and weight workarounds.
  TTNNOperandWorkarounds inputRowMajorInt32Workaround;
  inputRowMajorInt32Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  inputRowMajorInt32Workaround.tensorDataTypeWorkaround = DataType::UInt32;
  TTNNOperandWorkarounds bf16Workaround =
      TTNNOperandWorkarounds(DataType::BFloat16);
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(0, 0)
      .addInputOperandWorkaround(inputRowMajorInt32Workaround)
      .addInputOperandWorkaround(bf16Workaround)
      .addInputOperandWorkaround(bf16Workaround)
      .addOutputOperandWorkaround(bf16Workaround);
}

TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createEmbeddingBackwardOpOperandsWorkarounds() {
  // Create input and weight workarounds.
  TTNNOperandWorkarounds inputRowMajorInt32Workaround;
  inputRowMajorInt32Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  inputRowMajorInt32Workaround.tensorDataTypeWorkaround = DataType::UInt32;
  TTNNOperandWorkarounds bf16Workaround =
      TTNNOperandWorkarounds(DataType::BFloat16);
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(0, 0)
      .addInputOperandWorkaround(inputRowMajorInt32Workaround)
      .addInputOperandWorkaround(bf16Workaround)
      .addInputOperandWorkaround(bf16Workaround)
      .addInputOperandWorkaround(bf16Workaround)
      .addOutputOperandWorkaround(bf16Workaround);
}
} // namespace mlir::tt::ttnn::wa
