// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <mlir/IR/BuiltinTypes.h>
#include <optional>

namespace mlir::tt::ttnn::wa {
using TensorLayoutWorkaround = std::optional<Layout>;
using TensorBufferTypeWorkaround = std::optional<BufferType>;
using TensorMemoryLayoutWorkaround = std::optional<TensorMemoryLayout>;
using TensorDataTypeWorkaround = std::optional<DataType>;

// Struct that encapsulates operand workarounds.
// It contains tensor layout, tensor buffer type and tensor memory layout
// workarounds.
struct TTNNOperandWorkarounds {
  // Tensor layout workaround.
  TensorLayoutWorkaround tensorLayoutWorkaround;

  // Tensor buffer type workaround.
  TensorBufferTypeWorkaround tensorBufferTypeWorkaround;

  // Tensor memory layout workaround.
  TensorMemoryLayoutWorkaround tensorMemoryLayoutWorkaround;

  // Tensor data format workaround.
  TensorDataTypeWorkaround tensorDataTypeWorkaround;

  // Default constructor.
  TTNNOperandWorkarounds() = default;

  // Constructor that takes tensor layout, tensor buffer type and tensor memory.
  TTNNOperandWorkarounds(
      TensorLayoutWorkaround tensorLayoutWorkaround,
      TensorBufferTypeWorkaround tensorBufferTypeWorkaround,
      TensorMemoryLayoutWorkaround tensorMemoryLayoutWorkaround,
      TensorDataTypeWorkaround tensorDataTypeWorkaround)
      : tensorLayoutWorkaround(tensorLayoutWorkaround),
        tensorBufferTypeWorkaround(tensorBufferTypeWorkaround),
        tensorMemoryLayoutWorkaround(tensorMemoryLayoutWorkaround),
        tensorDataTypeWorkaround(tensorDataTypeWorkaround) {}

  // Constructor that takes tensor layout workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(TensorLayoutWorkaround tensorLayoutWorkaround)
      : TTNNOperandWorkarounds(tensorLayoutWorkaround, std::nullopt,
                               std::nullopt, std::nullopt) {}

  // Constructor that takes tensor buffer type workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(TensorBufferTypeWorkaround tensorBufferTypeWorkaround)
      : TTNNOperandWorkarounds(std::nullopt, tensorBufferTypeWorkaround,
                               std::nullopt, std::nullopt) {}

  // Constructor that takes tensor memory layout workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(
      TensorMemoryLayoutWorkaround tensorMemoryLayoutWorkaround)
      : TTNNOperandWorkarounds(std::nullopt, std::nullopt,
                               tensorMemoryLayoutWorkaround, std::nullopt) {}

  // Constructor that takes tensor data type workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(TensorDataTypeWorkaround tensorDataTypeWorkaround)
      : TTNNOperandWorkarounds(std::nullopt, std::nullopt, std::nullopt,
                               tensorDataTypeWorkaround) {}

  // Operand workarounds factory methods.
  static TTNNOperandWorkarounds createEmptyTTNNOperandWorkarounds();

  // Equality operator.
  bool operator==(const TTNNOperandWorkarounds &rhs) const {
    return tensorLayoutWorkaround == rhs.tensorLayoutWorkaround &&
           tensorBufferTypeWorkaround == rhs.tensorBufferTypeWorkaround &&
           tensorMemoryLayoutWorkaround == rhs.tensorMemoryLayoutWorkaround &&
           tensorDataTypeWorkaround == rhs.tensorDataTypeWorkaround;
  }

  // Inequality operator.
  bool operator!=(const TTNNOperandWorkarounds &rhs) const {
    return !(*this == rhs);
  }

  // Returns true if any of the workarounds is set.
  bool hasAnyWorkaround() const {
    return tensorLayoutWorkaround || tensorBufferTypeWorkaround ||
           tensorMemoryLayoutWorkaround || tensorDataTypeWorkaround;
  }
};

// Workaround result struct that encapsulates the previous and target
// (workaround) value and a method indicating whether the workaround modifies
// the workaround value.
template <typename T>
struct WorkaroundResult {
  T previousValue;
  T targetValue;
  bool isModified() const { return previousValue != targetValue; }
};

// Layout workaround result struct.
struct LayoutWorkaroundResult : public WorkaroundResult<Layout> {};

// Buffer type workaround result struct.
struct BufferTypeWorkaroundResult : public WorkaroundResult<BufferType> {};

// Memory layout workaround result struct.
struct MemoryLayoutWorkaroundResult
    : public WorkaroundResult<std::optional<TensorMemoryLayout>> {};

// Data type workaround result struct.
struct DataTypeWorkaroundResult : public WorkaroundResult<DataType> {};

// Struct that encapsulates the result of applying the workarounds.
// It contains the target tensor layout, buffer type and tensor memory layout
// results and a flag indicating whether the workarounds were applied.
struct WorkaroundResults {
  // Tensor layout workaround result.
  LayoutWorkaroundResult tensorLayoutResult;

  // Tensor buffer type workaround result.
  BufferTypeWorkaroundResult tensorBufferTypeResult;

  // Tensor memory layout workaround result.
  MemoryLayoutWorkaroundResult tensorMemoryLayoutResult;

  // Tensor data type workaround result.
  DataTypeWorkaroundResult tensorDataTypeResult;

  // Returns true if any of the workarounds were applied.
  bool isModified() const {
    return tensorLayoutResult.isModified() ||
           tensorBufferTypeResult.isModified() ||
           tensorMemoryLayoutResult.isModified() ||
           tensorDataTypeResult.isModified();
  }
};

// Apply the operand workarounds to the layout attribute that contains
// tensor layout, buffer type and tensor memory layout arguments.
// Returns the result of applying the workarounds.
WorkaroundResults applyWorkarounds(const TTNNOperandWorkarounds &workaround,
                                   const TTNNLayoutAttr &inputLayoutAttr);

// Class that encapsulates operands workarounds.
// It contains input and output workarounds for operands.
class TTNNOperandsWorkarounds {
public:
  // Returns input operand workarounds.
  llvm::ArrayRef<TTNNOperandWorkarounds> getInputOperandWorkarounds() const {
    return inputOperandWorkarounds;
  }

  // Returns output operand workarounds.
  llvm::ArrayRef<TTNNOperandWorkarounds> getOutputOperandWorkarounds() const {
    return outputOperandWorkarounds;
  }

  // Adds input operand workaround.
  TTNNOperandsWorkarounds &
  addInputOperandWorkaround(TTNNOperandWorkarounds inputOperandWorkaround) {
    inputOperandWorkarounds.emplace_back(inputOperandWorkaround);
    return *this;
  }

  // Adds output operand workaround.
  TTNNOperandsWorkarounds &
  addOutputOperandWorkaround(TTNNOperandWorkarounds outputOperandWorkaround) {
    outputOperandWorkarounds.emplace_back(outputOperandWorkaround);
    return *this;
  }

  // Operands workarounds factory method.
  static TTNNOperandsWorkarounds
  createEmptyTTNNOperandsWorkarounds(int inputSize, int outputSize);

  // Operands workarounds factory method.
  static TTNNOperandsWorkarounds createEmptyTTNNOperandsWorkarounds() {
    return createEmptyTTNNOperandsWorkarounds(0, 0);
  }

  // Operands workarounds factory method.
  static TTNNOperandsWorkarounds
  createEmptyTTNNOperandsWorkarounds(Operation *op);

private:
  // Default constructor with no workarounds.
  TTNNOperandsWorkarounds() {}

  // Constructor that takes input and output workarounds for operands.
  TTNNOperandsWorkarounds(
      llvm::SmallVector<TTNNOperandWorkarounds> inputOperandWorkarounds,
      llvm::SmallVector<TTNNOperandWorkarounds> outputOperandWorkarounds)
      : inputOperandWorkarounds(std::move(inputOperandWorkarounds)),
        outputOperandWorkarounds(std::move(outputOperandWorkarounds)) {}

  // Workarounds for input operands.
  llvm::SmallVector<TTNNOperandWorkarounds> inputOperandWorkarounds;

  // Workarounds for output operands.
  llvm::SmallVector<TTNNOperandWorkarounds> outputOperandWorkarounds;
};

// Workaround factory class that creates workarounds for ops.
class TTNNOperandsWorkaroundsFactory {
public:
  // Create workarounds for max_pool2d op operands.
  static TTNNOperandsWorkarounds createMaxPool2DOpOperandsWorkarounds();

  // Create workarounds for embedding op operands.
  static TTNNOperandsWorkarounds createEmbeddingOpOperandsWorkarounds();

  // Create workarounds for embedding backward op operands.
  static TTNNOperandsWorkarounds createEmbeddingBackwardOpOperandsWorkarounds();

  // Create workarounds for upsample op operands.
  static TTNNOperandsWorkarounds createUpsampleOpOperandsWorkarounds();

  static TTNNOperandsWorkarounds
  createCumSumOpOperandsWorkarounds(RankedTensorType inputType);

  // Create workarounds for full op operands.
  static TTNNOperandsWorkarounds createFullOpOperandsWorkarounds();

  // Create workarounds for mesh shard op operands.
  static TTNNOperandsWorkarounds createMeshShardOpOperandsWorkarounds();

  // Create workarounds for concat op operands.
  static TTNNOperandsWorkarounds
  createConcatOpOperandsWorkarounds(mlir::Operation::operand_range inputs,
                                    int64_t numOperands, int32_t dim);

  // Create workarounds for slice op operands.
  static TTNNOperandsWorkarounds
  createSliceOpOperandsWorkarounds(ttnn::TTNNLayoutAttr layoutAttr,
                                   mlir::ArrayAttr begins,
                                   mlir::ArrayAttr step);

  static TTNNOperandsWorkarounds createConstantOpOperandsWorkarounds();
};

} // namespace mlir::tt::ttnn::wa

#endif
