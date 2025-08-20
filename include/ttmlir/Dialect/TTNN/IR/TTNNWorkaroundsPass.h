// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDSPASS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDSPASS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinTypes.h"
#include <optional>

// TODO (azecevic): Forward declaration is a temporary solution to avoid
// circular dependency issue. https://github.com/tenstorrent/tt-mlir/issues/4405
namespace mlir::tt::ttnn {
class SortOp;
class SliceDynamicOp;
class SliceStaticOp;
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::wa {
using TensorLayoutWorkaround = std::optional<Layout>;
using TensorBufferTypeWorkaround = std::optional<BufferType>;
using TensorMemoryLayoutWorkaround = std::optional<TensorMemoryLayoutAttr>;
using TensorDataTypeWorkaround = std::optional<ttcore::DataType>;

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
      : TTNNOperandWorkarounds(tensorLayoutWorkaround,
                               /*tensorBufferType=*/std::nullopt,
                               /*tensorMemoryLayout=*/std::nullopt,
                               /*tensorDataType=*/std::nullopt) {}

  // Constructor that takes tensor buffer type workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(TensorBufferTypeWorkaround tensorBufferTypeWorkaround)
      : TTNNOperandWorkarounds(/*tensorLayout=*/std::nullopt,
                               tensorBufferTypeWorkaround,
                               /*tensorMemoryLayout=*/std::nullopt,
                               /*tensorDataType=*/std::nullopt) {}

  // Constructor that takes tensor memory layout workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(
      TensorMemoryLayoutWorkaround tensorMemoryLayoutWorkaround)
      : TTNNOperandWorkarounds(
            /*tensorLayout=*/std::nullopt, /*tensorBufferType=*/std::nullopt,
            tensorMemoryLayoutWorkaround, /*tensorDataType=*/std::nullopt) {}

  // Constructor that takes tensor data type workaround and sets the other
  // workarounds to nullopt.
  TTNNOperandWorkarounds(TensorDataTypeWorkaround tensorDataTypeWorkaround)
      : TTNNOperandWorkarounds(
            /*tensorLayout=*/std::nullopt, /*tensorBufferType=*/std::nullopt,
            /*tensorMemoryLayout=*/std::nullopt, tensorDataTypeWorkaround) {}

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
    : public WorkaroundResult<TensorMemoryLayoutAttr> {};

// Data type workaround result struct.
struct DataTypeWorkaroundResult : public WorkaroundResult<ttcore::DataType> {};

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
  // Create workarounds for pooling 2d ops (max_pool2d, avg_pool2d) operands.
  static TTNNOperandsWorkarounds createPool2DOpOperandsWorkarounds();

  // Create workarounds for embedding op operands.
  static TTNNOperandsWorkarounds createEmbeddingOpOperandsWorkarounds();

  // Create workarounds for embedding backward op operands.
  static TTNNOperandsWorkarounds createEmbeddingBackwardOpOperandsWorkarounds();

  // Create workarounds for upsample op operands.
  static TTNNOperandsWorkarounds createUpsampleOpOperandsWorkarounds();

  // Create workarounds for mesh shard op operands.
  static TTNNOperandsWorkarounds
  createMeshShardOpOperandsWorkarounds(ttcore::MeshShardType shardType);

  // Create workarounds for concat op operands.
  static TTNNOperandsWorkarounds
  createConcatOpOperandsWorkarounds(mlir::Operation::operand_range inputs,
                                    int64_t numOperands, int32_t dim);

  // Create workarounds for static slice op operands.
  static TTNNOperandsWorkarounds
  createSliceStaticOpOperandsWorkarounds(ttnn::SliceStaticOp op);

  // Create workarounds for dynamic slice op operands.
  static TTNNOperandsWorkarounds
  createSliceDynamicOpOperandsWorkarounds(ttnn::SliceDynamicOp op);

  // Workaround for tensor creation that is modeled as ConstantOp in TTNN
  // dialect.
  static TTNNOperandsWorkarounds createConstantOpOperandsWorkarounds();

  // Create workarounds for WhereOp operands.
  static TTNNOperandsWorkarounds
  createWhereOpOperandsWorkarounds(mlir::Operation::operand_range inputs);

  static TTNNOperandsWorkarounds
  createReshapeOpOperandsWorkarounds(RankedTensorType inputType);

  static TTNNOperandsWorkarounds
  createUpdateCacheOpOperandsWorkarounds(RankedTensorType updateIndex);

  // Create workarounds for binary op operands.
  static TTNNOperandsWorkarounds
  createBinaryOpOperandsWorkarounds(mlir::Operation *op);

  static TTNNOperandsWorkarounds createTanhOpOperandsWorkarounds();

  // Create workarounds for ArgMax op operands.
  static TTNNOperandsWorkarounds createArgMaxOpOperandsWorkarounds();

  // Create workarounds for pad op operands.
  static TTNNOperandsWorkarounds
  createPadOpOperandsWorkarounds(mlir::TypedValue<mlir::RankedTensorType> input,
                                 ttnn::TTNNLayoutAttr layoutAttr,
                                 llvm::ArrayRef<int32_t> padding);

  // Create workaround for permute op operands.
  static TTNNOperandsWorkarounds
  createPermuteOpOperandWorkaround(mlir::RankedTensorType inputType);

  // Create workarounds for conv2d/convtranspose2d op.
  template <typename T>
  static TTNNOperandsWorkarounds createConvOpOperandsWorkarounds(T op);

  // Create workarounds for reduction op operands.
  static TTNNOperandsWorkarounds
  createReductionOpOperandsWorkarounds(mlir::Operation *op);

  // Create workaround for reduce (full) product op operands.
  static TTNNOperandsWorkarounds
  createReduceProdOpOperandsWorkarounds(mlir::Type elementType,
                                        bool allDimensions);

  // Create workarounds for sort op operands.
  static TTNNOperandsWorkarounds
  createSortOpOperandsWorkarounds(ttnn::SortOp op);

  // Create workarounds for paged scaled dot product attention decode op
  // operands.
  static TTNNOperandsWorkarounds
  createPagedScaledDotProductAttentionDecodeOpOperandsWorkarounds();
};

} // namespace mlir::tt::ttnn::wa

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDSPASS_H
