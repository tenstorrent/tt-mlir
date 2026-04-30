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
class RotaryEmbeddingOp;
class Conv3dOp;
class TopKOp;
class TopKRouterGptOp;
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

  // Create workarounds for pooling 2d with indices op operands.
  static TTNNOperandsWorkarounds createPool2DWithIndicesOpOperandsWorkarounds();

  // Create workarounds for embedding op operands.
  static TTNNOperandsWorkarounds createEmbeddingOpOperandsWorkarounds();

  // Create workarounds for embedding backward op operands.
  static TTNNOperandsWorkarounds createEmbeddingBackwardOpOperandsWorkarounds();

  // Create workarounds for upsample op operands.
  static TTNNOperandsWorkarounds createUpsampleOpOperandsWorkarounds();

  // Create workarounds for mesh shard op operands.
  static TTNNOperandsWorkarounds
  createMeshShardOpOperandsWorkarounds(ttcore::MeshShardType shardType);

  // Create workarounds for mesh partition op operands. The input and output
  // tensors are always in row-major layout.
  // TODO (hshah): Remove once
  // https://github.com/tenstorrent/tt-metal/issues/37676 is fixed.
  static TTNNOperandsWorkarounds createMeshPartitionOpOperandsWorkarounds();

  // Create workarounds for gather op operands. The input and index tensors must
  // always be in TILED layout.
  // tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/41451
  static TTNNOperandsWorkarounds createGatherOpOperandsWorkarounds();

  // Create workarounds for scatter op operands.
  static TTNNOperandsWorkarounds
  createScatterOpOperandsWorkarounds(mlir::Operation *op);

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
  // tt-metal where only supports si32 and float types natively.
  // Integer operands (i8, ui8, ui32) are cast to si32.
  // Float predicate mismatches are cast to match input type.
  static TTNNOperandsWorkarounds
  createWhereOpOperandsWorkarounds(mlir::Operation::operand_range inputs);

  static TTNNOperandsWorkarounds
  createReshapeOpOperandsWorkarounds(RankedTensorType inputType);

  static TTNNOperandsWorkarounds createDropoutOpOperandsWorkarounds();

  static TTNNOperandsWorkarounds
  createUpdateCacheOpOperandsWorkarounds(RankedTensorType updateIndex);

  static TTNNOperandsWorkarounds
  createPagedUpdateCacheOpOperandsWorkarounds(Operation *op);

  static TTNNOperandsWorkarounds
  createPagedFillCacheOpOperandsWorkarounds(Operation *op);

  static TTNNOperandsWorkarounds createSamplingOpOperandsWorkarounds();

  // Create workarounds for binary op operands.
  static TTNNOperandsWorkarounds
  createBinaryOpOperandsWorkarounds(mlir::Operation *op);

  static TTNNOperandsWorkarounds
  createRotaryEmbeddingOpOperandsWorkarounds(ttnn::RotaryEmbeddingOp op);

  static TTNNOperandsWorkarounds createTanhOpOperandsWorkarounds();

  static TTNNOperandsWorkarounds
  createErfOpOperandsWorkarounds(mlir::RankedTensorType inputType);

  // Create workarounds for group norm op operands.
  static TTNNOperandsWorkarounds
  createGroupNormOpOperandsWorkarounds(mlir::Operation *op);

  // Create workarounds for unary bitwise op (bitwise_not) operands.
  static TTNNOperandsWorkarounds
  createUnaryBitwiseOpOperandsWorkarounds(mlir::Operation *op);

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

  // Create workarounds for conv3d op to force BFloat16 data type.
  static TTNNOperandsWorkarounds
  createConv3dOpOperandsWorkarounds(ttnn::Conv3dOp op);

  // Create workarounds for reduction op operands.
  static TTNNOperandsWorkarounds
  createReductionOpOperandsWorkarounds(mlir::Operation *op);

  // Create workaround for reduce (full) product op operands.
  static TTNNOperandsWorkarounds createReduceProdOpOperandsWorkarounds();

  // Create workarounds for sort op operands.
  static TTNNOperandsWorkarounds
  createSortOpOperandsWorkarounds(ttnn::SortOp op);

  // Create workarounds for SDPA ops: cast f32 inputs to bf16.
  // tt-metal SDPA only supports bf16/bfp8_b/bfp4_b.
  // Issue page: https://github.com/tenstorrent/tt-metal/issues/36717
  static TTNNOperandsWorkarounds
  createScaledDotProductAttentionOpOperandsWorkarounds(Operation *op);

  static TTNNOperandsWorkarounds
  createScaledDotProductAttentionDecodeOpOperandsWorkarounds(Operation *op);

  static TTNNOperandsWorkarounds
  createPagedScaledDotProductAttentionDecodeOpOperandsWorkarounds(
      Operation *op);

  static TTNNOperandsWorkarounds
  createPagedFlashMultiLatentAttentionDecodeOpOperandsWorkarounds(
      Operation *op);

  // Create workarounds for sparse_matmul op operands.
  // Sparsity tensor must be in ROW_MAJOR layout.
  // Issue page: https://github.com/tenstorrent/tt-metal/issues/39126
  static TTNNOperandsWorkarounds createSparseMatmulOpOperandsWorkarounds();

  // Create workarounds for all_to_all_dispatch op operands.
  // Expert indices and mapping require uint16 dtype and ROW_MAJOR layout.
  // Issue page: https://github.com/tenstorrent/tt-metal/issues/39127
  static TTNNOperandsWorkarounds createAllToAllDispatchOpOperandsWorkarounds();

  // Create workarounds for all_to_all_dispatch_metadata op operands.
  // Expert indices, scores, and mapping require specific dtypes and ROW_MAJOR
  // layout. Indices/scores outputs are HEIGHT_SHARDED on L1 by the metal
  // kernel.
  static TTNNOperandsWorkarounds
  createAllToAllDispatchMetadataOpOperandsWorkarounds(Operation *op);

  // Create workarounds for all_to_all_combine op operands.
  // Expert metadata and mapping require uint16 dtype and ROW_MAJOR layout.
  // Issue page: https://github.com/tenstorrent/tt-metal/issues/39127
  static TTNNOperandsWorkarounds createAllToAllCombineOpOperandsWorkarounds();

  // Create workarounds for moe_expert_token_remap op operands.
  // expert_metadata requires uint16 dtype and ROW_MAJOR layout.
  // Issue page: https://github.com/tenstorrent/tt-metal/issues/39128
  static TTNNOperandsWorkarounds
  createMoeExpertTokenRemapOpOperandsWorkarounds();

  // Create workarounds for topk ops.
  // Input must be BFloat16 or BFP_BFloat8.
  // Output values must be same data type as input.
  // Output indices must be uint16.
  // Issue page: https://github.com/tenstorrent/tt-metal/issues/40086
  static TTNNOperandsWorkarounds
  createTopKOpOperandsWorkarounds(ttnn::TopKOp op);

  // Create workarounds for topk_router_gpt op.
  // The kernel always returns both outputs (expert_indices, expert_weights) in
  // ROW_MAJOR layout in L1. expert_indices is always forced to UInt16
  // (unconditionally, unlike TopKOp which uses UInt16 or UInt32 depending on
  // dimension size). expert_weights is always forced to BFloat16.
  static TTNNOperandsWorkarounds createTopKRouterGptOpOperandsWorkarounds();
};

} // namespace mlir::tt::ttnn::wa

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNWORKAROUNDSPASS_H
