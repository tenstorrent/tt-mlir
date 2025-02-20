// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
  // TODO(#2103): This is a temporary fix to handle host tensors
  // If the target buffer type is SystemMemory, set tensor memory layout to
  // nullopt.
  if (results.tensorBufferTypeResult.targetValue == BufferType::SystemMemory) {
    results.tensorMemoryLayoutResult.targetValue = std::nullopt;
  }

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
  // Create input and bf16 workarounds.
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

// Factory method to create a set of workarounds for embedding backward
// operation operands. The embedding backward operation expects the input to be
// in row-major layout and the weight and the in gradient operands to use the
// bf16 data type. Since the output of the embedding operation follows the same
// format as the weight operand, the same workaround is applied to the output
// operand.
//
// Metal issue for input operand workaround:
// https://github.com/tenstorrent/tt-metal/issues/14915
//
// Metal issue weight operand workaround:
// TBD
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createEmbeddingBackwardOpOperandsWorkarounds() {
  // Create input and bf16 workarounds.
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

// Factory method to create a set of workarounds for UpsampleO. The UpsampleOp
// expects the input to be in row-major layout and to use the bf16 data type.
// Since the output of the UpsampleOp follows the same format as the input
// operand, the same workaround is applied to the output operand.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createUpsampleOpOperandsWorkarounds() {
  TTNNOperandWorkarounds rowMajorLayoutBF16Workaround;
  rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround = DataType::BFloat16;
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addOutputOperandWorkaround(rowMajorLayoutBF16Workaround);
}

// Factory method to create a set of workarounds for cumsum operation operands.
// The cumsum op generates incorrect results for integer data types. So input
// tensor is converted to float32 in case of integer input.

// Metal issue for generation of incorrect outputs for integer inputs.
// https://github.com/tenstorrent/tt-mlir/issues/1979

TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createCumSumOpOperandsWorkarounds(
    RankedTensorType inputType) {
  mlir::Type inputElementType = inputType.getElementType();
  // DataType dataType = elementTypeToDataType(inputElementType);
  TTNNOperandWorkarounds typeWorkaround =
      isa<IntegerType>(inputElementType)
          ? TTNNOperandWorkarounds(DataType::Float32)
          : TTNNOperandWorkarounds();
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(typeWorkaround)
      .addInputOperandWorkaround(typeWorkaround)
      .addOutputOperandWorkaround(typeWorkaround);
}

// Factory method to create a set of workarounds for full op output operand.
// ttnn::FullOp does not support 1D tilized tensors
// If the output of full is a 1D tensor and is tiled
// we need to convert it to row major layout then tilize separately
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createFullOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds rowMajorLayoutWorkaround;
  rowMajorLayoutWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(rowMajorLayoutWorkaround);
}

// Factory method to create a set of workarounds for mesh shard op input
// operand. ttnn::MeshShardOp supports host tensors only
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createMeshShardOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds sysMemWorkaround;
  sysMemWorkaround.tensorBufferTypeWorkaround = BufferType::SystemMemory;
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(sysMemWorkaround)
      .addOutputOperandWorkaround(sysMemWorkaround);
}

// Factory method to create a set of workaround for concat operation operands.
// tt-metal applies padding (before concatenation) to the input tensors if the
// layout is tile and the shape is not divisible by tile size along concatenated
// dimension for any input tensor. Padding can only be applied for float or
// bfloat16 tensors.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConcatOpOperandsWorkarounds(
    mlir::Operation::operand_range inputs, int64_t numOperands, int32_t dim) {
  mlir::RankedTensorType inputType =
      mlir::cast<RankedTensorType>(inputs.front().getType());
  ttnn::TTNNLayoutAttr layoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  mlir::Type elementType = inputType.getElementType();

  // Check if the op is using tile layout.
  bool isDataTypeWARequired = layoutAttr.isTiled();
  // Check if the tensor data type is neither float32 nor bfloat16.
  isDataTypeWARequired &= (!elementType.isF32() && !elementType.isBF16());
  // Check if shape (for any input tensor) along concatenated dimension is not
  // divisible by tileHeight (Assuming TileWidth and TileHeigh are same).
  int32_t tileWidth = 1;
  int32_t tileHeight = 1;
  if (isDataTypeWARequired) {
    TileType tile =
        mlir::cast<TileType>(layoutAttr.getMemref().getElementType());
    tileWidth = tile.getWidth();
    tileHeight = tile.getHeight();
  }
  assert(tileHeight == tileWidth);
  isDataTypeWARequired &= llvm::any_of(inputs, [&](mlir::Value value) {
    RankedTensorType inputTensor =
        mlir::dyn_cast<RankedTensorType>(value.getType());
    return inputTensor.getShape()[dim] % tileHeight != 0;
  });

  TTNNOperandWorkarounds bf16Workaround;
  if (isDataTypeWARequired) {
    bf16Workaround.tensorDataTypeWorkaround = DataType::BFloat16;
  }
  auto workaround =
      TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds();
  for (int64_t count = 0; count < numOperands; ++count) {
    workaround.addInputOperandWorkaround(bf16Workaround);
  }
  return workaround.addOutputOperandWorkaround(bf16Workaround);
}

// Factory method to create a set of workarounds for slice op input operands.
// ttnn::SliceOp requires bfloat16 data type for strided slice.
// ttnn::SliceOp requires row major layout if 'begins' elements (corresponding
// to Width and Height) are not divisible by tile width and height.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createSliceOpOperandsWorkarounds(
    ttnn::TTNNLayoutAttr layoutAttr, mlir::ArrayAttr begins,
    mlir::ArrayAttr step) {
  // Check if any element in 'step' is greater than 1, indicating a strided
  // slice operation.
  bool isStridedSliceOp = llvm::any_of(step, [](mlir::Attribute value) {
    mlir::IntegerAttr intAttr = mlir::dyn_cast<mlir::IntegerAttr>(value);
    return intAttr.getInt() > 1;
  });

  // Compute Width Index.
  int64_t idxWidth = begins.size() - 1;
  // Compute Height Index; 0 if input tensor is 1D.
  int64_t idxHeight = begins.size() > 1 ? begins.size() - 2 : 0;

  // Determine if workaround for row major layout is required.
  bool isLayoutWARequired = layoutAttr.isTiled();
  int32_t tileWidth = 1;
  int32_t tileHeight = 1;
  if (isLayoutWARequired) {
    TileType tile =
        mlir::cast<TileType>(layoutAttr.getMemref().getElementType());
    tileWidth = tile.getWidth();
    tileHeight = tile.getHeight();
  }
  isLayoutWARequired &=
      ((mlir::dyn_cast<mlir::IntegerAttr>(begins[idxWidth]).getInt() %
            tileWidth !=
        0) ||
       (mlir::dyn_cast<mlir::IntegerAttr>(begins[idxHeight]).getInt() %
            tileHeight !=
        0));

  TTNNOperandWorkarounds rowMajorLayoutBF16Workaround;
  if (isStridedSliceOp) {
    rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround = DataType::BFloat16;
  }
  if (!isStridedSliceOp && isLayoutWARequired) {
    rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  }
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addOutputOperandWorkaround(rowMajorLayoutBF16Workaround);
}

// ConstantOp is not a TTNN (lib) operation, but it is used to create TTNN
// tensors. Tensor is expected to be on host in ROW_MAJOR layout. This
// workaround is used to gurantee those ivariants.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConstantOpOperandsWorkarounds() {
  TTNNOperandWorkarounds hostRowMajorWorkaround = TTNNOperandWorkarounds();
  hostRowMajorWorkaround.tensorBufferTypeWorkaround = BufferType::SystemMemory;
  hostRowMajorWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(hostRowMajorWorkaround);
}

} // namespace mlir::tt::ttnn::wa
