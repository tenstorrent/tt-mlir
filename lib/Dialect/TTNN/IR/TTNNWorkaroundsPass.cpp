// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
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

// Factory method to create a set of workarounds for 2d pooling operations
// (avg_pool2d and max_pool2d) operands. The 2d pooling operation can accept
// input in both row-major and tile layout, but the output of the operation is
// strictly in row-major layout. In order to keep the output consistent with the
// input, the row-major workaround is applied to both the input and output
// operands. The input and output operands are expected to use the bf16 data
// type, so the bf16 workaround is applied to both the input and output
// operands.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createPool2DOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds rowMajorLayoutBF16Workaround;
  rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround =
      ttcore::DataType::BFloat16;
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
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
  inputRowMajorInt32Workaround.tensorDataTypeWorkaround =
      ttcore::DataType::UInt32;
  TTNNOperandWorkarounds bf16Workaround =
      TTNNOperandWorkarounds(ttcore::DataType::BFloat16);
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(0, 0)
      .addInputOperandWorkaround(inputRowMajorInt32Workaround)
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
  inputRowMajorInt32Workaround.tensorDataTypeWorkaround =
      ttcore::DataType::UInt32;
  TTNNOperandWorkarounds bf16Workaround =
      TTNNOperandWorkarounds(ttcore::DataType::BFloat16);
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(0, 0)
      .addInputOperandWorkaround(inputRowMajorInt32Workaround)
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
  rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround =
      ttcore::DataType::BFloat16;
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addOutputOperandWorkaround(rowMajorLayoutBF16Workaround);
}

// Factory method to create a set of workarounds for zeros op output operand.
// ttnn::zeros does not support output dtype int32. If the output data type of
// ttnn::zeros is int32, we override to float32 and typecast separately.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createZerosOpOperandsWorkarounds(
    RankedTensorType outputType) {
  wa::TTNNOperandWorkarounds fullOpOutputWorkarounds;
  mlir::tt::ttcore::DataType dataType =
      mlir::tt::ttcore::elementTypeToDataType(outputType.getElementType());
  if (dataType == mlir::tt::ttcore::DataType::Int32) {
    fullOpOutputWorkarounds.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::Float32;
  }
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(fullOpOutputWorkarounds);
}

// Factory method to create a set of workarounds for full op output operand.
// ttnn::FullOp does not support 1D tilized tensors
// If the output of full is a 1D tensor and is tiled
// we need to convert it to row major layout then tilize separately
// ttnn::full does not support output dtype int32. If the output data type of
// full is int32, we override to float32 and typecast separately.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createFullOpOperandsWorkarounds(
    RankedTensorType outputType) {
  wa::TTNNOperandWorkarounds fullOpOutputWorkarounds;
  ttnn::TTNNLayoutAttr layoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());
  if (outputType.getRank() == 1 && layoutAttr.isTiled()) {
    fullOpOutputWorkarounds.tensorLayoutWorkaround = Layout::RowMajor;
  }
  mlir::tt::ttcore::DataType dataType =
      mlir::tt::ttcore::elementTypeToDataType(outputType.getElementType());
  if (dataType == mlir::tt::ttcore::DataType::Int32) {
    fullOpOutputWorkarounds.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::Float32;
  }
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(fullOpOutputWorkarounds);
}

// Factory method to create a set of workarounds for mesh shard op input
// operand. ttnn::MeshShardOp supports host tensors only
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createMeshShardOpOperandsWorkarounds(
    mlir::tt::ttcore::MeshShardType shardType) {
  wa::TTNNOperandWorkarounds sysMemWorkaround;
  if (shardType != mlir::tt::ttcore::MeshShardType::Identity) {
    sysMemWorkaround.tensorBufferTypeWorkaround = BufferType::SystemMemory;
  }
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
    ttcore::TileType tile =
        mlir::cast<ttcore::TileType>(layoutAttr.getMemref().getElementType());
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
    bf16Workaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;
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
    ttcore::TileType tile =
        mlir::cast<ttcore::TileType>(layoutAttr.getMemref().getElementType());
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
    rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround =
        ttcore::DataType::BFloat16;
  }
  if (!isStridedSliceOp && isLayoutWARequired) {
    rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  }
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addOutputOperandWorkaround(rowMajorLayoutBF16Workaround);
}

// ConstantOp is not a TTNN (lib) operation, but it is used to create TTNN
// tensors. Tensor is expected to be on host in ROW_MAJOR layout. This
// workaround is used to guarantee those ivariants.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConstantOpOperandsWorkarounds() {
  TTNNOperandWorkarounds hostRowMajorWorkaround = TTNNOperandWorkarounds();
  hostRowMajorWorkaround.tensorBufferTypeWorkaround = BufferType::SystemMemory;
  hostRowMajorWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(hostRowMajorWorkaround);
}

// Factory method to create a set of workarounds for where op operands.
// tt-metal uses predicate type for where op operation. If the predicate data
// type does not match with inputs/output data type; tt-metal can generate
// incorrect results or other failures. Add a data type workaround if predicate
// type does not match with input.
// tt-metal issue to track mixed data types ops bug.
// https://github.com/tenstorrent/tt-metal/issues/17998
// Where also does not work with int32
// so we also force everything to float32 in that case
// https://github.com/tenstorrent/tt-mlir/issues/3154
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createWhereOpOperandsWorkarounds(
    mlir::Operation::operand_range inputs) {
  // Extract predicate type; defined as first input in TTNN Dialect.
  mlir::RankedTensorType predicateType =
      mlir::cast<RankedTensorType>(inputs.front().getType());
  mlir::Type predicateElementType = predicateType.getElementType();
  // Use last input to determine input data type.
  mlir::RankedTensorType inputType =
      mlir::cast<RankedTensorType>(inputs.back().getType());
  mlir::Type inputElementType = inputType.getElementType();
  TTNNOperandWorkarounds typeWorkaround = TTNNOperandWorkarounds();
  if (predicateElementType.isInteger() || inputElementType.isInteger()) {
    typeWorkaround = TTNNOperandWorkarounds(ttcore::DataType::Float32);
  } else if (predicateElementType != inputElementType) {
    typeWorkaround =
        TTNNOperandWorkarounds(ttcore::elementTypeToDataType(inputElementType));
  }

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(typeWorkaround)
      .addInputOperandWorkaround(typeWorkaround)
      .addInputOperandWorkaround(typeWorkaround)
      .addOutputOperandWorkaround(typeWorkaround);
}

// Factory method to create a set of workarounds for reshape operation operands.
// Reshape op only does not work with int32 - force to float32 then typecast
// separately.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createReshapeOpOperandsWorkarounds(
    RankedTensorType inputType) {
  mlir::Type inputElementType = inputType.getElementType();
  TTNNOperandWorkarounds typeWorkarounds;
  mlir::tt::ttcore::DataType dataType =
      mlir::tt::ttcore::elementTypeToDataType(inputElementType);
  if (dataType == mlir::tt::ttcore::DataType::Int32) {
    typeWorkarounds.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::Float32;
  }
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(typeWorkarounds)
      .addOutputOperandWorkaround(typeWorkarounds);
}

// Factory method to create a set of workarounds for UpdateCache operation
// operands. Update index of UpdateCacheOp must be unsigned
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createUpdateCacheOpOperandsWorkarounds(
    RankedTensorType updateIndex) {
  mlir::Type updateIndexElementType = updateIndex.getElementType();
  TTNNOperandWorkarounds nullWorkarounds;
  TTNNOperandWorkarounds typeWorkarounds;
  mlir::tt::ttcore::DataType dataType =
      mlir::tt::ttcore::elementTypeToDataType(updateIndexElementType);
  if (dataType == mlir::tt::ttcore::DataType::Int32) {
    typeWorkarounds.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::UInt32;
  }
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(nullWorkarounds)
      .addInputOperandWorkaround(nullWorkarounds)
      .addInputOperandWorkaround(typeWorkarounds);
}

// Helper function to determine if data type workaround is required for a binary
// op. Set the workaround data type based on the binary op.
static std::optional<mlir::tt::ttcore::DataType>
binaryOpDTypeWorkaround(mlir::Operation *op, mlir::Type elementType) {
  mlir::tt::ttcore::DataType dType =
      mlir::tt::ttcore::elementTypeToDataType(elementType);

  if (isa<ttnn::AddOp, ttnn::SubtractOp>(op)) {
    if (dType == mlir::tt::ttcore::DataType::Float32 ||
        dType == mlir::tt::ttcore::DataType::BFloat16 ||
        dType == mlir::tt::ttcore::DataType::BFP_BFloat8 ||
        dType == mlir::tt::ttcore::DataType::BFP_BFloat4) {
      return {};
    }
    if (dType == mlir::tt::ttcore::DataType::Int32) {
      // Although TTNN claims to support int32 for Add and Subtract ops,
      // broadcasting with int32 inputs does not currently work as expected.
      // As a temporary workaround, we fall back to BFloat16 when input shapes
      // differ. This should be removed once int32 broadcasting is properly
      // supported.
      auto lhsType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
      auto rhsType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());

      if (lhsType.getShape() != rhsType.getShape()) {
        return mlir::tt::ttcore::DataType::BFloat16;
      }
      return {};
    }
    return mlir::tt::ttcore::DataType::BFloat16;
  }
  // Left shift and right shift ops have same requirements but they are not
  // implemented for TTNN dialect currently.
  if (isa<ttnn::BitwiseAndOp, ttnn::BitwiseOrOp, ttnn::BitwiseXorOp>(op)) {
    if (dType == mlir::tt::ttcore::DataType::Int32) {
      return {};
    }
    return mlir::tt::ttcore::DataType::Int32;
  }
  // All remaining binary ops.
  if (dType == mlir::tt::ttcore::DataType::Float32 ||
      dType == mlir::tt::ttcore::DataType::BFloat16 ||
      dType == mlir::tt::ttcore::DataType::BFP_BFloat8 ||
      dType == mlir::tt::ttcore::DataType::BFP_BFloat4) {
    return {};
  }
  return mlir::tt::ttcore::DataType::BFloat16;
}

// Factory method to create a set of workarounds for binary operation operands.
// This workaround is based on tt-metal PR for data type checker for binary ops.
// https://github.com/tenstorrent/tt-metal/pull/17828
// Apply the workaround if any of the input does not satisfy the data type
// requirement.
// Elementwise binary ops requires TILE layout. Apply layout workaround if any
// of the input is using ROW_MAJOR layout.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createBinaryOpOperandsWorkarounds(
    mlir::Operation *op) {
  assert(op->getNumOperands() == 2 && "expected binary op");
  auto lhsType =
      mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto rhsType =
      mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());

  TTNNOperandWorkarounds operandWorkaround;
  if (auto dtype = binaryOpDTypeWorkaround(op, lhsType.getElementType())) {
    operandWorkaround.tensorDataTypeWorkaround = *dtype;
  }
  if (auto dtype = binaryOpDTypeWorkaround(op, rhsType.getElementType())) {
    operandWorkaround.tensorDataTypeWorkaround = *dtype;
  }
  if (!mlir::cast<ttnn::TTNNLayoutAttr>(lhsType.getEncoding()).isTiled()) {
    operandWorkaround.tensorLayoutWorkaround = Layout::Tile;
  }
  if (!mlir::cast<ttnn::TTNNLayoutAttr>(rhsType.getEncoding()).isTiled()) {
    operandWorkaround.tensorLayoutWorkaround = Layout::Tile;
  }

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)
      .addInputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(operandWorkaround);
}

TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createTanhOpOperandsWorkarounds() {
  TTNNOperandWorkarounds operandWorkaround;
  // Tanh op accurate mode requires bfloat16 data type.
  // Issue: https://github.com/tenstorrent/tt-metal/issues/22593
  operandWorkaround.tensorDataTypeWorkaround =
      mlir::tt::ttcore::DataType::BFloat16;

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(operandWorkaround);
}

// Factory method to create a set of workarounds for ArgMax op operands.
// Input tensor must have BFLOAT16 data type and ROW_MAJOR layout.
// No need for data type workaround for output tensor; only layout workaround is
// required to match original layout.
// tt-metal specs:
// https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.argmax.html
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createArgMaxOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds rowMajorLayoutWorkaround;
  rowMajorLayoutWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  wa::TTNNOperandWorkarounds rowMajorLayoutBF16Workaround;
  rowMajorLayoutBF16Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorLayoutBF16Workaround.tensorDataTypeWorkaround =
      mlir::tt::ttcore::DataType::BFloat16;

  wa::TTNNOperandWorkarounds rowMajorLayoutUint32Workaround;
  rowMajorLayoutUint32Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorLayoutUint32Workaround.tensorDataTypeWorkaround =
      mlir::tt::ttcore::DataType::UInt32;

  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutBF16Workaround)
      .addOutputOperandWorkaround(rowMajorLayoutUint32Workaround);
}

// Factory method to create a set of workarounds for Pad op operands.
// tt-metal only supports float32 and bfloat16 data types.
// tt-metal generates incorrect output for tile layout.
// https://github.com/tenstorrent/tt-metal/issues/19513
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createPadOpOperandsWorkarounds(
    mlir::TypedValue<mlir::RankedTensorType> input,
    ttnn::TTNNLayoutAttr layoutAttr) {
  TTNNOperandWorkarounds operandWorkaround;
  if (layoutAttr.isTiled()) {
    operandWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  }
  if (isa<IntegerType>(input.getType().getElementType())) {
    operandWorkaround.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::BFloat16;
  }
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(operandWorkaround);
}

// ttnn.permute will not work correctly on 32-bit integers. This workaround will
// typecast the 32-bit integers to 32-bit floats before the ttnn.permute op and
// typecast the output back.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/19950
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createPermuteOpOperandWorkaround(
    mlir::RankedTensorType inputType) {
  TTNNOperandWorkarounds operandWorkaround;
  if (auto elementType = dyn_cast<IntegerType>(inputType.getElementType());
      elementType && elementType.getWidth() == 32) {
    operandWorkaround.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::Float32;
  }

  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(operandWorkaround);
}

// Currently, there is more support for conv2d and conv_transpose2d for
// row-major inputs than there is for tile inputs.
// There is no single issue in tt-metal for this. This workaround is here
// to ensure we use the more generally-supported input layout for
// convolutions in ttnn. For example, here is an issue highliting
// some convolutions that will not work when the input is in tile layout,
// but will work when the input is in row-major layout:
// https://github.com/tenstorrent/tt-metal/issues/19762
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConv2dOpOperandsWorkarounds(
    bool hasBias) {

  TTNNOperandWorkarounds inputWorkaround;
  inputWorkaround.tensorLayoutWorkaround = Layout::RowMajor;

  // Convolution outputs are always in tile layout regardless
  // of the input layout. We explicitly state this here to
  // avoid accidentally assigning the output of a convolution
  // to row major layout just because its input is row major.
  TTNNOperandWorkarounds outputWorkaround;
  outputWorkaround.tensorLayoutWorkaround = Layout::Tile;

  TTNNOperandWorkarounds parameterWorkaround;

  auto workaround =
      wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
          .addInputOperandWorkaround(inputWorkaround)
          .addInputOperandWorkaround(parameterWorkaround)
          .addOutputOperandWorkaround(outputWorkaround);

  if (hasBias) {
    workaround = workaround.addInputOperandWorkaround(parameterWorkaround);
  }
  return workaround;
}

// TTNN Arange op only supports row-major output. Adding workaround to enforce
// row-major layout on its output.
// tt-metal issue to support tile layout for arange op:
// https://github.com/tenstorrent/tt-metal/issues/20251
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createArangeOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds arangeOpOperandWorkaround(Layout::RowMajor);
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(arangeOpOperandWorkaround);
}

// Factory method to create a set of workaround for reduction ops operands.
// Data type workaround is required for reduction ops for following cases:
// 1. Reduction ops requires padding if the input tensor shape is not same as
//    padded shape and padding op only supports bfloat16 and float32.
// 2. Reduction ops requires transpose in some cases which support bfloat16 and
//    float32 data types only.
// 3. Reduction ops generates incorrect output for integer input tensor.
//    tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/21071
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createReductionOpOperandsWorkarounds(
    mlir::Operation *op) {
  auto inputType =
      mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType())
          .getElementType();
  TTNNOperandWorkarounds operandWorkaround;
  if (!inputType.isF32() && !inputType.isBF16()) {
    operandWorkaround.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::BFloat16;
  }

  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(operandWorkaround);
}

// Factory method to create a set of workarounds for reduce product op operands.
// tt-metal only supports full product reduction for bfloat16 data type.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createReduceProdOpOperandsWorkarounds(
    mlir::Type elementType, bool allDimensions) {
  bool isDataTypeWARequired = allDimensions && !elementType.isBF16();
  TTNNOperandWorkarounds bf16Workaround;
  if (isDataTypeWARequired) {
    bf16Workaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;
  }

  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(bf16Workaround)
      .addOutputOperandWorkaround(bf16Workaround);
}
} // namespace mlir::tt::ttnn::wa
