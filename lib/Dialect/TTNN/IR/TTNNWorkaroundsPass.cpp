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
          ? workaround.tensorMemoryLayoutWorkaround.value()
          : inputLayoutAttr.getMemLayout();
  results.tensorMemoryLayoutResult.previousValue =
      inputLayoutAttr.getMemLayout();

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

// Factory method to create a set of workarounds for 2d max pooling with indices.
// This operation returns two outputs:
//   [0] pooled values (bf16 row-major)
//   [1] indices (row-major)
// The input can be in row-major or tile layout, but outputs are always row-major.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createPool2DWithIndicesOpOperandsWorkarounds() {
  // Input workaround: bf16 row-major.
  wa::TTNNOperandWorkarounds inputWorkaround;
  inputWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  inputWorkaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;

  // Output[0] workaround: pooled values (bf16 row-major).
  wa::TTNNOperandWorkarounds outputValuesWorkaround;
  outputValuesWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  outputValuesWorkaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;

  // Output[1] workaround: indices (row-major).
  wa::TTNNOperandWorkarounds outputIndicesWorkaround;
  outputIndicesWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  outputIndicesWorkaround.tensorDataTypeWorkaround = ttcore::DataType::UInt16;

  // Create empty workarounds and append input + both outputs.
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(inputWorkaround)
      .addOutputOperandWorkaround(outputValuesWorkaround)
      .addOutputOperandWorkaround(outputIndicesWorkaround);
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

// Factory method to create a set of workarounds for UpsampleOp. The UpsampleOp
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

// Factory method to create a set of workarounds for ScatterOp. The ScatterOp
// expects the input to be in row-major layout if using f32.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createScatterOpOperandsWorkarounds(
    mlir::Operation *op) {
  auto scatterOp = mlir::cast<ttnn::ScatterOp>(op);
  auto inputType =
      mlir::cast<mlir::RankedTensorType>(scatterOp.getInput().getType());
  auto sourceType =
      mlir::cast<mlir::RankedTensorType>(scatterOp.getSource().getType());

  ttnn::TTNNLayoutAttr inputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  ttnn::TTNNLayoutAttr sourceLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(sourceType.getEncoding());

  bool isLayoutWorkaroundRequired =
      (inputLayoutAttr.isTiled() && inputType.getElementType().isF32()) ||
      (sourceLayoutAttr.isTiled() && sourceType.getElementType().isF32());

  TTNNOperandWorkarounds operandWorkaround;

  if (isLayoutWorkaroundRequired) {
    operandWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  }

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)   // input
      .addInputOperandWorkaround(operandWorkaround)   // index
      .addInputOperandWorkaround(operandWorkaround)   // source
      .addOutputOperandWorkaround(operandWorkaround); // result
}

// Factory method to create a set of workarounds for mesh shard op input
// operand. ttnn::MeshShardOp supports host tensors only
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createMeshShardOpOperandsWorkarounds(
    mlir::tt::ttcore::MeshShardType shardType) {
  wa::TTNNOperandWorkarounds sysMemWorkaround;
  if (shardType != mlir::tt::ttcore::MeshShardType::Identity) {
    sysMemWorkaround.tensorBufferTypeWorkaround = BufferType::SystemMemory;
    sysMemWorkaround.tensorMemoryLayoutWorkaround = TensorMemoryLayoutAttr();
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
// ttnn::SliceStaticOp requires uint32 on input if the slice is strided
// and input is < uint32.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createSliceStaticOpOperandsWorkarounds(
    ttnn::SliceStaticOp op) {
  // Check if any element in 'step' is greater than 1, indicating a strided
  // slice operation.
  bool isStridedSliceOp = llvm::any_of(op.getStep(), [](mlir::Attribute value) {
    mlir::IntegerAttr intAttr = mlir::dyn_cast<mlir::IntegerAttr>(value);
    return intAttr.getInt() > 1;
  });

  TTNNOperandWorkarounds workaround;
  Type inputType = op.getInput().getType().getElementType();
  uint32_t bitWidth = inputType.getIntOrFloatBitWidth();
  if (inputType.isUnsignedInteger() && bitWidth < 32 && isStridedSliceOp) {
    workaround.tensorDataTypeWorkaround = ttcore::DataType::UInt32;
  }
  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(workaround)
      .addOutputOperandWorkaround(workaround);
}

// Factory method to create a set of workarounds for dynamic slice op input
// operands. ttnn::SliceDynamicOp requires uint32 for inputs if
// the input is < uint32.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createSliceDynamicOpOperandsWorkarounds(
    ttnn::SliceDynamicOp op) {
  TTNNOperandWorkarounds inputWorkaround;
  Type inputType = op.getInput().getType().getElementType();
  uint32_t bitWidth = inputType.getIntOrFloatBitWidth();
  if (inputType.isUnsignedInteger() && bitWidth < 32) {
    inputWorkaround.tensorDataTypeWorkaround = ttcore::DataType::UInt32;
  }
  TTNNOperandWorkarounds uInt32Workaround;
  uInt32Workaround.tensorDataTypeWorkaround = ttcore::DataType::UInt32;

  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(inputWorkaround)
      .addInputOperandWorkaround(uInt32Workaround)
      .addInputOperandWorkaround(uInt32Workaround)
      .addOutputOperandWorkaround(inputWorkaround);
}

// ConstantOp is not a TTNN (lib) operation, but it is used to create TTNN
// tensors. Tensor is expected to be on host in ROW_MAJOR layout. This
// workaround is used to guarantee those invariants.
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConstantOpOperandsWorkarounds() {
  TTNNOperandWorkarounds hostRowMajorWorkaround = TTNNOperandWorkarounds();
  hostRowMajorWorkaround.tensorBufferTypeWorkaround = BufferType::SystemMemory;
  hostRowMajorWorkaround.tensorMemoryLayoutWorkaround =
      TensorMemoryLayoutAttr();
  hostRowMajorWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addOutputOperandWorkaround(hostRowMajorWorkaround);
}

// Factory method to create a set of workarounds for where op operands.
// tt-metal uses predicate type for where op operation. If the predicate data
// type does not match with inputs/output data type; tt-metal can generate
// incorrect results or other failures. Add a data type workaround if predicate
// type does not match with input. Also, if predicate is integer, force it to
// match input data type, unless both are integers, then force both to
// float32.
// tt-metal issues to track mixed data types ops bug.
// https://github.com/tenstorrent/tt-metal/issues/17998
// https://github.com/tenstorrent/tt-metal/issues/24511
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
  TTNNOperandWorkarounds predicateTypeWorkaround = TTNNOperandWorkarounds();
  TTNNOperandWorkarounds inputTypeWorkaround;
  TTNNOperandWorkarounds outputTypeWorkaround;

  if (predicateElementType.isInteger() ||
      predicateElementType != inputElementType) {
    if (inputElementType.isInteger()) {
      // In an unlikely scenario, we could potentially upcast to float32, if
      // input is integer and predicate is for example bf16.
      // More importantly, if both are integers, we force both to float32.
      predicateTypeWorkaround =
          TTNNOperandWorkarounds(ttcore::DataType::Float32);
      inputTypeWorkaround = TTNNOperandWorkarounds(ttcore::DataType::Float32);
      outputTypeWorkaround = TTNNOperandWorkarounds(ttcore::DataType::Float32);
    } else {
      // Otherwise, we just force the predicate type to match the input type.
      predicateTypeWorkaround = TTNNOperandWorkarounds(
          ttcore::elementTypeToDataType(inputElementType));
    }
  }

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(predicateTypeWorkaround)
      .addInputOperandWorkaround(inputTypeWorkaround)
      .addInputOperandWorkaround(inputTypeWorkaround)
      .addOutputOperandWorkaround(outputTypeWorkaround);
}

// Factory method to create a set of workarounds for reshape operation operands.
// Reshape op only does not work with ui8 - force to int32 then typecast
// separately.
// TT-metal issue: https://github.com/tenstorrent/tt-metal/issues/27843
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createReshapeOpOperandsWorkarounds(
    RankedTensorType inputType) {
  mlir::Type inputElementType = inputType.getElementType();
  TTNNOperandWorkarounds typeWorkarounds;
  mlir::tt::ttcore::DataType dataType =
      mlir::tt::ttcore::elementTypeToDataType(inputElementType);
  if (dataType == mlir::tt::ttcore::DataType::UInt8) {
    typeWorkarounds.tensorDataTypeWorkaround =
        mlir::tt::ttcore::DataType::Int32;
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

  if (isa<ttnn::LogicalRightShiftOp, ttnn::LogicalLeftShiftOp>(op)) {
    if (dType == mlir::tt::ttcore::DataType::UInt32 ||
        dType == mlir::tt::ttcore::DataType::Int32) {
      return {};
    }
    return mlir::tt::ttcore::DataType::Int32;
  }

  // All remaining binary ops.
  // Tracked in :
  // https://github.com/issues/created?issue=tenstorrent%7Ctt-metal%7C25112
  if (isa<ttnn::DivideOp, ttnn::PowTensorOp>(op)) {
    if (dType == mlir::tt::ttcore::DataType::Float32 ||
        dType == mlir::tt::ttcore::DataType::BFloat16 ||
        dType == mlir::tt::ttcore::DataType::BFP_BFloat8 ||
        dType == mlir::tt::ttcore::DataType::BFP_BFloat4) {
      return {};
    }
    return mlir::tt::ttcore::DataType::Float32;
  }

  return {};
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
// Input tensor must have ROW_MAJOR layout.
// No need for data type workaround for output tensor; only layout workaround is
// required to match original layout.
// tt-metal specs:
// https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.argmax.html
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createArgMaxOpOperandsWorkarounds() {
  wa::TTNNOperandWorkarounds rowMajorLayoutWorkaround;
  rowMajorLayoutWorkaround.tensorLayoutWorkaround = Layout::RowMajor;

  wa::TTNNOperandWorkarounds rowMajorLayoutUint32Workaround;
  rowMajorLayoutUint32Workaround.tensorLayoutWorkaround = Layout::RowMajor;
  rowMajorLayoutUint32Workaround.tensorDataTypeWorkaround =
      mlir::tt::ttcore::DataType::UInt32;

  return wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(rowMajorLayoutWorkaround)
      .addOutputOperandWorkaround(rowMajorLayoutUint32Workaround);
}

// Factory method to create a set of workarounds for Pad op operands.
// tt-metal only supports float32 and bfloat16 data types.
// tt-metal does not support front padding for tile layout.
// https://github.com/tenstorrent/tt-metal/issues/10987
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createPadOpOperandsWorkarounds(
    mlir::TypedValue<mlir::RankedTensorType> input,
    ttnn::TTNNLayoutAttr layoutAttr, llvm::ArrayRef<int32_t> padding) {
  TTNNOperandWorkarounds operandWorkaround;

  // Determine whether front padding is applied. For each dimension, padding is
  // specified as a tuple <front padding, back padding>, indicating the number
  // of elements added before and after the data.
  bool isFrontPadding =
      llvm::any_of(llvm::enumerate(padding), [](const auto &indexedValue) {
        const auto [index, value] = indexedValue;
        return index++ % 2 == 0 && value != 0;
      });

  if (isFrontPadding && layoutAttr.isTiled()) {
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

// Conv2d and ConvTranspose2d are memory intensive operations and until
// there is a proper solution to handle large inputs ttnn recommends to
// use row major layout for conv activations. In general case (when optimizer is
// disabled) we will apply row major layout workaround to activation. When
// optimizer is enabled this workaround is skipped because from what we observed
// tile layout for activations works for the models we tested up to now.
//
// There is another workaround decompositon for conv2d and conv2d transpose
// which is run regardless if optimizer is on or off.
// Purpose of that decomposition is to move weight and bias to host memory
// in row major layout and rewrite output of conv2d and conv2d transpose to
// tile layout.
// https://github.com/tenstorrent/tt-metal/issues/19762
template <typename T>
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConvOpOperandsWorkarounds(T op) {

  TTNNOperandWorkarounds inputWorkaround;
  inputWorkaround.tensorLayoutWorkaround = Layout::RowMajor;

  // If input dtype is BFP_BFloat8, we need to apply a dtype workaround
  // to bfloat16 before we convert to row major.
  RankedTensorType inputType = op.getInput().getType();
  ttcore::DataType inputDataType =
      mlir::tt::ttcore::elementTypeToDataType(inputType.getElementType());
  if (inputDataType == ttcore::DataType::BFP_BFloat8) {
    inputWorkaround.tensorDataTypeWorkaround = ttcore::DataType::BFloat16;
  }

  TTNNOperandWorkarounds emptyWorkaround;

  auto workaround =
      wa::TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
          .addInputOperandWorkaround(inputWorkaround)
          .addInputOperandWorkaround(emptyWorkaround)
          .addOutputOperandWorkaround(emptyWorkaround);

  if (op.getBias()) {
    workaround = workaround.addInputOperandWorkaround(emptyWorkaround);
  }

  return workaround;
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

// Factory method to create a set of workaround for SortOp operands.
// tt-metal generates indices of type UInt16. Any mismatch between generated and
// expected data type will cause runtime to assert.
// Issue page: https://github.com/tenstorrent/tt-mlir/issues/4405
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createSortOpOperandsWorkarounds(
    ttnn::SortOp op) {
  auto indicesElementType = op.getIndices().getType().getElementType();

  TTNNOperandWorkarounds datatypeWorkaround;
  if (!(indicesElementType.isInteger(16) &&
        indicesElementType.isUnsignedInteger())) {
    datatypeWorkaround.tensorDataTypeWorkaround = ttcore::DataType::UInt16;
  }

  // Empty workaround object for operands which do not require any changes.
  TTNNOperandWorkarounds operandWorkaround;

  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds()
      .addInputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(operandWorkaround)
      .addOutputOperandWorkaround(datatypeWorkaround);
}

template TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConvOpOperandsWorkarounds(
    ttnn::Conv2dOp op);
template TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createConvOpOperandsWorkarounds(
    ttnn::ConvTranspose2dOp op);

} // namespace mlir::tt::ttnn::wa
