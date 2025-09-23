// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreTraits.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsResources.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/VerificationUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include <cstdint>
#include <numeric>
#include <optional>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.cpp.inc"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

template <typename T>
static ::mlir::LogicalResult
foldConsecutiveDataCastOps(T op, ::mlir::PatternRewriter &rewriter) {
  // Fold two consecutive data type cast ops into a single one
  T previousDataCastOp = op.getInput().template getDefiningOp<T>();

  // If there is no previous data cast op, return failure.
  if (!previousDataCastOp) {
    return ::mlir::failure();
  }

  // Check if the previous cast op has only one use. We can only fold if the
  // previous op has single use.
  if (!previousDataCastOp->hasOneUse()) {
    return ::mlir::failure();
  }

  // Replace the previous op with the merged data type cast op.
  Value foldedTypecastOp = rewriter.replaceOpWithNewOp<T>(
      previousDataCastOp, op.getType(), previousDataCastOp.getInput(),
      op.getDtypeAttr());

  // Replace all uses of the current op with the merged TypecastOp.
  rewriter.replaceAllUsesWith(op, foldedTypecastOp);

  // Erase the current op.
  rewriter.eraseOp(op);

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// RandOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::RandOp::verify() {
  ttcore::DataType dtype = getDtype();
  ttcore::DataType outputType = mlir::tt::ttcore::elementTypeToDataType(
      getResult().getType().getElementType());

  if (dtype != outputType) {
    return emitOpError() << "dtype does not match with output tensor type";
  }

  float low = getLow().convertToFloat();
  float high = getHigh().convertToFloat();
  if (low >= high) {
    return emitOpError() << "'low' value must be < 'high' value";
  }

  if (!llvm::equal(getResult().getType().getShape(), getSize().getShape())) {
    return emitOpError()
           << "size argument does not match with output tensor shape. [Size = ("
           << getSize().getShape() << "), output tensor shape = ("
           << getResult().getType().getShape() << ")]";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::ConstantOp::verify() {
  if (!isa<DenseResourceElementsAttr, DenseElementsAttr>(getValue())) {
    return emitOpError("value attribute must be one of "
                       "DenseResourceElementsAttr or DenseElementsAttr.");
  }

  ::mlir::RankedTensorType outputType = getResult().getType();
  TTNNLayoutAttr outputLayout =
      mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());

  if (outputLayout.getBufferType() != BufferType::SystemMemory &&
      !getDevice()) {
    return emitOpError("device operand must be specified for non-system memory "
                       "buffer type");
  }

  if (outputLayout.getBufferType() == BufferType::SystemMemory && getDevice()) {
    return emitOpError("device operand must not be specified for system memory "
                       "buffer type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LogicalRightShiftOp
//===----------------------------------------------------------------------===//

// LogicalRightShiftOp verifier
::mlir::LogicalResult mlir::tt::ttnn::LogicalRightShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that all operands have integer element types.
  auto lhsElemType = lhsTensorType.getElementType();
  auto rhsElemType = rhsTensorType.getElementType();
  auto outputElemType = outputTensorType.getElementType();

  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  if (!mlir::isa<mlir::IntegerType>(outputElemType)) {
    return emitOpError() << "Output element type must be integer, but got "
                         << outputElemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LogicalLeftShiftOp
//===----------------------------------------------------------------------===//

// LogicalLeftShiftOp verifier
::mlir::LogicalResult mlir::tt::ttnn::LogicalLeftShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that all operands have integer element types.
  auto lhsElemType = lhsTensorType.getElementType();
  auto rhsElemType = rhsTensorType.getElementType();
  auto outputElemType = outputTensorType.getElementType();

  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  if (!mlir::isa<mlir::IntegerType>(outputElemType)) {
    return emitOpError() << "Output element type must be integer, but got "
                         << outputElemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::ClampScalarOp::verify() {
  const RankedTensorType inputTensorType = getInput().getType();

  const RankedTensorType outputTensorType = getResult().getType();

  if (inputTensorType.getShape() != outputTensorType.getShape()) {
    return emitOpError("input and output must have same shape.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrepareConv2dWeightsOp
//===----------------------------------------------------------------------===//

// PrepareConv2dWeightsOp verification
::mlir::LogicalResult mlir::tt::ttnn::PrepareConv2dWeightsOp::verify() {
  mlir::RankedTensorType weightType = getWeightTensor().getType();

  if (weightType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }

  if (getWeightsFormat() != "OIHW") {
    return emitOpError("Only `OIHW` weights format is currently supported");
  }

  constexpr unsigned int WEIGHT_OUT_CHANNEL_DIM = 0, WEIGHT_IN_CHANNEL_DIM = 1;
  constexpr unsigned int WEIGHT_KERNEL_HEIGHT_DIM = 2,
                         WEIGHT_KERNEL_WIDTH_DIM = 3;

  if (weightType.getShape()[WEIGHT_OUT_CHANNEL_DIM] != getOutChannels()) {
    return emitOpError()
           << "Expected output channels attribute (" << getOutChannels()
           << ") to match the first dimension of the weight tensor ("
           << weightType.getShape()[WEIGHT_OUT_CHANNEL_DIM] << ")";
  }

  if (weightType.getShape()[WEIGHT_IN_CHANNEL_DIM] !=
      getInChannels() / getGroups()) {
    return emitOpError()
           << "Expected input channels attribute (" << getInChannels()
           << ") to match the number of input channels per group ("
           << weightType.getShape()[WEIGHT_IN_CHANNEL_DIM] / getGroups() << ")";
  }

  if (getKernelSize().size() != 2) {
    return emitOpError("Expected kernel size attribute to be a 2D tensor");
  }

  if (weightType.getShape()[WEIGHT_KERNEL_HEIGHT_DIM] != getKernelSize()[0]) {
    return emitOpError()
           << "Expected kernel height attribute (" << getKernelSize()[0]
           << ") to match the third dimension of the weight tensor ("
           << weightType.getShape()[WEIGHT_KERNEL_HEIGHT_DIM] << ")";
  }

  if (weightType.getShape()[WEIGHT_KERNEL_WIDTH_DIM] != getKernelSize()[1]) {
    return emitOpError()
           << "Expected kernel width attribute (" << getKernelSize()[1]
           << ") to match the fourth dimension of the weight tensor ("
           << weightType.getShape()[WEIGHT_KERNEL_WIDTH_DIM] << ")";
  }

  if (getStride().size() != 2) {
    return emitOpError("Expected stride attribute to be a 2D tensor");
  }

  if (getDilation().size() != 2) {
    return emitOpError("Expected dilation attribute to be a 2D tensor");
  }

  if (getPadding().size() != 2 && getPadding().size() != 4) {
    return emitOpError("Expected padding attribute to be a 2D tensor");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp
//===----------------------------------------------------------------------===//

// PrepareConv2dBiasOp verification
::mlir::LogicalResult mlir::tt::ttnn::PrepareConv2dBiasOp::verify() {
  mlir::RankedTensorType biasType = getBiasTensor().getType();

  if (biasType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }

  constexpr unsigned int BIAS_OUT_CHANNEL_DIM = 3;
  if (biasType.getShape()[BIAS_OUT_CHANNEL_DIM] != getOutChannels()) {
    return emitOpError()
           << "Expected output channels attribute (" << getOutChannels()
           << ") to match the number of output channels in the bias tensor ("
           << biasType.getShape()[BIAS_OUT_CHANNEL_DIM] << ")";
  }

  if (getKernelSize().size() != 2) {
    return emitOpError("Kernel size attribute must have two values, got: " +
                       std::to_string(getKernelSize().size()));
  }

  if (getStride().size() != 2) {
    return emitOpError("Stride attribute must have two values, got: " +
                       std::to_string(getStride().size()));
  }

  if (getPadding().size() != 2 && getPadding().size() != 4) {
    return emitOpError("Padding attribute must have two or four values, got: " +
                       std::to_string(getPadding().size()));
  }

  if (getDilation().size() != 2) {
    return emitOpError("Dilation attribute must have two values, got: " +
                       std::to_string(getDilation().size()));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

template <typename OpType>
static bool isDefinedByOp(mlir::Value value) {
  if (value.getDefiningOp<OpType>()) {
    return true;
  }

  // Handle the case where we're inside a trace function
  // We need to check the original defining operation
  if (!mlir::isa<mlir::BlockArgument>(value)) {
    return false;
  }
  auto arg = mlir::cast<mlir::BlockArgument>(value);

  func::FuncOp funcOp =
      mlir::dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
  if (!funcOp || !utils::isTTNNTraceFunc(funcOp)) {
    return false;
  }

  size_t argIndex = arg.getArgNumber();
  auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp) {
    return false;
  }

  bool foundDefinedByOp = false;
  moduleOp->walk([&](ttnn::CaptureOrExecuteTraceOp op) {
    llvm::StringRef captureCalleeName = op.getCaptureCallee();
    if (!captureCalleeName.consume_front(g_TTNNCaptureTracePrefix)) {
      return WalkResult::advance();
    }
    if (captureCalleeName != funcOp.getSymName()) {
      return WalkResult::advance();
    }
    mlir::Value targetValue = op.getInputs()[argIndex];
    if (targetValue.getDefiningOp<OpType>()) {
      foundDefinedByOp = true;
    }
    return WalkResult::interrupt();
  });

  return foundDefinedByOp;
}

// Conv2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::Conv2dOp::verify() {
  using namespace mlir::tt::ttnn::utils::verification_utils::
      conv2d_verification;

  if (verifyTensorRanks(this).failed()) {
    return mlir::failure();
  }

  if (getConv2dConfig() && getConv2dConfig()->getDeallocateActivation() &&
      getConv2dConfig()->getDeallocateActivation().getValue()) {
    for (auto *user : getInput().getUsers()) {
      if (this->getOperation()->isBeforeInBlock(user)) {
        return emitOpError()
               << "Conv2dOp with `deallocate_activation` set to true "
                  "must be the last user of the input tensor. ";
      }
    }
  }

  auto expectedParams = getAndVerifyConv2dParams(this);
  if (auto error = expectedParams.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }
  Conv2dParams params = *expectedParams;

  if (!isDefinedByOp<mlir::tt::ttnn::PrepareConv2dWeightsOp>(getWeight()) &&
      !isDefinedByOp<mlir::tt::ttcore::LoadCachedOp>(getWeight())) {
    // Only check when the weight is not prepared because it changes the shape
    // and ordering of dims.
    if (getWeight().getType().getDimSize(WEIGHT_OUT_CHANNEL) !=
        getOutChannels()) {
      return emitOpError()
             << "Expected output channels attribute (" << getOutChannels()
             << ") to match the output channels in the weight tensor ("
             << getWeight().getType().getDimSize(WEIGHT_OUT_CHANNEL) << ").";
    }

    if (getWeight().getType().getDimSize(WEIGHT_IN_CHANNEL) !=
        getInChannels() / getGroups()) {
      return emitOpError()
             << "Expected input channels / groups attribute ("
             << getInChannels() << "/" << getGroups()
             << ") = " << getInChannels() / getGroups()
             << " to match the number of input channels per group in the "
                "weight tensor ("
             << getWeight().getType().getDimSize(WEIGHT_IN_CHANNEL) << ").";
    }

    if (getWeight().getType().getDimSize(WEIGHT_KERNEL_HEIGHT) !=
        params.kernelSize.vertical) {
      return emitOpError()
             << "Expected kernel height attribute ("
             << params.kernelSize.vertical
             << ") to match the kernel height in the weight tensor ("
             << getWeight().getType().getDimSize(WEIGHT_KERNEL_HEIGHT) << ").";
    }

    if (getWeight().getType().getDimSize(WEIGHT_KERNEL_WIDTH) !=
        params.kernelSize.horizontal) {
      return emitOpError()
             << "Expected kernel width attribute ("
             << params.kernelSize.horizontal
             << ") to match the kernel width in the weight tensor ("
             << getWeight().getType().getDimSize(WEIGHT_KERNEL_WIDTH) << ").";
    }
  }

  int64_t expectedInputFlattenSize =
      getBatchSize() * getInputHeight() * getInputWidth();
  if (expectedInputFlattenSize !=
      getInput().getType().getDimSize(FLATTENED_DIM)) {
    int64_t actualSize = getInput().getType().getDimSize(FLATTENED_DIM);
    return emitOpError() << "The input tensor's flattened dimension ("
                         << actualSize
                         << ") does not match the product of batch_size_attr * "
                            "input_height_attr * input_width_attr ("
                         << getBatchSize() << " * " << getInputHeight() << " * "
                         << getInputWidth() << " = " << expectedInputFlattenSize
                         << ").";
  }

  auto [inputDims, weightDims, biasDims] = getConv2dInputDims(this);
  OutputTensorDims outputDims = getConv2dOutputDims(this);

  if (verifyConv2dInputDims(this, inputDims, weightDims, biasDims, params)
          .failed()) {
    return mlir::failure();
  }

  if (verifyOutputDimensions(this, inputDims, weightDims, biasDims, outputDims,
                             params)
          .failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

// Get number of output channels.
int64_t mlir::tt::ttnn::Conv2dOp::getOutputChannelSize() {
  RankedTensorType weightTy = getWeight().getType();
  return weightTy.getShape()[0];
}

// Verify that bias dimensions are compatible with conv2d operation.
bool mlir::tt::ttnn::Conv2dOp::isBiasCompatible(llvm::ArrayRef<int64_t> bias) {
  return bias[0] == 1 && bias[1] == 1 && bias[2] == 1 &&
         bias[3] == getOutputChannelSize();
}

//===----------------------------------------------------------------------===//
// Quantize Ops
//===----------------------------------------------------------------------===//

// Helper function to verify that a zero point is within the range of the
// storage type.
static ::mlir::LogicalResult verifyZeroPointInRange(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
    int64_t zeroPoint, int64_t min, int64_t max, mlir::Type storageType) {
  if (zeroPoint < min || zeroPoint > max) {
    return emitOpError() << "Zero point " << zeroPoint
                         << " is out of the range for storage type "
                         << storageType;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult verifyQuantizeOpCommon(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
    ::mlir::RankedTensorType inputType, ::mlir::RankedTensorType outputType,
    std::optional<uint32_t> axis) {
  // Verify that the input rank matches the rank of the output tensor.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank of " << inputType.getRank()
                         << " does not match the output tensor rank of "
                         << outputType.getRank();
  }

  // Verify that the shapes of the input and output of a quantize operation are
  // the same.
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError() << "Output tensor shape ("
                         << ttmlir::utils::join(outputType.getShape(), ",") +
                                ") must match the input tensor shape: (" +
                                ttmlir::utils::join(inputType.getShape(), ",") +
                                ")";
  }

  // Verify that the axis, if provided, is within the bounds of the input tensor
  // rank.
  if (axis.has_value()) {
    int32_t axisValue = axis.value();
    if (axisValue < 0 || axisValue >= inputType.getRank()) {
      return emitOpError() << "Axis value " << axisValue
                           << " is out of the range [0, " << inputType.getRank()
                           << ") for the input tensor of rank "
                           << inputType.getRank();
    }
  }

  for (auto tensorType : {inputType, outputType}) {
    auto elemType = tensorType.getElementType();
    if (auto quantPerAxisType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                elemType)) {
      // Verify that the scales size matches the axis size for per-axis
      // quantization on both input and output types. This aligns with the
      // runtime's behavior.
      int64_t axis = quantPerAxisType.getQuantizedDimension();
      auto shape = tensorType.getShape();
      auto scales = quantPerAxisType.getScales();
      if (scales.size() != static_cast<size_t>(shape[axis])) {
        return emitOpError()
               << "Number of scales (" << scales.size()
               << ") does not match the size of the quantized axis ("
               << shape[axis] << ")";
      }
      // Verify that the zero point is in the range of the storage type.
      // This aligns with the frontends' behavior.
      llvm::ArrayRef<int64_t> zps = quantPerAxisType.getZeroPoints();
      int64_t min = quantPerAxisType.getStorageTypeMin();
      int64_t max = quantPerAxisType.getStorageTypeMax();
      for (int64_t zp : zps) {
        if (auto result = verifyZeroPointInRange(
                emitOpError, zp, min, max, quantPerAxisType.getStorageType());
            failed(result)) {
          return result;
        }
      }
    }
    if (auto quantType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
      // Verify that the zero point is in the range of the storage type
      // (per-tensor). This aligns with the frontends' behavior.
      int64_t zp = quantType.getZeroPoint();
      int64_t min = quantType.getStorageTypeMin();
      int64_t max = quantType.getStorageTypeMax();
      if (auto result = verifyZeroPointInRange(emitOpError, zp, min, max,
                                               quantType.getStorageType());
          failed(result)) {
        return result;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

// QuantizeOp verification.
::mlir::LogicalResult QuantizeOp::verify() {
  RankedTensorType inputTensorType = getInput().getType();
  RankedTensorType outputTensorType = getResult().getType();

  auto inputElemType = inputTensorType.getElementType();
  auto outputElemType = outputTensorType.getElementType();

  if (!mlir::isa<mlir::FloatType>(inputElemType)) {
    return emitOpError() << "Input element type must be float, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
    return emitOpError()
           << "Output element type must be UniformQuantizedType or "
              "UniformQuantizedPerAxisType, but got "
           << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                inputTensorType, outputTensorType, getAxis());
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

// DequantizeOp verification.
::mlir::LogicalResult DequantizeOp::verify() {
  RankedTensorType inputTensorType = getInput().getType();
  RankedTensorType outputTensorType = getResult().getType();

  auto inputElemType = inputTensorType.getElementType();
  auto outputElemType = outputTensorType.getElementType();

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
    return emitOpError() << "Input element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::FloatType>(outputElemType)) {
    return emitOpError() << "Output element type must be float, but got "
                         << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                inputTensorType, outputTensorType, getAxis());
}

//===----------------------------------------------------------------------===//
// RequantizeOp
//===----------------------------------------------------------------------===//

// RequantizeOp verification.
::mlir::LogicalResult RequantizeOp::verify() {
  const RankedTensorType inputTensorType = getInput().getType();
  const RankedTensorType outputTensorType = getResult().getType();

  auto inputElemType = inputTensorType.getElementType();
  auto outputElemType = outputTensorType.getElementType();

  auto inputIsPerAxis =
      mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
  auto outputIsPerAxis =
      mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);
  auto inputIsPerTensor =
      mlir::isa<mlir::quant::UniformQuantizedType>(inputElemType);
  auto outputIsPerTensor =
      mlir::isa<mlir::quant::UniformQuantizedType>(outputElemType);

  if (!((inputIsPerAxis && outputIsPerAxis) ||
        (inputIsPerTensor && outputIsPerTensor))) {
    return emitOpError()
           << "Input and output element types must both be per-axis "
              "or both be per-tensor quantized types, but got "
           << inputElemType << " and " << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                inputTensorType, outputTensorType, getAxis());
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

// ConvTranspose2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::ConvTranspose2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getResult().getType();
  std::optional<mlir::RankedTensorType> bias =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  if (inputType.getRank() != 4) {
    return emitOpError("Input must be a 4D tensor");
  }

  if (outputType.getRank() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  if (weightType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }

  if (bias.has_value()) {
    if (bias->getRank() != 4) {
      return emitOpError("Bias must be a 4D tensor");
    }
  }

  std::function<mlir::LogicalResult(llvm::ArrayRef<int32_t> &, const char *,
                                    int32_t)>
      checkBiggerThan = [&](llvm::ArrayRef<int32_t> &values, const char *name,
                            int32_t minValue) -> mlir::LogicalResult {
    for (int32_t value : values) {
      if (value < minValue) {
        return emitOpError() << "Attribute '" << name
                             << "' contains a value less than: " << minValue;
      }
    }
    return mlir::success();
  };

  uint32_t inChannels = getInChannels();
  if (inChannels != inputType.getDimSize(inputType.getRank() - 1)) {
    return emitOpError("Input channels attribute must match "
                       "the last dimension of the input tensor");
  }

  uint32_t outChannels = getOutChannels();
  if (outChannels != outputType.getDimSize(outputType.getRank() - 1)) {
    return emitOpError("Output channels attribute match "
                       "the last dimension of the output tensor");
  }

  // If The input shape is unflattened then verify the input shape.
  if (getBatchSize() * getInputHeight() * getInputWidth() !=
      inputType.getDimSize(2)) {

    uint32_t batchSize = getBatchSize();
    if (batchSize != inputType.getDimSize(0)) {
      return emitOpError("Batch size attribute must match the first "
                         "dimension of the input tensor");
    }

    uint32_t inputHeight = getInputHeight();
    if (inputHeight != inputType.getDimSize(inputType.getRank() - 3)) {
      return emitOpError("Input height attribute must match the second "
                         "dimension of the input tensor");
    }

    uint32_t inputWidth = getInputWidth();
    if (inputWidth != inputType.getDimSize(inputType.getRank() - 2)) {
      return emitOpError("Input width attribute must match the third "
                         "dimension of the input tensor");
    }
  }

  llvm::ArrayRef<int32_t> stride = getStride();
  if (failed(checkBiggerThan(stride, "stride", 1))) {
    return mlir::failure();
  }

  llvm::ArrayRef<int32_t> padding = getPadding();
  if (failed(checkBiggerThan(padding, "padding", 0))) {
    return mlir::failure();
  }

  llvm::ArrayRef<int32_t> outputPadding = getOutputPadding();
  if (failed(checkBiggerThan(outputPadding, "output padding", 0))) {
    return mlir::failure();
  }

  llvm::ArrayRef<int32_t> dilation = getDilation();
  if (failed(checkBiggerThan(dilation, "dilation", 1))) {
    return mlir::failure();
  }

  llvm::ArrayRef<std::int64_t> kernelShape = weightType.getShape();

  int32_t inputChannels = inputType.getDimSize(inputType.getRank() - 1);
  int32_t outputChannels = outputType.getDimSize(outputType.getRank() - 1);
  uint32_t groups = getGroups();

  if (inputChannels % groups != 0) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << inputChannels << " input channels and "
                         << groups << " groups.";
  }

  if (outputChannels % groups != 0) {
    return emitOpError() << "Number of output channels from output tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << outputChannels << " output channels and "
                         << groups << " groups.";
  }

  if (inputChannels != kernelShape[0]) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "match the first dimension of the weight tensor. "
                         << "Got " << inputChannels << " input channels and "
                         << kernelShape[0] << " in the weight tensor.";
  }

  if (outputChannels / groups != kernelShape[1]) {
    return emitOpError() << "Number of output channels per group must match "
                            "the second dimension of the weight tensor. "
                         << "Got " << (outputChannels / groups)
                         << " output channels per group and " << kernelShape[1]
                         << " in the weight tensor.";
  }

  if (bias) {
    if (bias->getDimSize(bias->getRank() - 1) != outputChannels) {
      return emitOpError() << "Mismatch in bias tensor dimensions. "
                           << "Bias tensor has "
                           << bias->getDimSize(bias->getRank() - 1)
                           << " channels, "
                           << "but the output tensor has " << outputChannels
                           << " channels.";
    }
  }

  int32_t kernelHeight = kernelShape[2];
  int32_t kernelWidth = kernelShape[3];

  int32_t Hin = getInputHeight();
  int32_t Win = getInputWidth();

  int32_t expectedHOut = (Hin - 1) * stride[0] - 2 * padding[0] +
                         dilation[0] * (kernelHeight - 1) + outputPadding[0] +
                         1;
  int32_t expectedWOut = (Win - 1) * stride[1] - 2 * padding[1] +
                         dilation[1] * (kernelWidth - 1) + outputPadding[1] + 1;
  if (expectedHOut < 0 || expectedWOut < 0) {
    return emitOpError() << "Given input size per channel: (" << Hin << " x "
                         << Win << "). "
                         << "Calculated output size per channel: ("
                         << expectedHOut << " x " << expectedWOut << "). "
                         << "Output size is too small";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Typecast Op
//===----------------------------------------------------------------------===//

// Typecast Op verification
::mlir::LogicalResult mlir::tt::ttnn::TypecastOp::verify() {
  ::mlir::RankedTensorType outputType = getResult().getType();
  TTNNLayoutAttr outputLayout =
      mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());

  if (getDtype() != outputLayout.getDataType()) {
    return emitOpError() << "Output tensor data type "
                         << DataTypeEnumToString(outputLayout.getDataType())
                         << " must match the data type of dtype attribute "
                         << DataTypeEnumToString(getDtype()) << ".";
  }

  return success();
}

// TypecastOp folder
::mlir::OpFoldResult mlir::tt::ttnn::TypecastOp::fold(FoldAdaptor adaptor) {

  // If the input and output are same, fold to the input.
  if (getType() == getInput().getType()) {
    return getInput();
  }

  return nullptr;
}

// Typecast canonicalizer method
::llvm::LogicalResult
mlir::tt::ttnn::TypecastOp::canonicalize(TypecastOp typecastOp,
                                         ::mlir::PatternRewriter &rewriter) {
  return foldConsecutiveDataCastOps(typecastOp, rewriter);
}

//===----------------------------------------------------------------------===//
// ToDTypeOp
//===----------------------------------------------------------------------===//

// ToDTypeOp verification
::mlir::LogicalResult mlir::tt::ttnn::ToDTypeOp::verify() {
  ::mlir::RankedTensorType outputType = getResult().getType();
  TTNNLayoutAttr outputLayout =
      mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());

  if (getDtype() != outputLayout.getDataType()) {
    return emitOpError() << "Output tensor data type "
                         << DataTypeEnumToString(outputLayout.getDataType())
                         << " must match the data type of dtype attribute "
                         << DataTypeEnumToString(getDtype()) << ".";
  }

  return success();
}

// ToDTypeOp folder
::mlir::OpFoldResult mlir::tt::ttnn::ToDTypeOp::fold(FoldAdaptor adaptor) {

  // If the input and output are same, fold to the input.
  if (getType() == getInput().getType()) {
    return getInput();
  }

  return nullptr;
}

// ToDTypeOp canonicalizer method
::llvm::LogicalResult
mlir::tt::ttnn::ToDTypeOp::canonicalize(ToDTypeOp op,
                                        ::mlir::PatternRewriter &rewriter) {
  // NOLINTNEXTLINE
  return foldConsecutiveDataCastOps(op, rewriter);
}

//===----------------------------------------------------------------------===//
// Common verifier for 2d pooling ops
//===----------------------------------------------------------------------===//
static mlir::LogicalResult
verifyPoolingOp(llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
                mlir::RankedTensorType inputType,
                llvm::ArrayRef<int32_t> kernelSize, int32_t inputHeight,
                int32_t inputWidth, int32_t batchSize, int32_t channels,
                llvm::StringRef opName) {
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();

  if (kernelSize[0] > inputHeight) {
    return emitOpError() << "Kernel height " << kernelSize[0]
                         << " is greater than input height " << inputHeight
                         << ". This " << opName << " configuration is invalid.";
  }

  if (kernelSize[1] > inputWidth) {
    return emitOpError() << "Kernel width " << kernelSize[1]
                         << " is greater than input width " << inputWidth
                         << ". This " << opName << " configuration is invalid.";
  }

  if (inputType.getRank() != 4) {
    return emitOpError()
           << "Input tensor rank must be 4. Received input with rank "
           << inputType.getRank() << ". Shape: (" << inputShape << ").";
  }

  if (inputShape[0] != 1 || inputShape[1] != 1) {
    return emitOpError() << opName
                         << " input must be in the form (1, 1, N*H*W, "
                            "C). Received shape ("
                         << inputShape << ").";
  }

  if (inputShape[2] != batchSize * inputHeight * inputWidth) {
    return emitOpError() << opName << " shape (" << inputShape
                         << ") at dim -2 must be equal to N*H*W. However the "
                            "attributes given are N="
                         << batchSize << ", H=" << inputHeight
                         << ", W=" << inputWidth << ". " << batchSize << "*"
                         << inputHeight << "*" << inputWidth
                         << " != " << inputShape[2] << ".";
  }

  if (inputShape[3] != channels) {
    return emitOpError() << opName << " shape (" << inputShape
                         << ") at dim -3 must be equal to C. However the "
                            "attribute given is C="
                         << channels << ". " << inputShape[3]
                         << " != " << channels;
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AvgPool2dOp
//===----------------------------------------------------------------------===//

// AvgPool2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::AvgPool2dOp::verify() {
  return verifyPoolingOp([&]() { return emitOpError(); }, getInput().getType(),
                         getKernelSize(), getInputHeight(), getInputWidth(),
                         getBatchSize(), getChannels(), getOperationName());
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

// MaxPool2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::MaxPool2dOp::verify() {
  return verifyPoolingOp([&]() { return emitOpError(); }, getInput().getType(),
                         getKernelSize(), getInputHeight(), getInputWidth(),
                         getBatchSize(), getChannels(), getOperationName());
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::ArangeOp::verify() {

  if (getStep() == 0) {
    return emitOpError("Step cannot be zero.");
  }

  int64_t numValues = (getEnd() - getStart()) / getStep();

  if (numValues <= 0) {
    return emitOpError("Invalid range: start=")
           << getStart() << ", end=" << getEnd() << ", step=" << getStep();
  }

  std::vector<int64_t> expectedShape = {numValues};
  if (getType().getShape().vec() != expectedShape) {
    return emitOpError() << "Output tensor shape must be " << expectedShape
                         << ", but got " << getType().getShape();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NamedFullOp
//===----------------------------------------------------------------------===//

template <typename Op>
static ::mlir::LogicalResult namedOpVerify(Op op) {
  RankedTensorType output = op.getResult().getType();
  if (op.getDtype()) {
    if (op.getDtype() !=
        ttcore::elementTypeToDataType(output.getElementType())) {
      return op.emitOpError("Data type mismatch between op and output tensor.");
    }
  }

  ArrayRef<int64_t> shape = op.getShape().getShape();
  ArrayRef<int64_t> outputShape = output.getShape();

  if (shape != outputShape) {
    return op.emitOpError("Output tensor shape must be ")
           << shape << ", but got " << outputShape;
  }

  return success();
}

::mlir::LogicalResult mlir::tt::ttnn::ZerosOp::verify() {
  return namedOpVerify(*this);
}

::mlir::LogicalResult mlir::tt::ttnn::OnesOp::verify() {
  return namedOpVerify(*this);
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

void mlir::tt::ttnn::FullOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state,
                                   mlir::Type resultType,
                                   mlir::Attribute fillValue,
                                   mlir::Value device) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(resultType);
  ttnn::TTNNLayoutAttr layoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

  ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(ctx, tensorType.getShape());
  ttcore::DataTypeAttr dtypeAttr =
      ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
  ttnn::LayoutAttr tensorLayoutAttr =
      ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());

  build(builder, state, resultType, device, shapeAttr, fillValue, dtypeAttr,
        tensorLayoutAttr, /*memory_config=*/nullptr);
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

// EmptyOp verification
::mlir::LogicalResult mlir::tt::ttnn::EmptyOp::verify() {
  // Check that the attributes of the op match the attributes of the output
  // tensor type.
  //
  RankedTensorType output = getResult().getType();

  TTNNLayoutAttr layoutAttr = mlir::cast<TTNNLayoutAttr>(output.getEncoding());

  // Shape
  //
  if (output.getShape() != getShape().getShape()) {
    return emitOpError() << "Output tensor shape must be "
                         << getShape().getShape() << ", but got "
                         << output.getShape();
  }

  // DataType and Layout
  //
  if (getLayout() != layoutAttr.getLayout()) {
    return emitOpError("Layout mismatch between op and layoutAttr.");
  }
  if (getDtype() != layoutAttr.getDataType()) {
    return emitOpError("Data type mismatch between op and layoutAttr.");
  }

  // MemoryConfig
  // Compare internal attrs with output tensor attrs.
  //
  if (getMemoryConfig().getBufferType().getValue() !=
      layoutAttr.getBufferType()) {
    return emitOpError("Buffer type mismatch between op and layoutAttr.");
  }
  if (getMemoryConfig().getTensorMemoryLayout() != layoutAttr.getMemLayout()) {
    return emitOpError(
        "Tensor memory layout mismatch between op and layoutAttr.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// ConcatOp verification
::mlir::LogicalResult mlir::tt::ttnn::ConcatOp::verify() {
  mlir::OperandRange inputs = getInputs();
  int32_t dim = getDim();
  mlir::RankedTensorType firstTensor =
      mlir::cast<mlir::RankedTensorType>(inputs.front().getType());
  int64_t firstTensorRank = firstTensor.getRank();

  if (dim < 0) {
    dim += firstTensorRank;
  }

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= firstTensor.getRank()) {
    return emitOpError() << "Invalid dimension " << getDim()
                         << " for concatenation.";
  }

  // Get the rank of the first input tensor
  // and check that all input tensors have the same rank
  // and that all dimensions except `dim` are the same.
  for (auto input : inputs.drop_front()) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    // Check if all inputs have the same rank.
    if (inputType.getRank() != firstTensorRank) {
      return emitOpError("All input tensors must have the same rank.");
    }

    // Check that dimensions (except `dim`) are the same.
    for (int64_t i = 0; i < firstTensorRank; ++i) {
      if (i != dim && inputType.getDimSize(i) != firstTensor.getDimSize(i)) {
        return emitOpError() << "All input tensors must have the same "
                                "dimensions, except for dimension "
                             << dim << ".";
      }
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

// RepeatOp verification
::mlir::LogicalResult mlir::tt::ttnn::RepeatOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  llvm::ArrayRef<int64_t> repeatDims = getRepeatDims().getShape();

  // Verify that the input tensor and repeat_dims argument have same rank.
  if (inputType.getRank() != static_cast<int64_t>(repeatDims.size())) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the number of repeat dimensions "
                         << repeatDims.size() << ".";
  }

  // Verify that the input and output tensor have same rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the output tensor rank "
                         << outputType.getRank() << ".";
  }

  // Verify expected output shape.
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  for (size_t i = 0; i < repeatDims.size(); i++) {
    // Verify that the repeat dimension is greater than 0.
    if (repeatDims[i] <= 0) {
      return emitOpError() << "Repeat dimension at index " << i
                           << " must be greater than 0.";
    }

    int64_t dimValue = repeatDims[i];
    if (inputShape[i] * dimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",")
                           << ") at index " << i
                           << " does not repeat to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using repeat value " << dimValue << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// ReshapeOp verification
::mlir::LogicalResult mlir::tt::ttnn::ReshapeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  auto shape = getShape();
  int64_t shapeSize = static_cast<int64_t>(shape.size());

  // Check that the shape size matches the rank of the output tensor.
  if (shapeSize != static_cast<int64_t>(outputType.getRank())) {
    return emitOpError() << "Shape attribute size " << shapeSize
                         << " must match output tensor rank "
                         << outputType.getRank();
  }

  // Cardinality of the input and output tensors must be the same.
  if (inputType.getNumElements() != outputType.getNumElements()) {
    return emitOpError() << "Input tensor number of elements "
                         << inputType.getNumElements()
                         << " and output tensor number of elements "
                         << outputType.getNumElements() << " must be the same";
  }

  bool hasNegative = false;
  auto outputShape = outputType.getShape();

  // Check that all dimensions are positive except for at most one -1
  // Check that the non-negative dimensions match the output tensor shape
  // Calculate the product of the known dimensions
  for (int64_t i = 0; i < shapeSize; i++) {
    int64_t dimValue = mlir::cast<IntegerAttr>(shape[i]).getInt();

    if (dimValue == -1) {
      if (hasNegative) {
        return emitOpError("Shape attribute must have at most one -1 element");
      }
      hasNegative = true;
    } else {
      if (dimValue <= 0) {
        return emitOpError(
            "All dimensions must be positive except the one with -1");
      }

      // Ensure that the non-negative dimensions match the output tensor shape
      if (dimValue != outputShape[i]) {
        return emitOpError()
               << "Shape attribute " << dimValue
               << " must match the output tensor shape " << outputShape[i]
               << " at index " << i << " for dimension that is not -1";
      }
    }
  }

  return success();
}

// Fold the operation if the type of the input and output types are the same.
static mlir::OpFoldResult foldIdentityReshape(mlir::tt::ttnn::ReshapeOp op) {
  if (op.getType() == op.getInput().getType()) {
    return op.getInput();
  }
  return nullptr;
}

// Back to back reshapes can be replaced with the final reshape.
static mlir::OpFoldResult foldConsecutiveReshape(mlir::tt::ttnn::ReshapeOp op) {
  if (auto reshapeOperand =
          op.getInput().getDefiningOp<mlir::tt::ttnn::ReshapeOp>()) {
    op.getOperation()->setOperand(0, reshapeOperand.getInput());
    return op.getResult();
  }
  return nullptr;
}

// ReshapeOp folder
::mlir::OpFoldResult mlir::tt::ttnn::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityReshape(*this)) {
    return foldResult;
  }
  if (auto foldResult = foldConsecutiveReshape(*this)) {
    return foldResult;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// PadOp verification
::mlir::LogicalResult mlir::tt::ttnn::PadOp::verify() {

  ::mlir::RankedTensorType inputType = getInput().getType();

  // Check that size of padding is correct
  if (static_cast<int64_t>(getPadding().size()) != 2 * inputType.getRank()) {
    return emitOpError("Padding must have the same number of elements as twice "
                       "the rank of the input tensor");
  }

  std::vector<int64_t> inferredShapeVec = inputType.getShape().vec();
  llvm::ArrayRef<int32_t> padding = getPadding();
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    inferredShapeVec[i] += padding[2 * i];
    inferredShapeVec[i] += padding[2 * i + 1];
  }
  llvm::ArrayRef<int64_t> inferredShape = inferredShapeVec;

  // Check that the output tensor shape is correct
  ::mlir::RankedTensorType resultType = getResult().getType();
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape != inferredShape) {
    return emitOpError("Output tensor shape (" +
                       ttmlir::utils::join(resultShape, ",") +
                       ") must match the inferred shape: (" +
                       ttmlir::utils::join(inferredShape, ",") + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SliceStaticOp
//===----------------------------------------------------------------------===//

// SliceStaticOp verification
::mlir::LogicalResult mlir::tt::ttnn::SliceStaticOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::ArrayAttr begins = getBeginsAttr();
  ::mlir::ArrayAttr ends = getEndsAttr();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Verify that the input is at least 1D tensor
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Verify that the input rank matches number of elements in begins, ends, and
  // step
  size_t input_rank = static_cast<size_t>(inputType.getRank());
  if (input_rank != begins.size() || input_rank != ends.size() ||
      input_rank != stepAttr.size()) {
    return emitOpError("Begins, ends, and step attributes must have the same "
                       "number of elements as the input tensor rank");
  }

  // Validate that the output tensor has the same element type as the input
  // tensor
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError(
        "Output tensor must have the same element type as the input tensor");
  }

  // Verify the output tensor rank
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError(
        "Output tensor must have the same rank as the input tensor");
  }

  // Verify begin, end, step and the output tensor dimensions
  for (size_t i = 0; i < input_rank; ++i) {
    int64_t dimSize = inputShape[i];

    int32_t begin = ::mlir::cast<::mlir::IntegerAttr>(begins[i]).getInt();
    int32_t end = ::mlir::cast<::mlir::IntegerAttr>(ends[i]).getInt();
    int32_t step = ::mlir::cast<::mlir::IntegerAttr>(stepAttr[i]).getInt();

    // Adjust negative begin and end
    int32_t adjustedBegin = (begin < 0) ? (begin + dimSize) : begin;
    int32_t adjustedEnd = (end < 0) ? (end + dimSize) : end;

    std::ostringstream inputShapeStream;
    inputShapeStream << "(";
    for (size_t i = 0; i < inputShape.size(); ++i) {
      inputShapeStream << inputShape[i];
      if (i != inputShape.size() - 1) {
        inputShapeStream << ", ";
      }
    }
    inputShapeStream << ")";
    std::string inputShapeStr = inputShapeStream.str();
    bool isEmptySliceOp = adjustedEnd == adjustedBegin;

    if (!isEmptySliceOp && (adjustedBegin < 0 || adjustedBegin >= dimSize)) {
      return emitOpError() << "Invalid begin index for dimension "
                           << std::to_string(i) << ". Expected value in range ["
                           << std::to_string(-dimSize) << ", " << dimSize
                           << "), got " << begin
                           << ". Input shape: " << inputShapeStr;
    }
    if (!isEmptySliceOp && (adjustedEnd < 0 || adjustedEnd > dimSize)) {
      return emitOpError() << "Invalid end index for dimension "
                           << std::to_string(i) << ". Expected value in range ["
                           << std::to_string(-dimSize) << ", " << dimSize
                           << "], got " << end
                           << ". Input shape: " << inputShapeStr;
    }

    auto formatValueMessage = [](int value, int adjustedValue) {
      return value < 0 ? std::to_string(adjustedValue) + " (" +
                             std::to_string(value) + ")"
                       : std::to_string(value);
    };
    std::string beginValueMessage = formatValueMessage(begin, adjustedBegin);
    std::string endValueMessage = formatValueMessage(end, adjustedEnd);

    if (step == 0) {
      return emitOpError("Step value for dimension " + std::to_string(i) +
                         " cannot be zero");
    }

    if (step > 0 && adjustedBegin > adjustedEnd) {
      return emitOpError() << "For positive step, begin index must be less "
                              "than or equal to end index for dimension "
                           << i << ". Got begin: " << beginValueMessage
                           << ", end: " << endValueMessage << ", step: " << step
                           << ", input shape: " << inputShapeStr;
    }

    if (step < 0 && adjustedBegin < adjustedEnd) {
      return emitOpError() << "For negative step, begin index must be greater "
                              "than or equal to end index for dimension "
                           << i << ". Got begin: " << beginValueMessage
                           << ", end: " << endValueMessage << ", step: " << step
                           << ", input shape: " << inputShapeStr;
    }

    // Calculate the expected size of the output dimension
    int32_t expectedDimSize =
        (std::abs(adjustedEnd - adjustedBegin) + std::abs(step) - 1) /
        std::abs(step);
    if (outputType.getDimSize(i) != expectedDimSize) {
      return emitOpError() << "Mismatch in dimension " << std::to_string(i)
                           << " of the output tensor: expected size "
                           << expectedDimSize << ", but got "
                           << outputType.getDimSize(i);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SliceDynamicOp
//===----------------------------------------------------------------------===//

// SliceDynamicOp verification
::mlir::LogicalResult mlir::tt::ttnn::SliceDynamicOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType beginsType = getBegins().getType();
  ::llvm::ArrayRef<int64_t> beginsShape = beginsType.getShape();
  ::mlir::RankedTensorType endsType = getEnds().getType();
  ::llvm::ArrayRef<int64_t> endsShape = endsType.getShape();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Verify that the input is at least 1D tensor.
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Verify that begins and ends are 1D tensors.
  size_t begins_rank = static_cast<size_t>(beginsType.getRank());
  size_t ends_rank = static_cast<size_t>(endsType.getRank());
  if (begins_rank != 1 || ends_rank != 1) {
    return emitOpError("Begins and ends must be 1D tensors");
  }

  // Verify that the input rank matches number of elements in begins, ends, and
  // step.
  auto input_rank = inputType.getRank();

  if (input_rank != beginsShape[0] || input_rank != endsShape[0] ||
      (stepAttr && static_cast<size_t>(input_rank) != stepAttr.size())) {
    return emitOpError("Begins, ends, and step must have the same "
                       "number of elements as the input tensor rank");
  }

  // Validate that the output tensor has the same element type as the input
  // tensor.
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError(
        "Output tensor must have the same element type as the input tensor");
  }

  // Verify the output tensor rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError(
        "Output tensor must have the same rank as the input tensor");
  }

  if (stepAttr) {
    // Verify that step isn't zero for any dimension.
    for (auto i = 0; i < input_rank; ++i) {
      int32_t step = ::mlir::cast<::mlir::IntegerAttr>(stepAttr[i]).getInt();
      if (step == 0) {
        return emitOpError("Step value for dimension " + std::to_string(i) +
                           " cannot be zero");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// TransposeOp verification
::mlir::LogicalResult mlir::tt::ttnn::TransposeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int32_t dim0 = getDim0();
  int32_t dim1 = getDim1();
  if (inputType.getRank() < 2) {
    return emitOpError("Input must be at least a 2D tensor");
  }
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError("Input must have the same rank as output");
  }
  if (dim0 >= inputType.getRank() || dim0 < -inputType.getRank()) {
    return emitOpError(
        "Dimension 0 attribute must be within the bounds of the input tensor");
  }
  if (dim1 >= inputType.getRank() || dim1 < -inputType.getRank()) {
    return emitOpError(
        "Dimension 1 attribute must be within the bounds of the input tensor");
  }
  if (dim0 < 0) {
    dim0 += inputType.getRank();
  }
  if (dim1 < 0) {
    dim1 += inputType.getRank();
  }
  if (outputShape[dim0] != inputShape[dim1] ||
      outputShape[dim1] != inputShape[dim0]) {
    return emitOpError("Input-output transpose dimension mismatch.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

// EmbeddingOp verification
::mlir::LogicalResult mlir::tt::ttnn::EmbeddingOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Intput tensor must be at most 2D tensor.
  if (inputType.getRank() > 2) {
    return emitOpError("input must be at most a 2D tensor, got ")
           << inputType.getRank() << "D ranked tensor";
  }

  // Weight tensor must be effectively 2D tensor. It means that it must have
  // shape of (1, 1,..., 1, N, M) where N is the dictionary size and M is the
  // embedding size.
  if (weightType.getRank() < 2) {
    return emitOpError("weight must be at least 2D tensor, got ")
           << weightType.getRank() << "D ranked tensor";
  }
  if (std::any_of(weightType.getShape().begin(),
                  weightType.getShape().end() - 2,
                  [](int64_t dim) { return dim != 1; })) {
    return emitOpError("weight must be effectively 2D tensor");
  }

  // Output tensor is expected to have the shape of (*inputTensorShape,
  // embeddingSize).
  int64_t embeddingSize = weightType.getDimSize(weightType.getRank() - 1);
  llvm::SmallVector<int64_t, 3> expectedOutputShape(inputType.getShape());
  expectedOutputShape.push_back(embeddingSize);

  if (!llvm::equal(expectedOutputShape, outputType.getShape())) {
    return emitOpError() << "expected output shape of (" << expectedOutputShape
                         << ") but got (" << outputType.getShape() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

// EmbeddingBackwardOp verification
::mlir::LogicalResult mlir::tt::ttnn::EmbeddingBackwardOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType inputGradType = getInGradient().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // inputType checks:
  // 1. Last dimension must be divisible by TILE_WIDTH.
  if (inputType.getShape().back() % TILE_WIDTH != 0) {
    return emitOpError("Input's last dim must be divisible by TILE_WIDTH");
  }

  // weightType must have rank of 2: (dictionary_size, embedding_size).
  if (weightType.getRank() != 2) {
    return emitOpError("Input must be a 2D tensor");
  }

  // inputGradType checks:
  // 1. inputGradType should have rank of 4, first 2 dimensions must be equal
  //    to 1, third dimension should match the volume of inputType, and the
  //    fourth dimension should match the second dimension of weightType.
  // 2. inputGradType must be of type bfloat16 or bfloat8.
  // 3. inputGradType and outputType must have the same dtype.
  if (inputGradType.getRank() != 4) {
    return emitOpError("Input gradient must be a 4D tensor");
  }
  if (inputGradType.getDimSize(0) != 1 || inputGradType.getDimSize(1) != 1) {
    return emitOpError("Input gradient must be in the form (1, 1, R, C)");
  }

  int64_t inputTypeVolume = 1;
  for (int64_t dim : inputType.getShape()) {
    inputTypeVolume *= dim;
  }
  if (inputGradType.getDimSize(2) != inputTypeVolume) {
    return emitOpError("Input gradient first dimension must match the volume "
                       "of the input tensor");
  }
  if (inputGradType.getDimSize(3) != weightType.getDimSize(1)) {
    return emitOpError("Input gradient second dimension must match the second "
                       "dimension of the weight tensor");
  }
  if (inputGradType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input gradient and output must have the same dtype");
  }

  // outputType should have the same shape as weightType.
  if (outputType.getShape() != weightType.getShape()) {
    return emitOpError("Output must have the same shape as weight");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp
//===----------------------------------------------------------------------===//

// Utility methods
static bool isValidDeviceLayout(TensorMemoryLayoutAttr memLayoutAttr) {
  return memLayoutAttr &&
         (memLayoutAttr.getValue() == TensorMemoryLayout::Interleaved ||
          isShardedMemoryLayout(memLayoutAttr.getValue()));
}

// ToMemoryConfigOp verification
::mlir::LogicalResult mlir::tt::ttnn::ToMemoryConfigOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getResult().getType();
  auto inputLayout =
      mlir::dyn_cast_if_present<TTNNLayoutAttr>(inputTy.getEncoding());
  auto outputLayout =
      mlir::dyn_cast_if_present<TTNNLayoutAttr>(outputTy.getEncoding());
  if (not inputLayout) {
    return emitOpError("Input tensor type missing layout attribute");
  }
  if (not outputLayout) {
    return emitOpError("Output tensor type missing layout attribute");
  }
  BufferType outputBufferType = outputLayout.getBufferType();
  TensorMemoryLayoutAttr outputMemoryLayout = outputLayout.getMemLayout();

  if (isDeviceBufferType(outputBufferType) &&
      !isValidDeviceLayout(outputMemoryLayout)) {
    return emitOpError("Device memory space only supports interleaved or "
                       "sharded memory layouts");
  }

  if (outputBufferType == BufferType::DRAM &&
      outputMemoryLayout.getValue() != TensorMemoryLayout::Interleaved) {
    return emitOpError(
        "Device DRAM memory space only supports interleaved memory layout");
  }

  if (outputLayout.hasShardedTensorMemoryLayout()) {
    if (not outputLayout.hasShardedL1TensorMemoryLayout()) {
      return emitOpError("Sharded tensors layout must reside in L1");
    }
    ::llvm::SmallVector<int64_t> shardShape = outputLayout.getShardShape();
    // Currently TTNN backend only supports 2D shard shape
    if (shardShape.size() != 2) {
      return emitOpError("Shard shape must be 2D");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

namespace {
// ToLayoutOp can be folded if its input has the same layout as the output of
// ToLayoutOp.
mlir::OpFoldResult foldIdentityToLayoutOp(ttnn::ToLayoutOp op) {
  mlir::RankedTensorType inputType = op.getInput().getType();
  ttnn::TTNNLayoutAttr inputLayout =
      mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  // Verify if input tensor has layout attribute.
  if (!inputLayout) {
    return nullptr;
  }

  mlir::RankedTensorType outputType = op.getType();
  ttnn::TTNNLayoutAttr outputLayout =
      mlir::dyn_cast<TTNNLayoutAttr>(outputType.getEncoding());
  // Verify if the output tensor has layout attribute.
  if (!outputLayout) {
    return nullptr;
  }

  return inputLayout == outputLayout ? op.getInput() : nullptr;
}

// Two consecutive ToLayoutOps can be merged together in the following way:
// df - data format, l - layout, ms - memory
// space, tml - tensor memory layout
//
//                |
//      -----------------------
//      |     ToLayoutOp      |                     |
//      | df1, l1, ms1, tml1  |          -----------------------
//      -----------------------          |     ToLayoutOp      |
//                |                 -->  | df2, l1, ms2, tml1  |
//                |                      -----------------------
//      -----------------------                     |
//      |     ToLayoutOp      |
//      |      df2, ms2       |
//      -----------------------
//                |
//
mlir::OpFoldResult foldConsecutiveToLayoutOp(ttnn::ToLayoutOp op) {
  // Get the input operand and verify that the previous op is ToLayoutOp.
  ttnn::ToLayoutOp producerOp = op.getInput().getDefiningOp<ttnn::ToLayoutOp>();

  if (!producerOp) {
    return nullptr;
  }

  if (!op.getDtype()) {
    op.setDtypeAttr(producerOp.getDtypeAttr());
  }
  if (!op.getMemoryConfig()) {
    op.setMemoryConfigAttr(producerOp.getMemoryConfigAttr());
  }
  op.getInputMutable().set(producerOp.getInput());

  return op.getResult();
}
} // namespace

// ToLayoutOp folder
mlir::OpFoldResult ttnn::ToLayoutOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityToLayoutOp(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldConsecutiveToLayoutOp(*this)) {
    return foldResult;
  }

  return nullptr;
}

// ToLayoutOp canonicalization.
void mlir::tt::ttnn::ToLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Merge to layout op into TTNN creation ops.
  patterns.add(+[](mlir::tt::ttnn::ToLayoutOp toLayoutOp,
                   mlir::PatternRewriter &rewriter) {
    Operation *creationOp = toLayoutOp.getInput().getDefiningOp();
    if (!creationOp ||
        !creationOp
             ->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>()) {
      return failure();
    }

    // Verify that the creation op has a single use.
    if (!creationOp->hasOneUse()) {
      return failure();
    }

    // Check that the creation op has a single result.
    if (creationOp->getNumResults() != 1) {
      return failure();
    }

    auto ttnnLayoutAttr =
        mlir::dyn_cast<TTNNLayoutAttr>(toLayoutOp.getType().getEncoding());
    if (!ttnnLayoutAttr) {
      return failure();
    }

    ttcore::DataTypeAttr targetDataTypeAttr = toLayoutOp.getDtypeAttr();
    LayoutAttr targetLayoutAttr = toLayoutOp.getLayoutAttr();
    MemoryConfigAttr targetMemoryConfigAttr = toLayoutOp.getMemoryConfigAttr();

    // If the to layout op tends to move the tensor to host, we can't merge it
    // into creation op if creation op doesn't support execution on host. For
    // example Rand and Empty op can only work on device.
    if (!creationOp->hasTrait<CanExecuteOnHostTrait>() &&
        (!targetMemoryConfigAttr ||
         isSystemBufferType(
             targetMemoryConfigAttr.getBufferType().getValue()))) {
      return failure();
    }

    auto tensorSpecOp = mlir::cast<TTNNTensorSpecInterface>(creationOp);

    rewriter.startOpModification(tensorSpecOp);

    tensorSpecOp.setDtypeAttr(targetDataTypeAttr);
    tensorSpecOp.setLayoutAttr(targetLayoutAttr);
    tensorSpecOp.setMemoryConfigAttr(targetMemoryConfigAttr);

    BufferTypeAttr newBufferType = nullptr;
    if (tensorSpecOp.getMemoryConfigAttr()) {
      newBufferType = tensorSpecOp.getMemoryConfigAttr().getBufferType();
    }

    TTNNDeviceOperandInterface deviceOperandInterface =
        mlir::cast<TTNNDeviceOperandInterface>(creationOp);
    // If the new buffer type is a device buffer type, we need to insert a
    // device operand.
    if (!deviceOperandInterface.getDevice() && newBufferType &&
        isDeviceBufferType(newBufferType.getValue())) {
      deviceOperandInterface.setDevice(
          utils::getOrInsertDevice(rewriter, toLayoutOp));
    } else if (deviceOperandInterface.getDevice() && newBufferType &&
               isSystemBufferType(newBufferType.getValue())) {
      // If the new buffer type is a system buffer type, we need to remove
      // device operands.
      deviceOperandInterface.setDevice(nullptr);
    }

    // Update the tensor ranked type of creation op with the new layout and new
    // data type.
    tensorSpecOp->getResult(0).setType(
        RankedTensorType::Builder(
            mlir::cast<RankedTensorType>(creationOp->getResult(0).getType()))
            .setEncoding(ttnnLayoutAttr)
            .setElementType(ttnnLayoutAttr.getScalarElementType()));

    rewriter.finalizeOpModification(tensorSpecOp);
    rewriter.replaceAllOpUsesWith(toLayoutOp, tensorSpecOp);
    rewriter.eraseOp(toLayoutOp);
    return success();
  });

  // Merging to layout op into TTNN empty op on host should produce ttnn.zeros
  // op on host.
  patterns.add(+[](mlir::tt::ttnn::ToLayoutOp toLayoutOp,
                   mlir::PatternRewriter &rewriter) {
    // Check if the toLayoutOp is being applied on a TTNN empty op
    EmptyOp emptyOp = toLayoutOp.getInput().getDefiningOp<ttnn::EmptyOp>();
    if (!emptyOp) {
      return mlir::failure();
    }

    // Verify that the empty op has a single use.
    if (!emptyOp->hasOneUse()) {
      return failure();
    }

    // Verify that the target buffer type is a system memory.
    BufferTypeAttr bufferTypeAttr = nullptr;
    if (toLayoutOp.getMemoryConfigAttr()) {
      bufferTypeAttr = toLayoutOp.getMemoryConfigAttr().getBufferType();
    }

    if (bufferTypeAttr && !isSystemBufferType(bufferTypeAttr.getValue())) {
      return mlir::failure();
    }

    auto zerosOp = rewriter.replaceOpWithNewOp<mlir::tt::ttnn::ZerosOp>(
        emptyOp, toLayoutOp.getType(), /*device=*/nullptr, emptyOp.getShape(),
        toLayoutOp.getDtypeAttr() ? toLayoutOp.getDtypeAttr()
                                  : emptyOp.getDtypeAttr(),
        toLayoutOp.getLayoutAttr() ? toLayoutOp.getLayoutAttr()
                                   : emptyOp.getLayoutAttr(),
        toLayoutOp.getMemoryConfigAttr() ? toLayoutOp.getMemoryConfigAttr()
                                         : emptyOp.getMemoryConfigAttr());

    rewriter.replaceAllOpUsesWith(toLayoutOp, zerosOp);
    rewriter.eraseOp(toLayoutOp);
    return mlir::success();
  });
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

// LinearOp verification
::mlir::LogicalResult mlir::tt::ttnn::LinearOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  std::optional<::mlir::RankedTensorType> biasType =
      getBias() ? std::make_optional(getBias().getType()) : std::nullopt;
  ::mlir::RankedTensorType outputType = getResult().getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor.
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor.
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimensions for
  // the purpose of the matrix multiplication. After the matrix multiplication,
  // the prepended dimension is removed. Otherwise, check if the LHS needs to be
  // transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards. Otherwise,
  // check if the RHS needs to be transposed.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  } else if (getTransposeB()) {
    std::swap(inputBShape[inputBShape.size() - 1],
              inputBShape[inputBShape.size() - 2]);
  }

  // Verify that the input A and input B has matching inner dimensions.
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A[-1](")
           << inputAShape[inputAShape.size() - 1] << ") and B[-2]("
           << inputBShape[inputBShape.size() - 2]
           << ") must have matching inner dimensions";
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct the
  // expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(inputABatchDims, inputBBatchDims,
                                            broadcastedShape)) {
      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape.
    expectedOutputShape = std::move(broadcastedShape);
  }

  // Insert the input A and B inner dimensions in expected output shape
  // Consider the case where input A and B are vectors. In that case,
  // the dimension 1 is ommited from the output shape.
  if (inputAType.getRank() > 1) {
    expectedOutputShape.push_back(inputAShape[inputAShape.size() - 2]);
  }

  if (inputBType.getRank() > 1) {
    expectedOutputShape.push_back(inputBShape[inputBShape.size() - 1]);
  }

  if (biasType) {
    // Verify that the input bias is at least 1D tensor.
    if (biasType->getRank() < 1) {
      return emitOpError("Bias must be at least a 1D tensor");
    }

    llvm::SmallVector<int64_t> biasShape(biasType->getShape());

    // Verify that the dimensions of the matmul of A and B are broadcast
    // compatible with input bias.
    llvm::SmallVector<int64_t> matmulShape = expectedOutputShape;
    if (!OpTrait::util::getBroadcastedShape(matmulShape, biasShape,
                                            expectedOutputShape)) {
      return emitOpError("Bias shape(")
             << ttmlir::utils::join(biasShape, ",")
             << ") is not broadcast compatible with the matmul output shape("
             << ttmlir::utils::join(matmulShape, ",") << ")";
    }
  }

  // Check the case of a vector-vector product. At this moment we don't support
  // scalars in IR, hence check that the output is at least 1D tensor of size 1.
  if (expectedOutputShape.size() == 0) {
    if (outputType.getRank() < 1) {
      return emitOpError("Scalar output is not supported, output must be at "
                         "least a 1D tensor");
    }

    if (outputType.getRank() > 1 || outputType.getShape()[0] != 1) {
      return emitOpError("Scalar output must be a 1D tensor of size 1");
    }

    return success();
  }

  // Verify that the output shape is correct.
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(")
           << outputShape.size()
           << ") must match the expected output shape rank("
           << expectedOutputShape.size() << ")";
  }

  // Verify each dim of the output shape.
  for (auto [index, outputDim, expectedDim] : llvm::zip(
           llvm::seq(outputShape.size()), outputShape, expectedOutputShape)) {
    if (outputDim != expectedDim) {
      return emitOpError("Output shape dimension[")
             << index << "](" << outputDim
             << ") doesn't match the expected output shape dimension[" << index
             << "](" << expectedDim << ")";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_matmul_ttnn_verify
// MatmulOp verification
::mlir::LogicalResult mlir::tt::ttnn::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor.
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor.
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimensions for
  // the purpose of the matrix multiplication. After the matrix multiplication,
  // the prepended dimension is removed. Otherwise, check if the LHS needs to be
  // transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards. Otherwise,
  // check if the RHS needs to be transposed.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  } else if (getTransposeB()) {
    std::swap(inputBShape[inputBShape.size() - 1],
              inputBShape[inputBShape.size() - 2]);
  }

  // Verify that the input A and input B has matching inner dimensions.
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A[-1](")
           << inputAShape[inputAShape.size() - 1] << ") and B[-2]("
           << inputBShape[inputBShape.size() - 2]
           << ") must have matching inner dimensions";
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct the
  // expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(inputABatchDims, inputBBatchDims,
                                            broadcastedShape)) {

      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape.
    expectedOutputShape = std::move(broadcastedShape);
  }

  // Insert the input A and B inner dimensions in expected output shape
  // Consider the case where input A and B are vectors. In that case,
  // the dimension 1 is ommited from the output shape.
  if (inputAType.getRank() > 1) {
    expectedOutputShape.push_back(inputAShape[inputAShape.size() - 2]);
  }

  if (inputBType.getRank() > 1) {
    expectedOutputShape.push_back(inputBShape[inputBShape.size() - 1]);
  }

  // Check the case of a vector-vector product. At this moment we don't support
  // scalars in IR, hence check that the output is at least 1D tensor of size 1.
  if (expectedOutputShape.size() == 0) {
    if (outputType.getRank() < 1) {
      return emitOpError("Scalar output is not supported, output must be at "
                         "least a 1D tensor");
    }

    if (outputType.getRank() > 1 || outputType.getShape()[0] != 1) {
      return emitOpError("Scalar output must be a 1D tensor of size 1");
    }

    return success();
  }

  // Verify that the output shape is correct.
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(")
           << outputShape.size()
           << ") must match the expected output shape rank("
           << expectedOutputShape.size() << ")";
  }

  // Verify each dim of the output shape.
  for (auto [index, outputDim, expectedDim] : llvm::zip(
           llvm::seq(outputShape.size()), outputShape, expectedOutputShape)) {
    if (outputDim != expectedDim) {
      return emitOpError("Output shape dimension[")
             << index << "](" << outputDim
             << ") doesn't match the expected output shape dimension[" << index
             << "](" << expectedDim << ")";
    }
  }

  return success();
}
// ANCHOR_END: adding_an_op_matmul_ttnn_verify

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

// AllocOp verification
::mlir::LogicalResult AllocOp::verify() {
  auto layout = mlir::dyn_cast_if_present<TTNNLayoutAttr>(
      getResult().getType().getEncoding());
  if (not layout) {
    return emitOpError("Result type missing layout attribute");
  }

  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto bufferType = layout.getBufferType();
  if (bufferType != getBufferType()) {
    return emitOpError(
        "Input tensor layout memory space must match alloc memory space");
  }

  if (isSystemBufferType(getBufferType()) and getAddress() != 0) {
    return emitOpError("Allocating from system memory space must have address "
                       "set to 0, implicitly allocated by the runtime");
  }

  if (isDeviceBufferType(bufferType) and getAddress() == 0) {
    return emitOpError(
        "Allocating from a device memory space must have address "
        "set to a non-zero value, device addresses are statically allocated");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

// RepeatInterleaveOp verification
::mlir::LogicalResult mlir::tt::ttnn::RepeatInterleaveOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  uint32_t repeats = getRepeats();
  int32_t dim = getDim();

  // Verify that the input is at least a 1D tensor.
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Check that the repeats is not zero.
  if (repeats == 0) {
    return emitOpError("Repeats attribute must be non-zero");
  }

  // Check that the dim is within the bounds of the input tensor.
  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError("Dimension attribute must be within the bounds")
           << "[" << -inputType.getRank() << ", " << inputType.getRank() << ")"
           << ", got " << inputType.getRank();
  }

  // Normalize dim to [0, n) range.
  if (dim < 0) {
    dim += inputType.getRank();
  }

  // Compute the expected output shape.
  llvm::SmallVector<int64_t> expectedOutputShape(inputType.getShape());
  expectedOutputShape[dim] *= repeats;

  // Verify that the output shape matches the expected shape.
  if (outputType.getShape() != ::llvm::ArrayRef(expectedOutputShape)) {
    return emitOpError("Output shape ")
           << "[" << ttmlir::utils::join(outputType.getShape(), ",") << "]"
           << " does not match the expected shape "
           << "[" << ttmlir::utils::join(expectedOutputShape, ",") << "]";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

// SoftmaxOp verification
::mlir::LogicalResult mlir::tt::ttnn::SoftmaxOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Shapes of input and output of a softmax operation must be the same
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("Input and output shapes must be the same");
  }

  int32_t dim = getDimension();

  // Check that the dim is within the bounds of the input tensor
  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError(
        "Dimension attribute must be within the bounds of the input tensor");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

// SortOp verification
::mlir::LogicalResult mlir::tt::ttnn::SortOp::verify() {
  auto dim = getDim();
  auto input = getInput();
  auto rank = input.getType().getRank();
  if (dim >= rank || dim < -rank) {
    return emitOpError("Dimension out of range (expected to be in range of [")
           << -rank << ", " << (rank - 1) << "], but got " << dim << ")";
  }

  auto indicesType =
      mlir::cast<RankedTensorType>(getResults().back().getType());
  auto elementType = indicesType.getElementType();
  if (!isa<IntegerType>(elementType)) {
    return emitOpError("Expected integer data type for indices but got ")
           << elementType;
  }

  auto values = getResults().front();
  if (input.getType() != values.getType()) {
    return emitOpError("Sorted tensor type does not match with input tensor.");
  }

  if (input.getType().getShape() != indicesType.getShape()) {
    return emitOpError("Indices shape does not match with input tensor shape.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormOp
//===----------------------------------------------------------------------===//

// BatchNormOp verification
::mlir::LogicalResult mlir::tt::ttnn::BatchNormOp::verify() {

  // Verify that all inputs have dimension 4.
  if (getInput().getType().getRank() != 4) {
    return emitOpError("Input tensor must have rank 4");
  }
  if (getRunningMean().getType().getRank() != 4) {
    return emitOpError("Scale tensor must have rank 4");
  }
  if (getRunningVar().getType().getRank() != 4) {
    return emitOpError("Bias tensor must have rank 4");
  }
  if (getWeight().getType().getRank() != 4) {
    return emitOpError("Weight tensor must have rank 4");
  }
  if (getBias().getType().getRank() != 4) {
    return emitOpError("Bias tensor must have rank 4");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttnn::RMSNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  // Input and output must have the same shape.
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  // For 0D tensors, weight and bias validation is different.
  if (inputType.getRank() == 0) {
    // For 0D tensors, weight and bias should also be 0D if present.
    if (getWeight()) {
      RankedTensorType weightType = getWeight().getType();
      if (weightType.getRank() != 0) {
        return emitOpError("weight tensor must be 0D for 0D input tensor");
      }
    }
    if (getBias()) {
      RankedTensorType biasType = getBias().getType();
      if (biasType.getRank() != 0) {
        return emitOpError("bias tensor must be 0D for 0D input tensor");
      }
    }
    return success();
  }

  // For non-0D tensors, get the last dimension size for weight/bias validation.
  int64_t lastDimSize = inputType.getShape().back();

  // Verify weight tensor shape if present.
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getRank() != 1 || weightType.getShape()[0] != lastDimSize) {
      return emitOpError("weight tensor must be 1D with size matching the last "
                         "dimension of input");
    }
  }

  // Verify bias tensor shape if present.
  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getRank() != 1 || biasType.getShape()[0] != lastDimSize) {
      return emitOpError("bias tensor must be 1D with size matching the last "
                         "dimension of input");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult AllGatherOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t gatherDim = getAllGatherDim();

  if (gatherDim >= inputType.getRank() || gatherDim < -inputType.getRank()) {
    return emitOpError("Invalid gather dimension for all reduce op. Gather "
                       "dimension must be >= to input tensor rank or < -input "
                       "tensor rank, got gather_dim = ")
           << gatherDim;
  }

  return success();
}

::mlir::OpFoldResult mlir::tt::ttnn::AllGatherOp::fold(FoldAdaptor adaptor) {
  ttcore::DeviceAttr device = ttcore::lookupDevice(*this);
  llvm::SmallVector<int64_t> meshShape{device.getMeshShape()};
  // AllGather Op is semantically meaningless when gathering across a single
  // mesh device.
  if (meshShape.empty() || meshShape[getClusterAxis()] != 1) {
    return {};
  }
  // The input and output shapes must be identical in order to fold this op as
  // a no-op.
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  llvm::ArrayRef<int64_t> outputShape = getResult().getType().getShape();
  if (inputShape != outputShape) {
    return {};
  }
  emitWarning() << "Removing this CCL op because performing a CCL operation "
                   "on a single mesh device is semantically meaningless.";
  return getInput();
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ReduceScatterOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t scatterDim = getScatterDim();
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();

  if (scatterDim >= inputType.getRank() || scatterDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid scatter dimension for reduce scatter op. Scatter "
               "dimension must be >= to input tensor rank or < -input "
               "tensor rank, got scatter_dim = ")
           << scatterDim;
  }

  // Currently TTNN only supports the following reduce types. Compiler is able
  // to model the full ReduceType list but only the following can be lowered
  // into TTNN.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Max &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Min) {
    return emitOpError("Invalid reduction op for reduce scatter op.");
  }

  return success();
}

::mlir::OpFoldResult
mlir::tt::ttnn::ReduceScatterOp::fold(FoldAdaptor adaptor) {
  ttcore::DeviceAttr device = ttcore::lookupDevice(*this);
  llvm::SmallVector<int64_t> meshShape{device.getMeshShape()};
  // ReduceScatter Op is semantically meaningless when gathering across a single
  // mesh device.
  if (meshShape.empty() || meshShape[getClusterAxis()] != 1) {
    return {};
  }
  // The input and output shapes must be identical in order to fold this op as
  // a no-op.
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  llvm::ArrayRef<int64_t> outputShape = getResult().getType().getShape();
  if (inputShape != outputShape) {
    return {};
  }
  emitWarning() << "Removing this CCL op because performing a CCL operation "
                   "on a single mesh device is semantically meaningless.";
  return getInput();
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult AllReduceOp::verify() {
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();

  // Currently TTNN only supports the following reduce types.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Max &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Min) {
    return emitOpError("Invalid reduction op for all reduce op.");
  }

  return success();
}

::mlir::OpFoldResult mlir::tt::ttnn::AllReduceOp::fold(FoldAdaptor adaptor) {
  ttcore::DeviceAttr device = ttcore::lookupDevice(*this);
  llvm::SmallVector<int64_t> meshShape{device.getMeshShape()};
  // AllReduce Op is semantically meaningless when gathering across a single
  // mesh device.
  if (!meshShape.empty() && meshShape[getClusterAxis()] == 1) {
    return getInput();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

// CollectivePermuteOp verification
::mlir::LogicalResult CollectivePermuteOp::verify() {
  auto sourceTargetPairs = getSourceTargetPairs().getValues<int64_t>();

  // Check that the rank of sourceTargetPairs is 2D.
  llvm::ArrayRef<int64_t> sourceTargetPairsShape =
      getSourceTargetPairs().getType().getShape();
  const size_t sourceTargetPairsRank = sourceTargetPairsShape.size();

  if (sourceTargetPairsRank != 2) {
    return emitOpError("The rank of source target pairs must be 2, got rank = ")
           << sourceTargetPairsRank;
  }

  /* Check that the 'src' values and 'dest' values in sourceTargetPairs is
  unique. Given a 2D rank tensor of source target pairs eg. [['src', 'target'],
  ['src', 'target'] ...], we need to ensure that each 'src' is unique and each
  'target' is unique.
  */
  auto areElementsUnique = [](const auto &sourceTargetPairs) -> bool {
    for (size_t i = 0; i < sourceTargetPairs.size(); i++) {
      int target = sourceTargetPairs[i];
      for (size_t j = i + 2; j < sourceTargetPairs.size(); j += 2) {
        if (sourceTargetPairs[j] == target) {
          return false;
        }
      }
    }

    return true;
  };

  if (!areElementsUnique(sourceTargetPairs)) {
    return emitOpError(
        "There are duplicate 'src' or 'dest' devices in source target pairs");
  }

  return success();
}

::mlir::OpFoldResult
mlir::tt::ttnn::CollectivePermuteOp::fold(FoldAdaptor adaptor) {
  ::mlir::DenseIntElementsAttr srcTargetPairs = getSourceTargetPairs();

  // Filter out self-mapping src-target pairs.
  auto elements = srcTargetPairs.getValues<APInt>();
  SmallVector<APInt> filteredPairs;
  for (size_t idx = 0; idx < elements.size(); idx += 2) {
    auto src = elements[idx];
    auto target = elements[idx + 1];
    if (src == target) {
      continue;
    }
    filteredPairs.push_back(src);
    filteredPairs.push_back(target);
  }

  if (filteredPairs.empty()) {
    // No permutations left. Exclude this op.
    return getInput();
  }

  // There are effective permutations left.
  if (srcTargetPairs.getNumElements() !=
      static_cast<int64_t>(filteredPairs.size())) {
    // Update source_target_pairs if changed.
    std::array<int64_t, 2> shape = {
        static_cast<int64_t>(filteredPairs.size() / 2), 2};
    auto newType =
        RankedTensorType::get(shape, srcTargetPairs.getType().getElementType());
    setSourceTargetPairsAttr(DenseIntElementsAttr::get(newType, filteredPairs));
  }
  return {};
}
//===----------------------------------------------------------------------===//
// MeshShardOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult MeshShardOp::verify() {
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  llvm::ArrayRef<int64_t> shardShape = getShardShape();
  ::mlir::tt::ttcore::MeshShardType shardType = getShardType();

  // Check shard_type is not maximal.
  if (shardType == ::mlir::tt::ttcore::MeshShardType::Maximal) {
    return emitOpError("Invalid shard_type (maximal) for mesh_shard op.");
  }

  if (shardType == ::mlir::tt::ttcore::MeshShardType::Devices) {
    // Check if rank(shardShape) is eqaul to rank(input).
    if (shardShape.size() != inputShape.size()) {
      return emitOpError("Invalid rank(shard_shape) != rank(input) for "
                         "mesh_shard op with devices partition.");
    }

    // Check if overall partition is eqaul to or greater than two.
    int64_t overallPartition = 1;
    for (auto partition : shardShape) {
      // Each partition value is limited to one of the dimensions of hardware
      // mesh. Thus, overallPartition remains lower than or equal to the number
      // of multi-chips.
      overallPartition *= partition;
    }
    if (overallPartition < 2) {
      return emitOpError("Invalid overall partition (<2) for mesh_shard op "
                         "with devices partition.");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult UpdateCacheOp::verify() {
  if (getBatchOffset() != 0) {
    return emitOpError(
        "Only single-batch is supported. Batch offset must be 0");
  }

  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const ::mlir::tt::ttcore::DataType cacheDataType =
      ::mlir::tt::ttcore::elementTypeToDataType(cacheType.getElementType());
  const ::mlir::tt::ttcore::DataType inputDataType =
      ::mlir::tt::ttcore::elementTypeToDataType(inputType.getElementType());

  if (cacheDataType != inputDataType) {
    return emitOpError(
        "Cache and input tensors must have the same dtype. "
        "Got cache dtype = " +
        DataTypeEnumToString(cacheDataType) +
        ", input dtype = " + DataTypeEnumToString(inputDataType));
  }

  if (cacheType.getRank() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (inputType.getShape()[2] != 1) {
    return emitOpError("Input tensor requires that dim 2 have size 1, got "
                       "input dim 2 size = " +
                       std::to_string(inputType.getShape()[2]));
  }

  if (cacheType.getShape()[0] != inputType.getShape()[0] ||
      cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "all dimensions except dim 2. Got cache shape (" +
                       std::to_string(cacheType.getShape()[0]) + ", " +
                       std::to_string(cacheType.getShape()[1]) + ", " +
                       std::to_string(cacheType.getShape()[2]) + ", " +
                       std::to_string(cacheType.getShape()[3]) +
                       "), input shape ()" +
                       std::to_string(inputType.getShape()[0]) + "x" +
                       std::to_string(inputType.getShape()[1]) + "x" +
                       std::to_string(inputType.getShape()[2]) + "x" +
                       std::to_string(inputType.getShape()[3]) + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult FillCacheOp::verify() {
  if (getBatchOffset() != 0) {
    return emitOpError(
        "Only single-batch is supported. Batch offset must be 0");
  }

  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const ::mlir::tt::ttcore::DataType cacheDataType =
      ::mlir::tt::ttcore::elementTypeToDataType(cacheType.getElementType());
  const ::mlir::tt::ttcore::DataType inputDataType =
      ::mlir::tt::ttcore::elementTypeToDataType(inputType.getElementType());

  if (cacheDataType != inputDataType) {
    return emitOpError(
        "Cache and input tensors must have the same dtype. "
        "Got cache dtype = " +
        DataTypeEnumToString(cacheDataType) +
        ", input dtype = " + DataTypeEnumToString(inputDataType));
  }

  if (cacheType.getRank() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (inputType.getShape()[2] > cacheType.getShape()[2]) {
    return emitOpError(
        "Input tensor requires that dim 2 have a size which is less than or "
        "equal to the size of dim 2 of the cache tensor. Got cache dim 2 size "
        "= " +
        std::to_string(cacheType.getShape()[2]) +
        ", input dim 2 size = " + std::to_string(inputType.getShape()[2]));
  }

  if (cacheType.getShape()[0] != inputType.getShape()[0] ||
      cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "all dimensions except dim 2. Got cache shape (" +
                       std::to_string(cacheType.getShape()[0]) + ", " +
                       std::to_string(cacheType.getShape()[1]) + ", " +
                       std::to_string(cacheType.getShape()[2]) + ", " +
                       std::to_string(cacheType.getShape()[3]) +
                       "), input shape (" +
                       std::to_string(inputType.getShape()[0]) + ", " +
                       std::to_string(inputType.getShape()[1]) + ", " +
                       std::to_string(inputType.getShape()[2]) + ", " +
                       std::to_string(inputType.getShape()[3]) + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

// PermuteOp verification
::mlir::LogicalResult mlir::tt::ttnn::PermuteOp::verify() {
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  const size_t inputRank = inputShape.size();
  llvm::ArrayRef<int64_t> resultShape = getResult().getType().getShape();

  // Check that given attribute `permutation` is a valid permutation of the
  // dimensions.
  llvm::ArrayRef<int64_t> permutation = getPermutation();
  llvm::SmallVector<int64_t> dimensions(inputRank);
  std::iota(dimensions.begin(), dimensions.end(), 0);
  if (inputRank != permutation.size() ||
      !std::is_permutation(permutation.begin(), permutation.end(),
                           dimensions.begin())) {
    return emitOpError("Expected a permutation of (")
           << ttmlir::utils::join(dimensions, ", ")
           << "), got (" + ttmlir::utils::join(permutation, ", ") << ")";
  }

  // Check that the result shape matches the shape of input tensor after
  // permutation is applied.
  llvm::SmallVector<int64_t> expectedResultShape =
      ttmlir::utils::applyPermutation(inputShape, permutation);
  if (!llvm::equal(expectedResultShape, resultShape)) {
    return emitOpError("Expected result shape (" +
                       ttmlir::utils::join(expectedResultShape, ", ") +
                       "), got (" + ttmlir::utils::join(resultShape, ", ") +
                       ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

// UpsampleOp verification
::mlir::LogicalResult UpsampleOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Input tensor is assumed to be 4D tensor.
  if (inputType.getRank() != 4) {
    return emitOpError("Expected rank of input tensor is 4, got rank " +
                       std::to_string(inputType.getRank()));
  }
  if (outputType.getRank() != 4) {
    return emitOpError("Expected rank of output tensor is 4, got rank " +
                       std::to_string(outputType.getRank()));
  }

  auto scaleFactor = ttmlir::utils::getPairOfInteger<int32_t>(getScaleFactor());
  if (auto error = scaleFactor.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }
  int32_t scaleH = scaleFactor->first;
  int32_t scaleW = scaleFactor->second;

  if (scaleH <= 0 || scaleW <= 0) {
    return emitOpError("Scale factors H = ")
           << scaleH << " and W = " << scaleW << " must be positive integers";
  }

  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  // Input tensor is assumed to be in NHWC format.
  enum Dimensions { DIM_N = 0, DIM_H = 1, DIM_W = 2, DIM_C = 3 };
  if (inputShape[DIM_H] * scaleH != outputShape[DIM_H]) {
    return emitOpError("Expected output H dimension to be input H dimension * "
                       "scaleH = ")
           << (inputShape[DIM_H] * scaleH) << ", got " << outputShape[DIM_H];
  }
  if (inputShape[DIM_W] * scaleW != outputShape[DIM_W]) {
    return emitOpError("Expected output W dimension to be input W dimension * "
                       "scaleW = ")
           << (inputShape[DIM_W] * scaleW) << ", got " << outputShape[DIM_W];
  }
  if (inputShape[DIM_N] != outputShape[DIM_N]) {
    return emitOpError("Expected output N dimension to be ")
           << inputShape[DIM_N] << ", got " << outputShape[DIM_N];
  }
  if (inputShape[DIM_C] != outputShape[DIM_C]) {
    return emitOpError("Expected output C dimension to be ")
           << inputShape[DIM_C] << ", got " << outputShape[DIM_C];
  }

  // Verify that the mode attribute is one of the legal modes. These two modes
  // are currently only supported modes in TTNN.
  llvm::SmallVector<llvm::StringRef> legalModes = {"nearest", "bilinear"};
  if (std::find(legalModes.begin(), legalModes.end(), getMode()) ==
      legalModes.end()) {
    return emitOpError("Expected modes are (")
           << llvm::join(legalModes, ", ") << "), got \"" << getMode() << "\"";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

// Common verifier for all Reduction ops.
static mlir::LogicalResult
verifyReduceOp(llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
               mlir::RankedTensorType inputType,
               const std::optional<mlir::ArrayAttr> &reduceDims, bool keepDim,
               ::llvm::ArrayRef<int64_t> specifiedOutputShape) {

  int64_t inputTensorRank = inputType.getRank();

  llvm::BitVector reduceDimsMask(inputTensorRank, false);
  if (reduceDims) {
    for (mlir::Attribute attr : *reduceDims) {
      int64_t reduceDim = mlir::cast<mlir::IntegerAttr>(attr).getInt();
      // Normalize range to [0, inputTensorRank).
      if (reduceDim < 0) {
        reduceDim += inputTensorRank;
      }
      reduceDimsMask.set(reduceDim);
    }
  } else {
    reduceDimsMask.set();
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  for (int64_t index = 0; index < inputTensorRank; ++index) {
    if (!reduceDimsMask[index]) {
      expectedOutputShape.push_back(inputType.getDimSize(index));
    } else if (keepDim) {
      expectedOutputShape.push_back(1);
    }
  }

  // Finally, compare shapes.
  if (!llvm::equal(specifiedOutputShape, expectedOutputShape)) {
    return emitOpError() << "Expected output shape ("
                         << ttmlir::utils::join(expectedOutputShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(specifiedOutputShape, ", ")
                         << ")";
  }

  return mlir::success();
}

// Verifier for Reduce ProdOp.
static mlir::LogicalResult
verifyReduceProdOp(tt::ttnn::ProdOp *reduceOp,
                   mlir::RankedTensorType inputType) {
  int64_t inputTensorRank = inputType.getRank();

  if (inputTensorRank > 4) {
    return reduceOp->emitOpError(
        "Input tensor rank is greater than 4 for reduce(product).");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

// MaxOp verification.
::mlir::LogicalResult MaxOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(),
                        getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// MeanOp verification.
::mlir::LogicalResult MeanOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(),
                        getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

// SumOp verification.
::mlir::LogicalResult SumOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(),
                        getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce MinOp
//===----------------------------------------------------------------------===//

// MinOp verification.
::mlir::LogicalResult MinOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(),
                        getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ProdOp
//===----------------------------------------------------------------------===//

// ProdOp verification.
::mlir::LogicalResult ProdOp::verify() {
  return verifyReduceProdOp(this, getInput().getType());
}

//===----------------------------------------------------------------------===//
// WriteTensorOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult WriteTensorOp::verify() {
  auto hostTensorType =
      ::mlir::cast<RankedTensorType>(this->getHostTensor().getType());
  if (!hostTensorType) {
    return emitOpError() << "Host tensor must be RankedTensorType";
  }
  if (utils::isTensorOnDevice(hostTensorType)) {
    return emitOpError() << "Host tensor must be on system memory";
  }

  auto deviceTensorType =
      ::mlir::cast<RankedTensorType>(this->getDeviceTensor().getType());
  if (!deviceTensorType) {
    return emitOpError() << "Device tensor must be RankedTensorType";
  }
  if (!utils::isTensorOnDevice(deviceTensorType)) {
    return emitOpError() << "Device tensor must be on device memory";
  }

  uint32_t cqId = this->getCqId();
  if (llvm::find(VALID_CQ_IDS, cqId) == VALID_CQ_IDS.end()) {
    return emitOpError() << "Invalid CQ ID " << cqId;
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// TraceOps
//===----------------------------------------------------------------------===//

static ::mlir::LogicalResult verifyTraceIdTensor(Operation *op, Value traceId) {
  if (!traceId) {
    return op->emitError() << "Trace ID must be set";
  }
  if (!mlir::isa<mlir::RankedTensorType>(traceId.getType())) {
    return op->emitError() << "Trace ID must be a ranked tensor type";
  }
  auto traceIdTensor = mlir::cast<mlir::RankedTensorType>(traceId.getType());
  if (traceIdTensor.getRank() != 0) {
    return op->emitError() << "Trace ID must be a scalar";
  }
  auto intType =
      mlir::dyn_cast<mlir::IntegerType>(traceIdTensor.getElementType());
  if (!intType) {
    return op->emitError() << "Trace ID must be an integer";
  }
  if (!intType.isUnsigned()) {
    return op->emitError() << "Trace ID must be unsigned";
  }
  if (intType.getWidth() != 32) {
    return op->emitError() << "Trace ID must be 32-bit";
  }
  if (!mlir::isa_and_present<ttnn::TraceIdAttr>(traceIdTensor.getEncoding())) {
    return op->emitError() << "Trace ID must have the TraceIdAttr encoding";
  }
  return ::mlir::success();
}

void BeginTraceCaptureOp::getEffects(
    ::mlir::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(),
                       TraceResource::get());
  effects.emplace_back(::mlir::MemoryEffects::Write::get(),
                       TraceResource::get());
}

::mlir::LogicalResult BeginTraceCaptureOp::verify() {
  uint32_t cqId = this->getCqId();
  if (llvm::find(VALID_CQ_IDS, cqId) == VALID_CQ_IDS.end()) {
    return emitOpError() << "Invalid CQ ID " << cqId;
  }

  ::mlir::LogicalResult traceIdResult =
      verifyTraceIdTensor(this->getOperation(), this->getTraceId());
  if (failed(traceIdResult)) {
    return traceIdResult;
  }

  return ::mlir::success();
}

void EndTraceCaptureOp::getEffects(
    ::mlir::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(),
                       TraceResource::get());
  effects.emplace_back(::mlir::MemoryEffects::Write::get(),
                       TraceResource::get());
}

::mlir::LogicalResult EndTraceCaptureOp::verify() {
  uint32_t cqId = this->getCqId();
  if (llvm::find(VALID_CQ_IDS, cqId) == VALID_CQ_IDS.end()) {
    return emitOpError() << "Invalid CQ ID " << cqId;
  }

  ::mlir::LogicalResult traceIdResult =
      verifyTraceIdTensor(this->getOperation(), this->getTraceId());
  if (failed(traceIdResult)) {
    return traceIdResult;
  }

  return ::mlir::success();
}

void ExecuteTraceOp::getEffects(
    ::mlir::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(),
                       TraceResource::get());
  effects.emplace_back(::mlir::MemoryEffects::Write::get(),
                       TraceResource::get());
}

::mlir::LogicalResult ExecuteTraceOp::verify() {
  uint32_t cqId = this->getCqId();
  if (llvm::find(VALID_CQ_IDS, cqId) == VALID_CQ_IDS.end()) {
    return emitOpError() << "Invalid CQ ID " << cqId;
  }

  ::mlir::LogicalResult traceIdResult =
      verifyTraceIdTensor(this->getOperation(), this->getTraceId());
  if (failed(traceIdResult)) {
    return traceIdResult;
  }

  return ::mlir::success();
}

void CaptureOrExecuteTraceOp::getEffects(
    ::mlir::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(),
                       TraceResource::get());
  effects.emplace_back(::mlir::MemoryEffects::Write::get(),
                       TraceResource::get());
}

::mlir::LogicalResult CaptureOrExecuteTraceOp::verify() {
  // Verify that the callee exists
  FlatSymbolRefAttr captureCalleeAttr = this->getCaptureCalleeAttr();
  func::FuncOp captureFuncOp =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this,
                                                         captureCalleeAttr);
  if (!captureFuncOp) {
    return emitOpError() << "'" << captureCalleeAttr.getValue()
                         << "' does not reference a function";
  }

  FlatSymbolRefAttr traceFuncCalleeAttr;
  captureFuncOp.walk([&](func::CallOp callOp) {
    traceFuncCalleeAttr = callOp.getCalleeAttr();
    return WalkResult::interrupt();
  });

  if (!traceFuncCalleeAttr) {
    return emitOpError() << "No trace function found in capture function";
  }

  func::FuncOp traceFuncOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      *this, traceFuncCalleeAttr);
  if (!traceFuncOp) {
    return emitOpError() << "'" << traceFuncCalleeAttr.getValue()
                         << "' does not reference a function";
  }

  for (BlockArgument arg : traceFuncOp.getArguments()) {
    if (!::mlir::isa<RankedTensorType>(arg.getType())) {
      return emitOpError() << "All input arguments of trace function must be "
                           << "ranked tensors";
    }
    auto tensorType = ::mlir::cast<RankedTensorType>(arg.getType());
    if (!utils::isTensorOnDevice(tensorType)) {
      return emitOpError()
             << "All input arguments of trace function must be on device."
             << arg << " is not on device.";
    }
  }

  ::mlir::WalkResult walkResult =
      traceFuncOp.walk(
          [&](Operation *op) -> ::mlir::WalkResult {
            if (::mlir::isa<ttnn::CaptureOrExecuteTraceOp>(op)) {
              emitOpError() << "CaptureOrExecuteTraceOp op must not be nested "
                               "within trace function";
              return ::mlir::WalkResult::interrupt();
            }

            if (::mlir::isa<::mlir::tt::ttcore::LoadCachedOp>(op)) {
              emitOpError()
                  << "LoadCached op must not be nested within trace function";
              return ::mlir::WalkResult::interrupt();
            }

            if (::mlir::isa<GetDeviceOp>(op)) {
              auto traceDevice = this->getDevice();
              auto traceDeviceOp = traceDevice.getDefiningOp<GetDeviceOp>();

              auto calleeDeviceOp = ::mlir::cast<GetDeviceOp>(op);

              // Need to make sure that the device attributes of the trace op
              // and the device within the callee function match
              if (traceDeviceOp.getMeshShape() !=
                      calleeDeviceOp.getMeshShape() ||
                  traceDeviceOp.getMeshOffset() !=
                      calleeDeviceOp.getMeshOffset()) {
                return emitOpError()
                       << "Device configuration of get_device op in callee "
                       << "must match device configuration of trace op";
              }
              return ::mlir::WalkResult::advance();
            }

            // Make sure all input tensors are on device
            for (Value operand : op->getOperands()) {
              if (!::mlir::isa<RankedTensorType>(operand.getType())) {
                continue;
              }
              auto tensorType =
                  ::mlir::cast<RankedTensorType>(operand.getType());
              if (!utils::isTensorOnDevice(tensorType)) {
                emitOpError()
                    << "All input tensors of trace function must be on device."
                    << operand << " is not on device.";
                return ::mlir::WalkResult::interrupt();
              }
            }

            // Make sure all output tensors are on device
            for (Value result : op->getResults()) {
              if (!::mlir::isa<RankedTensorType>(result.getType())) {
                continue;
              }
              auto tensorType =
                  ::mlir::cast<RankedTensorType>(result.getType());
              if (!utils::isTensorOnDevice(tensorType)) {
                emitOpError()
                    << "All output tensors of trace function must be on device."
                    << result << " is not on device.";
                return ::mlir::WalkResult::interrupt();
              }
            }

            return ::mlir::WalkResult::advance();
          });

  if (walkResult.wasInterrupted()) {
    return ::mlir::failure();
  }

  FlatSymbolRefAttr executeCalleeAttr = this->getExecuteCalleeAttr();
  func::FuncOp executeFuncOp =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this,
                                                         executeCalleeAttr);
  if (!executeFuncOp) {
    return emitOpError() << "'" << executeCalleeAttr.getValue()
                         << "' does not reference a function";
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// PointToPointOp
//===----------------------------------------------------------------------===//

// PointToPointOp verification
::mlir::LogicalResult mlir::tt::ttnn::PointToPointOp::verify() {
  if (getAccumTensor()) { // accum_tensor is optional
    auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
    auto outputType =
        llvm::dyn_cast<RankedTensorType>(getAccumTensor().getType());

    if (inputType.getElementType() != outputType.getElementType() ||
        inputType.getShape() != outputType.getShape()) {
      return emitOpError(
          "Accum tensor must match input tensor in shape and element type.");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ConcatenateHeadsOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  ::mlir::tt::ttcore::DataType inputDataType =
      ::mlir::tt::ttcore::elementTypeToDataType(inputType.getElementType());
  ::mlir::tt::ttcore::DataType outputDataType =
      ::mlir::tt::ttcore::elementTypeToDataType(outputType.getElementType());

  if (inputDataType != outputDataType) {
    return emitOpError()
           << "input and output tensors must have the same dtype, "
           << "got input dtype = " << DataTypeEnumToString(inputDataType)
           << ", output dtype = " << DataTypeEnumToString(outputDataType);
  }

  if (inputType.getRank() != 4) {
    return emitOpError() << "input tensor must be a 4D tensor";
  }

  if (outputType.getRank() != 3) {
    return emitOpError() << "output tensor must be a 3D tensor";
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Input tensor dimensions [batch_size, num_heads, sequence_size, head_size]
  // Output tensor dimensions [batch_size, sequence_size, num_heads * head_size]
  using namespace ttmlir::utils::transformer;

  // Verify batch_size dimension matches
  if (inputShape[INPUT_BATCH] != outputShape[OUTPUT_BATCH]) {
    return emitOpError() << "input and output batch dimensions must match,"
                            "got input batch size = "
                         << inputShape[INPUT_BATCH] << ", output batch size = "
                         << outputShape[OUTPUT_BATCH];
  }

  // Verify sequence_size dimension matches
  if (inputShape[INPUT_SEQ] != outputShape[OUTPUT_SEQ]) {
    return emitOpError()
           << "input sequence dimension must match output sequence dimension, "
              "got input sequence size = "
           << inputShape[INPUT_SEQ]
           << ", output sequence size = " << outputShape[OUTPUT_SEQ];
  }

  // Verify that num_heads * head_size equals the output hidden dimension
  int64_t expectedHiddenSize =
      inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE];
  if (expectedHiddenSize != outputShape[OUTPUT_HIDDEN]) {
    return emitOpError()
           << "output hidden dimension must equal num_heads * head_size, "
              "got num_heads = "
           << inputShape[INPUT_NUM_HEADS]
           << ", head_size = " << inputShape[INPUT_HEAD_SIZE]
           << ", expected hidden size = " << expectedHiddenSize
           << ", actual output hidden size = " << outputShape[OUTPUT_HIDDEN];
  }

  return success();
}

//===-----------------------------------------------------------------------===//
// NLPConcatHeadsOp
// ===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::NLPConcatHeadsOp::verify() {
  ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  ArrayRef<int64_t> outputShape = getResult().getType().getShape();

  if (outputShape.size() != 4) {
    return emitOpError() << "output tensor must be a 4D tensor";
  }

  if (inputShape.size() != 4) {
    return emitOpError() << "input tensor must be a 4D tensor";
  }

  using namespace ttmlir::utils::transformer;

  llvm::SmallVector<int64_t> expectedOutputShape = {
      inputShape[INPUT_BATCH], 1, inputShape[INPUT_SEQ],
      inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE]};

  if (!llvm::equal(expectedOutputShape, outputShape)) {
    return emitOpError() << "expected output shape ("
                         << ttmlir::utils::join(expectedOutputShape, ", ")
                         << "), got (" << ttmlir::utils::join(outputShape, ", ")
                         << ")";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

// GenericOp verification
::mlir::LogicalResult mlir::tt::ttnn::GenericOp::verify() {
  ProgramAttr program = getProgram();
  size_t numberOfInputsAndOutputs = getInputsAndOutputs().size();
  size_t numberOfSemaphores = program.getSemaphores().size();

  for (auto kernel : program.getKernels()) {
    auto kernelInterface = llvm::cast<KernelInterface>(kernel);

    for (auto arg : kernelInterface.getCommonRtArgs()) {
      if (auto addressOfTensor =
              llvm::dyn_cast_or_null<KernelArgAddressOfTensorAttr>(arg)) {
        if (addressOfTensor.getTensorIndex() >= numberOfInputsAndOutputs) {
          return emitError() << "Address of tensor at index is out of bounds";
        }
      }
      if (auto semaphoreAt =
              llvm::dyn_cast_or_null<KernelArgSemaphoreAtAttr>(arg)) {
        if (semaphoreAt.getSemaphoreIndex() >= numberOfSemaphores) {
          return emitError() << "Semaphore at index is out of bounds";
        }
      }
    }

    for (auto arg : kernelInterface.getCtArgs()) {
      if (auto semaphoreAt =
              llvm::dyn_cast_or_null<KernelArgSemaphoreAtAttr>(arg)) {
        if (semaphoreAt.getSemaphoreIndex() >= numberOfSemaphores) {
          return emitError() << "Semaphore at index is out of bounds";
        }
      }
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOp
//===----------------------------------------------------------------------===//

// NLPConcatHeadsDecodeOp verification
::mlir::LogicalResult NLPConcatHeadsDecodeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  if (inputType.getRank() != 4) {
    return emitOpError() << "input tensor must be a 4D tensor";
  }

  if (outputType.getRank() != 4) {
    return emitOpError() << "output tensor must be a 4D tensor";
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Input tensor dimensions [sequence_size, batch_size, num_heads, head_size]
  // Output tensor dimensions [sequence_size, 1, batch_size, num_heads *
  // head_size]
  enum InputDimensions {
    INPUT_SEQ = 0,
    INPUT_BATCH = 1,
    INPUT_NUM_HEADS = 2,
    INPUT_HEAD_SIZE = 3
  };

  enum OutputDimensions { OUTPUT_SEQ = 0, OUTPUT_BATCH = 2, OUTPUT_HIDDEN = 3 };

  uint32_t numHeads = getNumHeads();

  if (outputShape[1] != 1) {
    return emitOpError() << "output dimension 1 must be 1, got "
                         << outputShape[1];
  }

  if (inputShape[INPUT_NUM_HEADS] < numHeads) {
    return emitOpError() << "num_heads attribute must be less than or equal to "
                            "input num_heads "
                            "dimension, got num_heads = "
                         << numHeads << ", input num_heads dimension = "
                         << inputShape[INPUT_NUM_HEADS];
  }

  if (inputShape[INPUT_SEQ] != outputShape[OUTPUT_SEQ]) {
    return emitOpError()
           << "input sequence dimension must match output sequence dimension, "
              "got input sequence size = "
           << inputShape[INPUT_SEQ]
           << ", output sequence size = " << outputShape[OUTPUT_SEQ];
  }

  // Verify that num_heads * head_size equals the output hidden dimension
  int64_t expectedHiddenSize = numHeads * inputShape[INPUT_HEAD_SIZE];
  if (expectedHiddenSize != outputShape[OUTPUT_HIDDEN]) {
    return emitOpError()
           << "Output hidden dimension must equal num_heads * head_size, "
              "got num_heads = "
           << inputShape[INPUT_NUM_HEADS] << ", head_size = " << numHeads
           << ", expected hidden size = " << expectedHiddenSize
           << ", actual output hidden size = " << outputShape[OUTPUT_HIDDEN];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOp
//===----------------------------------------------------------------------===//

// RotaryEmbeddingLlamaOp verification
mlir::LogicalResult RotaryEmbeddingLlamaOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType cosType = getCosCache().getType();
  mlir::RankedTensorType sinType = getSinCache().getType();
  mlir::RankedTensorType transMatType = getTransMat().getType();

  llvm::SmallVector<mlir::RankedTensorType> inputTypes = {
      inputType, cosType, sinType, transMatType};
  auto dtypePredicate = [](mlir::RankedTensorType type) {
    return isa<BFloat16Type>(type.getElementType());
  };

  auto tileLayoutPredicate = [](mlir::RankedTensorType type) {
    auto encoding = cast<ttnn::TTNNLayoutAttr>(type.getEncoding());
    return encoding.isTiled();
  };

  auto deviceStoragePredicate = [](mlir::RankedTensorType type) {
    return ttnn::utils::isTensorOnDevice(type);
  };

  if (!llvm::all_of(inputTypes, dtypePredicate)) {
    return emitOpError("all input tensors must be bfloat16 type.");
  }

  if (!llvm::all_of(inputTypes, tileLayoutPredicate)) {
    return emitOpError("all input tensors must have tiled layout.");
  }

  if (!llvm::all_of(inputTypes, deviceStoragePredicate)) {
    return emitOpError("all input tensors must be on device.");
  }

  // Cos and sin must have the same shape
  if (cosType.getShape() != sinType.getShape()) {
    return emitOpError("cos and sin tensors must have the same shape.");
  }

  // Head dimension validation
  int64_t headDim = inputType.getShape().back();
  if (headDim > 256) {
    return emitOpError("head dimension must be ≤ 256, got ") << headDim << ".";
  }

  // Transformation matrix shape validation
  auto transShape = transMatType.getShape();
  if (transShape.size() != 4) {
    return emitOpError("transformation matrix must be at least 4D tensor.");
  }

  llvm::SmallVector<int64_t> expectedTransShape = {1, 1, TILE_WIDTH,
                                                   TILE_HEIGHT};
  if (!llvm::equal(transShape.take_back(4), expectedTransShape)) {
    return emitOpError() << "transformation matrix must have shape ("
                         << ttmlir::utils::join(expectedTransShape, ",")
                         << ") but got ("
                         << ttmlir::utils::join(transShape, ",") << ").";
  }

  // Decode mode specific validations
  if (getIsDecodeMode()) {
    auto isHeightShardedPredicate = [](RankedTensorType type) {
      auto encoding = mlir::cast<ttnn::TTNNLayoutAttr>(type.getEncoding());
      return encoding.getMemLayout() &&
             encoding.getMemLayout().getValue() ==
                 ttnn::TensorMemoryLayout::HeightSharded;
    };

    if (!llvm::all_of(inputTypes, isHeightShardedPredicate)) {
      return emitOpError("in decode mode, all input tensors must have "
                         "HeightSharded memory layout.");
    }
  }

  mlir::RankedTensorType outputType = getResult().getType();

  // Output shape should match input shape
  if (outputType.getShape() != inputType.getShape()) {
    return emitOpError("output shape must match input shape.");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// NLPCreateQKVHeadsDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult NLPCreateQKVHeadsDecodeOp::verify() {
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DumpTensorOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult DumpTensorOp::verify() {
  if (!getFilePath().ends_with(".tensorbin")) {
    return emitOpError() << "file " << getFilePath()
                         << " must end with .tensorbin extension";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LoadTensorOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult LoadTensorOp::verify() {
  if (!getFilePath().ends_with(".tensorbin")) {
    return emitOpError() << "file " << getFilePath()
                         << " must end with .tensorbin extension";
  }

  auto resultLayout =
      mlir::cast<TTNNLayoutAttr>(getResult().getType().getEncoding());
  auto device = getDevice();

  if (device && !resultLayout.isDeviceBufferType()) {
    return emitOpError(
        "device operand must be null for system memory buffer type");
  }

  if (!device && resultLayout.isDeviceBufferType()) {
    return emitOpError(
        "device operand must be specified for device memory buffer type");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType valueType = getValue().getType();
  RankedTensorType curPosTensorType = getCurPosTensor().getType();
  RankedTensorType resultType = getResult().getType();

  if (queryType != resultType) {
    return emitOpError("Query and result must have the same type");
  }

  if (!curPosTensorType.getElementType().isInteger()) {
    return emitOpError("Cur pos tensor must be a tensor of integers");
  }

  if (curPosTensorType.getShape().size() != 1) {
    return emitOpError("Cur pos tensor must be a 1D tensor");
  }

  if (keyType != valueType) {
    return emitOpError("Key and value must have the same type");
  }
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key/Value must be a 4D tensor");
  }
  if (resultType.getShape().size() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  if (queryType.getShape()[0] != 1) {
    return emitOpError("Query dim 0 must be 1");
  }

  int64_t batchSize = queryType.getShape()[1];
  int64_t nQueryHeads = queryType.getShape()[2];
  int64_t nKVHeads = keyType.getShape()[1];
  int64_t headSize = queryType.getShape()[3];
  int64_t maxSeqLen = keyType.getShape()[2];

  if (curPosTensorType.getShape()[0] != batchSize) {
    return emitOpError("Cur pos tensor batch size must match query batch size");
  }

  if (keyType.getShape()[0] != batchSize) {
    return emitOpError("Key/Value batch size must match query batch size");
  }

  if (keyType.getShape()[3] != headSize) {
    return emitOpError("Key/Value head size must match query head size");
  }

  if (nQueryHeads % nKVHeads != 0) {
    return emitOpError(
        "Query num heads must be divisible by key/value num heads");
  }

  if (getAttentionMask()) {
    if (getIsCausal()) {
      return emitOpError(
          "Attention mask is not allowed when is_causal is true");
    }
    RankedTensorType attentionMaskType = getAttentionMask().getType();
    if (attentionMaskType.getShape().size() != 4) {
      return emitOpError("Attention mask must be a 4D tensor");
    }
    if (attentionMaskType.getShape()[0] != batchSize) {
      return emitOpError(
          "Attention mask batch size must match query batch size");
    }
    if (attentionMaskType.getShape()[1] != 1) {
      return emitOpError("Attention mask dim 1 must be 1");
    }
    if (attentionMaskType.getShape()[2] != nQueryHeads) {
      return emitOpError("Attention mask num heads must match query num heads");
    }
    if (attentionMaskType.getShape()[3] != maxSeqLen) {
      return emitOpError("Attention mask sequence length must match key/value "
                         "sequence length");
    }
  } else {
    if (!getIsCausal()) {
      return emitOpError("Attention mask is required when is_causal is false");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::ScaledDotProductAttentionOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType valueType = getValue().getType();
  RankedTensorType resultType = getResult().getType();

  if (queryType != resultType) {
    return emitOpError("Query and result must have the same type");
  }

  if (keyType != valueType) {
    return emitOpError("Key and value must have the same type");
  }
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key/Value must be a 4D tensor");
  }
  if (resultType.getShape().size() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  int64_t batchSize = queryType.getShape()[0];
  int64_t nQueryHeads = queryType.getShape()[1];
  int64_t nKVHeads = keyType.getShape()[1];
  int64_t headSize = queryType.getShape()[3];
  int64_t seqLen = queryType.getShape()[2];
  int64_t maxSeqLen = keyType.getShape()[2];

  // NOTE: The q_chunk_size is 32 by default in ttnn. This is configurable via
  // the program config. However, this is not modelled in the ttnn dialect.
  if (seqLen % 32 != 0) {
    return emitOpError(
        "Sequence length must be divisible by q_chunk_size (32)");
  }

  if (keyType.getShape()[0] != batchSize) {
    return emitOpError("Key/Value batch size must match query batch size");
  }

  if (keyType.getShape()[3] != headSize) {
    return emitOpError("Key/Value head size must match query head size");
  }

  if (nQueryHeads % nKVHeads != 0) {
    return emitOpError(
        "Query num heads must be divisible by key/value num heads");
  }

  if (getAttentionMask()) {
    if (getIsCausal()) {
      return emitOpError(
          "Attention mask is not allowed when is_causal is true");
    }
    RankedTensorType attentionMaskType = getAttentionMask().getType();
    if (attentionMaskType.getShape().size() != 4) {
      return emitOpError("Attention mask must be a 4D tensor");
    }
    if (attentionMaskType.getShape()[0] != batchSize) {
      return emitOpError(
          "Attention mask batch size must match query batch size");
    }
    if (attentionMaskType.getShape()[1] != 1) {
      return emitOpError("Attention mask dim 1 must be 1");
    }
    if (attentionMaskType.getShape()[2] != seqLen) {
      return emitOpError(
          "Attention mask at dim 2 must match query sequence length");
    }
    if (attentionMaskType.getShape()[3] != maxSeqLen) {
      return emitOpError("Attention mask at dim 3 must match key/value "
                         "sequence length (max sequence length)");
    }
  } else {
    if (!getIsCausal()) {
      return emitOpError("Attention mask is required when is_causal is false");
    }
  }

  if (getIsCausal()) {
    if (seqLen != maxSeqLen) {
      return emitOpError("Sequence length must match key/value sequence length "
                         "when is_causal is true");
    }
  }

  return success();
}

} // namespace mlir::tt::ttnn
