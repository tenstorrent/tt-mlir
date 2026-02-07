// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.cpp.inc"
#include "ttmlir/Dialect/TTIR/Utils/QuantUtils.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTIR/Utils/VerificationUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <numeric>
#include <string>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"

namespace mlir::tt::ttir {

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

bool mlir::tt::ttir::AddOp::isQuantizedRewriteFavorable(
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // If the operands are both quantized but the types do not align, return
  // false.
  return mlir::tt::ttir::utils::areQuantizationParamsAligned(sourceOperands);
}

mlir::Operation *mlir::tt::ttir::AddOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter,
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // Two cases:
  // 1. One operand is quantized and the other is not: apply quantization and
  //    proceed to case two.
  // 2. Both operands are quantized: supported, return quantized add.
  assert(sourceOperands.size() == 2 && "AddOp should have two operands.");
  auto lhs = sourceOperands[0];
  auto rhs = sourceOperands[1];

  RankedTensorType lhsType = mlir::cast<RankedTensorType>(lhs.getType());
  RankedTensorType rhsType = mlir::cast<RankedTensorType>(rhs.getType());

  auto lhsElemQ =
      mlir::dyn_cast<mlir::quant::QuantizedType>(lhsType.getElementType());
  auto rhsElemQ =
      mlir::dyn_cast<mlir::quant::QuantizedType>(rhsType.getElementType());

  // One operand is dequantized, one is quantized — try to quantize the
  // dequantized one.
  if ((lhsElemQ && !rhsElemQ) || (!lhsElemQ && rhsElemQ)) {
    Value quantVal = lhsElemQ ? lhs : rhs;
    Value dequantVal = lhsElemQ ? rhs : lhs;
    auto quantElemQ = lhsElemQ ? lhsElemQ : rhsElemQ;
    auto quantType = mlir::cast<mlir::RankedTensorType>(quantVal.getType());
    auto expressedType =
        mlir::cast<mlir::RankedTensorType>(dequantVal.getType())
            .getElementType();

    // Insert quantize op for the dequantized value (the types must be
    // compatible).
    if (!isa<mlir::quant::UniformQuantizedType,
             mlir::quant::UniformQuantizedPerAxisType>(quantElemQ)) {
      return nullptr;
    }
    if (expressedType != quantElemQ.getExpressedType()) {
      return nullptr;
    }

    RankedTensorType newType = RankedTensorType::get(
        mlir::cast<mlir::RankedTensorType>(dequantVal.getType()).getShape(),
        quantElemQ, quantType.getEncoding());

    auto quantizedInput =
        rewriter.create<ttir::QuantizeOp>(getLoc(), newType, dequantVal);

    // Update operands.
    if (lhsElemQ) {
      rhs = quantizedInput;
    } else {
      lhs = quantizedInput;
      lhsElemQ = quantElemQ;
    }
  }
  // Now both values are quantized (and are equivalent).
  RankedTensorType oldType = mlir::cast<RankedTensorType>(getType());
  RankedTensorType newResultType = RankedTensorType::get(
      oldType.getShape(), lhsElemQ, oldType.getEncoding());

  // Emit new AddOp with quantized types.
  auto newAdd = rewriter.create<ttir::AddOp>(getLoc(), newResultType, lhs, rhs);
  return newAdd.getOperation();
}

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

// BitwiseXorOp canonicalization
void mlir::tt::ttir::BitwiseXorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // x ^ x == 0
  patterns.add(
      +[](mlir::tt::ttir::BitwiseXorOp op, mlir::PatternRewriter &rewriter) {
        if (op.getLhs() != op.getRhs()) {
          return mlir::failure();
        }

        mlir::RankedTensorType tensorType = op.getResult().getType();
        auto elementType = tensorType.getElementType();
        Attribute zeroAttr;
        if (mlir::isa<mlir::FloatType>(elementType)) {
          zeroAttr = mlir::FloatAttr::get(elementType, 0.0);
        } else if (mlir::isa<mlir::IntegerType>(elementType)) {
          zeroAttr = mlir::IntegerAttr::get(elementType, 0);
        } else {
          return mlir::failure();
        }
        auto resultType = mlir::SplatElementsAttr::get(tensorType, zeroAttr);

        rewriter.replaceOpWithNewOp<ttir::ConstantOp>(
            op, op->getOperand(0).getType(), resultType);
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// BroadcastOp folder
::mlir::OpFoldResult mlir::tt::ttir::BroadcastOp::fold(FoldAdaptor adaptor) {
  // If the input doesn't change the shape, we can fold the operation.
  if (llvm::all_of(getBroadcastDimensions(),
                   [](const int32_t dim) { return dim == 1; })) {
    return getInput();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ClampScalarOp
//===----------------------------------------------------------------------===//

// ClampScalarOp verifier
::mlir::LogicalResult mlir::tt::ttir::ClampScalarOp::verify() {
  const RankedTensorType inputTensorType =
      mlir::cast<RankedTensorType>(getInput().getType());

  const RankedTensorType outputTensorType =
      mlir::cast<RankedTensorType>(getResult().getType());

  if (inputTensorType != outputTensorType) {
    return emitOpError("input and output must have same shape.");
  }

  return success();
}

// ClampScalarOp canonicalization
void mlir::tt::ttir::ClampScalarOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Fold two consecutive ClampScalarOp into a single one with tighter bounds
  patterns.add(+[](mlir::tt::ttir::ClampScalarOp op,
                   mlir::PatternRewriter &rewriter) {
    auto producerOp = op.getInput().getDefiningOp<ClampScalarOp>();
    if (!producerOp || !producerOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Get the min/max values from both ops
    auto producerMin = producerOp.getMin();
    auto producerMax = producerOp.getMax();
    auto consumerMin = op.getMin();
    auto consumerMax = op.getMax();

    // Calculate the tightest bounds
    auto newMin = std::max(producerMin, consumerMin);
    auto newMax = std::min(producerMax, consumerMax);

    // Replace with a single ClampScalarOp with the new bounds
    rewriter.replaceOpWithNewOp<ClampScalarOp>(
        op, op.getResult().getType(), producerOp.getInput(), newMin, newMax);
    return mlir::success();
  });

  // Fold clamp with min=0 and max=6 to relu6
  patterns.add(+[](mlir::tt::ttir::ClampScalarOp op,
                   mlir::PatternRewriter &rewriter) {
    auto minVal = op.getMin();
    auto maxVal = op.getMax();

    if (minVal.convertToFloat() == 0.0f && maxVal.convertToFloat() == 6.0f) {
      rewriter.replaceOpWithNewOp<ttir::Relu6Op>(op, op.getResult().getType(),
                                                 op.getInput());
      return mlir::success();
    }

    return mlir::failure();
  });
}

//===----------------------------------------------------------------------===//
// LogicalRightShiftOp
//===----------------------------------------------------------------------===//

// LogicalRightShiftOp verifier
::mlir::LogicalResult mlir::tt::ttir::LogicalRightShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that left operand (value to be shifted) has integer element type.
  auto lhsElemType = lhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  // Check that right operand (shift amount) has integer element type.
  auto rhsElemType = rhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  // Check that output has integer element type.
  auto outputElemType = outputTensorType.getElementType();
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
::mlir::LogicalResult mlir::tt::ttir::LogicalLeftShiftOp::verify() {
  RankedTensorType lhsTensorType = getLhs().getType();
  RankedTensorType rhsTensorType = getRhs().getType();
  RankedTensorType outputTensorType = getResult().getType();

  // Check that left operand (value to be shifted) has integer element type.
  auto lhsElemType = lhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(lhsElemType)) {
    return emitOpError()
           << "Left operand element type must be integer, but got "
           << lhsElemType;
  }

  // Check that right operand (shift amount) has integer element type.
  auto rhsElemType = rhsTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(rhsElemType)) {
    return emitOpError()
           << "Right operand element type must be integer, but got "
           << rhsElemType;
  }

  // Check that output has integer element type.
  auto outputElemType = outputTensorType.getElementType();
  if (!mlir::isa<mlir::IntegerType>(outputElemType)) {
    return emitOpError() << "Output element type must be integer, but got "
                         << outputElemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClampTensorOp
//===----------------------------------------------------------------------===//

// ClampTensorOp verifier
::mlir::LogicalResult mlir::tt::ttir::ClampTensorOp::verify() {
  llvm::ArrayRef<int64_t> minShape = getMin().getType().getShape();

  llvm::ArrayRef<int64_t> outputShape = getResult().getType().getShape();

  llvm::SmallVector<int64_t, 4> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(minShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError("Min attribute shape (" +
                       ttmlir::utils::join(minShape, ",") +
                       ") cannot be broadcasted to output shape (" +
                       ttmlir::utils::join(outputShape, ",") + ").");
  }

  llvm::ArrayRef<int64_t> maxShape = getMax().getType().getShape();
  if (!mlir::OpTrait::util::getBroadcastedShape(maxShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError("Max attribute shape (" +
                       ttmlir::utils::join(maxShape, ",") +
                       ") cannot be broadcasted to output shape (" +
                       ttmlir::utils::join(outputShape, ",") + ").");
  }

  return success();
}

// Helper function to extract constant value.
static mlir::FloatAttr getConstantValue(mlir::Value value) {
  mlir::Operation *op = value.getDefiningOp();
  while (mlir::isa_and_present<mlir::tt::ttir::BroadcastOp,
                               mlir::tt::ttir::ReshapeOp,
                               mlir::tt::ttir::TypecastOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }

  auto fullOp = mlir::dyn_cast_if_present<mlir::tt::ttir::FullOp>(op);
  if (!fullOp) {
    return {};
  }

  mlir::Attribute fillValueAttr = fullOp.getFillValueAttr();

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(fillValueAttr)) {
    return floatAttr;
  }
  if (auto integerAttr = mlir::dyn_cast<mlir::IntegerAttr>(fillValueAttr)) {
    return mlir::FloatAttr::get(
        mlir::Float32Type::get(integerAttr.getContext()),
        static_cast<double>(integerAttr.getValue().getSExtValue()));
  }
  return {};
}

// ClampTensorOp canonicalization
void mlir::tt::ttir::ClampTensorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(
      +[](mlir::tt::ttir::ClampTensorOp op, mlir::PatternRewriter &rewriter) {
        RankedTensorType outputType = op.getResult().getType();

        FloatAttr minValue = getConstantValue(op.getMin());
        FloatAttr maxValue = getConstantValue(op.getMax());
        if (minValue && maxValue) {
          rewriter.replaceOpWithNewOp<ttir::ClampScalarOp>(
              op, outputType, op.getInput(), minValue, maxValue);

          return success();
        }

        if (outputType.getShape() == op.getMin().getType().getShape() &&
            outputType.getShape() == op.getMax().getType().getShape()) {
          return failure();
        }

        Location loc = op->getLoc();
        mlir::Value minTensor;
        LogicalResult legalityResult = ttir::utils::broadcastValue(
            rewriter, op.getMin(), outputType, minTensor, loc,
            /*frontUnsqueeze=*/false);
        assert(legalityResult.succeeded() &&
               "Min attribute cannot be broadcasted to provided dimensions.");

        mlir::Value maxTensor;
        legalityResult = ttir::utils::broadcastValue(rewriter, op.getMax(),
                                                     outputType, maxTensor, loc,
                                                     /*frontUnsqueeze=*/false);
        assert(legalityResult.succeeded() &&
               "Max attribute cannot be broadcasted to provided dimensions.");

        rewriter.replaceOpWithNewOp<ttir::ClampTensorOp>(
            op, outputType, op.getInput(), minTensor, maxTensor);
        return success();
      });
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ArangeOp::verify() {
  int64_t start = getStart();
  int64_t end = getEnd();
  int64_t step = getStep();

  if (step == 0) {
    return emitOpError("Step value cannot be zero");
  }

  int64_t numValues = (end - start) / step;

  if (numValues <= 0) {
    return emitOpError() << "Invalid range: start=" << start << ", end=" << end
                         << ", step=" << step;
  }

  if (numValues != getType().getDimSize(getArangeDimension())) {
    return emitOpError() << "Output tensor shape must be " << numValues
                         << " at dim " << getArangeDimension()
                         << " (since start=" << start << ", end=" << end
                         << ", step=" << step << "), but got "
                         << getType().getDimSize(getArangeDimension());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RandOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::RandOp::verify() {
  auto dtype = getDtype();
  auto outputType = getResult().getType().getElementType();

  if (dtype != outputType) {
    return emitOpError()
           << "dtype does not match with output tensor type [dtype = " << dtype
           << ", output tensor type = " << outputType << "].";
  }

  float low = getLow().convertToFloat();
  float high = getHigh().convertToFloat();
  if (low >= high) {
    return emitOpError() << "'low' value must be < 'high' value.";
  }

  llvm::SmallVector<int64_t> sizeVec;
  for (auto size : getSize()) {
    sizeVec.push_back(mlir::cast<mlir::IntegerAttr>(size).getInt());
  }
  if (!llvm::equal(getResult().getType().getShape(), sizeVec)) {
    return emitOpError()
           << "Size argument does not match with output tensor shape. [Size = "
           << getSize() << ", output tensor shape = ("
           << getResult().getType().getShape() << ")].";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DropoutOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::DropoutOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getResult().getType();

  if (inputType != outputType) {
    return emitOpError() << "Input tensor type does not match with output "
                            "tensor type. [Input tensor type = "
                         << inputType << ", output tensor type = " << outputType
                         << "].";
  }

  float prob = getProb().convertToFloat();
  if (prob < 0.0f || prob > 1.0f) {
    return emitOpError() << "Probability must be in range [0.0, 1.0], but got "
                         << prob;
  }

  float scale = getScale().convertToFloat();
  if (scale <= 0.0f) {
    return emitOpError() << "Scale must be positive, but got " << scale;
  }

  int64_t rank = inputType.getRank();
  if (rank < 2 || rank > 4) {
    return emitOpError() << "Input tensor rank must be 2, 3, or 4, but got "
                         << rank;
  }

  return success();
}
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// ConstantOp folder
::mlir::OpFoldResult mlir::tt::ttir::ConstantOp::fold(FoldAdaptor) {
  return getValueAttr();
}

// ConstantOp canonicalization
void mlir::tt::ttir::ConstantOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {

  // Canonicalize ConstantOp to FullOp when the value is a splat value (i.e. all
  // elements are the same).
  patterns.add(+[](mlir::tt::ttir::ConstantOp op,
                   mlir::PatternRewriter &rewriter) {
    auto valueAttr = op.getValueAttr();
    if (!valueAttr.isSplat()) {
      return failure();
    }

    mlir::Attribute fillValueAttr;
    if (auto integerType =
            mlir::dyn_cast<mlir::IntegerType>(valueAttr.getElementType())) {
      auto fillValue = valueAttr.getSplatValue<llvm::APInt>();
      if (integerType.isSigned()) {
        fillValueAttr = rewriter.getI32IntegerAttr(fillValue.getSExtValue());
      } else {
        fillValueAttr = rewriter.getI32IntegerAttr(fillValue.getZExtValue());
      }
    } else if (valueAttr.getElementType().isIntOrFloat()) {
      auto fillValue = valueAttr.getSplatValue<mlir::APFloat>();
      fillValueAttr = rewriter.getF32FloatAttr(fillValue.convertToDouble());
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::FullOp>(
        op, op.getType(),
        rewriter.getDenseI32ArrayAttr(
            llvm::to_vector_of<int32_t>(op.getType().getShape())),
        fillValueAttr);

    return success();
  });
}

::mlir::LogicalResult mlir::tt::ttir::ConstantOp::verify() {
  if (!isa<DenseResourceElementsAttr, DenseElementsAttr>(getValue())) {
    return emitOpError("value attribute must be one of "
                       "DenseResourceElementsAttr or DenseElementsAttr.");
  }

  if (!getValue().getElementType().isIntOrFloat()) {
    return emitOpError("value attribute must be of int or float type.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

// GetDimensionSizeOp verification
::mlir::LogicalResult mlir::tt::ttir::GetDimensionSizeOp::verify() {
  RankedTensorType inputTensorType = getOperand().getType();

  int64_t dimensionIndex = getDimension();

  if (dimensionIndex >=
      static_cast<int64_t>(inputTensorType.getShape().size())) {
    return failure();
  };

  return success();
}

// GetDimensionSizeOp folder
::mlir::OpFoldResult
mlir::tt::ttir::GetDimensionSizeOp::fold(FoldAdaptor adaptor) {
  RankedTensorType inputTensorType = getOperand().getType();
  uint32_t dimensionIndex = getDimension();
  uint32_t dimSize = inputTensorType.getShape()[dimensionIndex];

  auto resultElType = IntegerType::get(
      getContext(), 32, IntegerType::SignednessSemantics::Unsigned);
  auto resultType = RankedTensorType::get(/*shape=*/{1}, resultElType);
  return mlir::DenseElementsAttr::get<uint32_t>(resultType, dimSize);
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

bool mlir::tt::ttir::Conv2dOp::isQuantizedRewriteFavorable(
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // Convolution op requires both input and weight to be quantized.
  // Bias (if present) can be float - it will be quantized in
  // rewriteWithQuantizedInputs.
  assert(sourceOperands.size() >= 2 &&
         "Conv2dOp should have at least two operands (input and weight).");
  // Only check input and weight operands. If bias exists, it can remain float.
  size_t numToCheck = getBias() ? 2 : sourceOperands.size();
  for (size_t i = 0; i < numToCheck; ++i) {
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(sourceOperands[i].getType());
    if (!type) {
      return false;
    }
    auto qType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType());
    if (!qType || qType.getStorageType().getIntOrFloatBitWidth() != 8) {
      return false;
    }
  }
  return true;
}

mlir::Operation *mlir::tt::ttir::Conv2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter, mlir::ArrayRef<Value> sourceOperands) {
  // rewrite the convolution op to be quantized.
  // create the output quantized type, whose scale is input * weight.
  // If bias is provided, output storage type is i8 (bias addition handles
  // requantization internally). Otherwise, storage type is i32 (accumulator).
  auto storageType =
      getBias()
          ? IntegerType::get(rewriter.getContext(), 8, IntegerType::Signed)
          : IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
  auto quantInputType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[0].getType())
          .getElementType());
  auto quantWeightType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[1].getType())
          .getElementType());
  auto oldConvOutputType = cast<RankedTensorType>(getResult().getType());

  // Pass back axes needed for computation of output scale and zero point.
  // Use the channel dimension from the op's attributes.
  const int64_t outFeatAxis = getChannelDim();
  // Weight is always in OIHW format, so output channel is at dim 0.
  const int64_t weightOcAxis = 0;
  const int64_t ocSize = oldConvOutputType.getDimSize(outFeatAxis);

  mlir::quant::QuantizedType quantOutputType =
      mlir::tt::ttir::utils::computeOutputScalesAndZeroPoint(
          quantInputType, quantWeightType, storageType, getLoc(), outFeatAxis,
          weightOcAxis, ocSize);
  if (!quantOutputType) {
    return nullptr;
  }
  auto quantConvOutputType =
      quantOutputType.castFromExpressedType(oldConvOutputType.getElementType());
  if (!quantConvOutputType) {
    return nullptr;
  }
  RankedTensorType newType =
      RankedTensorType::get(oldConvOutputType.getShape(), quantConvOutputType,
                            oldConvOutputType.getEncoding());
  // we need to quantize the bias if it is provided, the new bias type should be
  // the same as the quantconvoutput type
  auto quantBias = getBias();
  if (quantBias) {
    RankedTensorType quantBiasType = RankedTensorType::get(
        getBias().getType().getShape(), quantConvOutputType,
        getBias().getType().getEncoding());
    quantBias = rewriter.create<mlir::tt::ttir::QuantizeOp>(
        getLoc(), quantBiasType, quantBias);
  }
  auto quantConv = rewriter.create<mlir::tt::ttir::Conv2dOp>(
      getLoc(), newType, sourceOperands[0], sourceOperands[1], quantBias,
      getStrideAttr(), getPaddingAttr(), getDilationAttr(),
      rewriter.getI32IntegerAttr(getGroups()), getBatchDimAttr(),
      getHeightDimAttr(), getWidthDimAttr(), getChannelDimAttr(),
      /*flattenedCompatInfo=*/nullptr);

  return quantConv.getOperation();
}

// Conv2dOp verification
::mlir::LogicalResult mlir::tt::ttir::Conv2dOp::verify() {
  // Verify tensor ranks.
  if (verification_utils::verifyTensorRanks<Conv2dOp, true>(this).failed()) {
    return mlir::failure();
  }

  auto flatInfo = getFlattenedCompatInfoAttr();
  if (flatInfo &&
      flatInfo.getBatchSize() * flatInfo.getInputHeight() *
              flatInfo.getInputWidth() !=
          getInput().getType().getDimSize(verification_utils::FLATTENED_DIM)) {
    int64_t expectedSize = flatInfo.getBatchSize() * flatInfo.getInputHeight() *
                           flatInfo.getInputWidth();
    int64_t actualSize =
        getInput().getType().getDimSize(verification_utils::FLATTENED_DIM);
    return emitOpError()
           << "The input tensor's flattened dimension (" << actualSize
           << ") does not match the product of batch_size * input_height * "
              "input_width from FlattenedCompatInfo ("
           << flatInfo.getBatchSize() << " * " << flatInfo.getInputHeight()
           << " * " << flatInfo.getInputWidth() << " = " << expectedSize
           << ").";
  }

  auto [inputDims, weightDims, biasDims] =
      verification_utils::conv2d::getConv2dInputDims(this);
  verification_utils::OutputTensorDims outputDims =
      verification_utils::conv2d::getConv2dOutputDims(this);
  auto expectedParams = verification_utils::conv2d::getConv2dParams(this);
  if (auto error = expectedParams.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }
  verification_utils::conv2d::Conv2dParams params = *expectedParams;

  if (verification_utils::conv2d::verifyConv2dParams(this, params).failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv2d::verifyConv2dInputDims(
          this, inputDims, weightDims, biasDims, params)
          .failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv2d::verifyOutputDimensions(
          this, inputDims, weightDims, biasDims, outputDims, params)
          .failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

// Get number of output channels
int64_t mlir::tt::ttir::Conv2dOp::getOutputChannelSize() {
  RankedTensorType weightTy = getWeight().getType();
  return weightTy.getShape()[0];
}

// Verify that bias dimensions are compatible with conv2d operation
bool mlir::tt::ttir::Conv2dOp::isBiasCompatible(llvm::ArrayRef<int64_t> bias) {
  return bias[0] == 1 && bias[1] == 1 && bias[2] == 1 &&
         bias[3] == getOutputChannelSize();
}

//===----------------------------------------------------------------------===//
// Conv3dOp
//===----------------------------------------------------------------------===//

// Conv3dOp verification
::mlir::LogicalResult mlir::tt::ttir::Conv3dOp::verify() {
  // Verify tensor ranks
  if (this->getInput().getType().getRank() != 5) {
    return this->emitOpError("input must be a 5D tensor");
  }

  if (this->getWeight().getType().getRank() != 5) {
    return this->emitOpError("weight must be a 5D tensor");
  }

  // Bias is optional
  if (this->getBias() && this->getBias().getType().getRank() != 5) {
    return this->emitOpError(
        "bias must be a 5D tensor with shape (1,1,1,1,C_out)");
  }

  if (this->getResult().getType().getRank() != 5) {
    return this->emitOpError("output must be a 5D tensor");
  }

  auto [inputDims, weightDims, biasDims] =
      verification_utils::conv3d::getConv3dInputDims(this);

  verification_utils::conv3d::OutputTensorDims3d outputDims =
      verification_utils::conv3d::getConv3dOutputDims(this);

  auto expectedParams = verification_utils::conv3d::getConv3dParams(this);

  if (auto error = expectedParams.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }

  verification_utils::conv3d::Conv3dParams params = *expectedParams;
  if (verification_utils::conv3d::verifyConv3dParams(this, params).failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv3d::verifyConv3dInputDims(
          this, inputDims, weightDims, biasDims, params)
          .failed()) {
    return mlir::failure();
  }

  if (verification_utils::conv3d::verifyOutputDimensions(
          this, inputDims, weightDims, biasDims, outputDims, params)
          .failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Quantize ops
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

// Common verifier for all Quantize ops.
static ::mlir::LogicalResult verifyQuantizeOpCommon(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
    ::mlir::RankedTensorType inputType, ::mlir::RankedTensorType outputType,
    std::optional<uint32_t> axis = std::nullopt, bool isUnrolled = false) {
  // Sanity check to make sure that input rank matches the rank of the output
  // tensor.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank of " << inputType.getRank()
                         << " does not match the output tensor rank of "
                         << outputType.getRank();
  }

  // Shapes of input and output of a quantize operation must be the same.
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError() << "Output tensor shape ("
                         << ttmlir::utils::join(outputType.getShape(), ",") +
                                ") must match the inferred shape: (" +
                                ttmlir::utils::join(inputType.getShape(), ",") +
                                ")";
  }

  if (!isUnrolled) {
    return ::mlir::success();
  }

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
      llvm::ArrayRef<int64_t> zeroPoints = quantPerAxisType.getZeroPoints();
      int64_t min = quantPerAxisType.getStorageTypeMin();
      int64_t max = quantPerAxisType.getStorageTypeMax();
      for (int64_t curZeroPoint : zeroPoints) {
        if (auto result =
                verifyZeroPointInRange(emitOpError, curZeroPoint, min, max,
                                       quantPerAxisType.getStorageType());
            failed(result)) {
          return result;
        }
      }
    }
    if (auto quantType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
      // Verify that the zero point is in the range of the storage type
      // (per-tensor). This aligns with the frontends' behavior.
      int64_t curZeroPoint = quantType.getZeroPoint();
      int64_t min = quantType.getStorageTypeMin();
      int64_t max = quantType.getStorageTypeMax();
      return verifyZeroPointInRange(emitOpError, curZeroPoint, min, max,
                                    quantType.getStorageType());
    }
  }

  return ::mlir::success();
}

// QuantizeOp verification.
::mlir::LogicalResult mlir::tt::ttir::QuantizeOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

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
                                getInput().getType(), getType(),
                                /*axis=*/std::nullopt, /*isUnrolled=*/false);
}

// QuantizeUnrolledOp verification.
::mlir::LogicalResult mlir::tt::ttir::QuantizeUnrolledOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

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
                                getInput().getType(), getType(), getAxis(),
                                /*isUnrolled=*/true);
}

// DequantizeOp verification.
::mlir::LogicalResult mlir::tt::ttir::DequantizeOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

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
                                getInput().getType(), getType(),
                                /*axis=*/std::nullopt, /*isUnrolled=*/false);
}

// DequantizeUnrolledOp verification.
::mlir::LogicalResult mlir::tt::ttir::DequantizeUnrolledOp::verify() {
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
                                inputTensorType, outputTensorType, getAxis(),
                                /*isUnrolled=*/true);
}

// RequantizeOp folder for identity requantize.
::mlir::OpFoldResult mlir::tt::ttir::RequantizeOp::fold(FoldAdaptor adaptor) {
  // if types of input and output are equivalent, return input.
  if (getInput().getType() == getType()) {
    return getInput();
  }
  return nullptr;
}

// RequantizeOp verification.
::mlir::LogicalResult mlir::tt::ttir::RequantizeOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
    return emitOpError() << "Input element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << inputElemType;
  }

  if (!mlir::isa<mlir::quant::UniformQuantizedType,
                 mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
    return emitOpError() << "Output element type must be UniformQuantizedType "
                            "or UniformQuantizedPerAxisType, but got "
                         << outputElemType;
  }

  return verifyQuantizeOpCommon([&]() { return emitOpError(); },
                                getInput().getType(), getType(),
                                /*axis=*/std::nullopt, /*isUnrolled=*/false);
}

// RequantizeUnrolledOp verification.
::mlir::LogicalResult mlir::tt::ttir::RequantizeUnrolledOp::verify() {
  auto inputElemType = getInput().getType().getElementType();
  auto outputElemType = getResult().getType().getElementType();

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
                                getInput().getType(), getType(),
                                /*axis=*/getAxis(), /*isUnrolled=*/true);
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

bool mlir::tt::ttir::ConvTranspose2dOp::isQuantizedRewriteFavorable(
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // convolution op currently requires both input and weight to be quantized
  // TODO(anuragsingh): enable float bias support
  assert(sourceOperands.size() == 2 &&
         "Quantized ConvolutionOp should have two operands (only input and "
         "weight).");
  return llvm::all_of(sourceOperands, [](mlir::Value val) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    if (!type) {
      return false;
    }
    auto qType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType());
    return qType && qType.getStorageType().getIntOrFloatBitWidth() == 8;
  });
}

mlir::Operation *mlir::tt::ttir::ConvTranspose2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter, mlir::ArrayRef<Value> sourceOperands) {
  // rewrite the convolution op to be quantized.
  // create the output quantized type, whose scale is input * weight and
  // storage type is i32.
  auto storageType =
      IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
  auto quantInputType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[0].getType())
          .getElementType());
  auto quantWeightType = mlir::cast<mlir::quant::QuantizedType>(
      mlir::cast<RankedTensorType>(sourceOperands[1].getType())
          .getElementType());
  auto oldConvOutputType = cast<RankedTensorType>(getResult().getType());

  // Pass back axes needed for computation of output scale and zero point.
  // ConvTranspose2d op has output in NHWC format and weight in OIHW format.
  const int64_t outFeatAxis = 3;
  const int64_t weightOcAxis = 0;
  const int64_t ocSize = oldConvOutputType.getDimSize(outFeatAxis);

  mlir::quant::QuantizedType quantOutputType =
      mlir::tt::ttir::utils::computeOutputScalesAndZeroPoint(
          quantInputType, quantWeightType, storageType, getLoc(), outFeatAxis,
          weightOcAxis, ocSize);
  if (!quantOutputType) {
    return nullptr;
  }
  auto quantConvOutputType =
      quantOutputType.castFromExpressedType(oldConvOutputType.getElementType());
  if (!quantConvOutputType) {
    return nullptr;
  }
  RankedTensorType newType =
      RankedTensorType::get(oldConvOutputType.getShape(), quantConvOutputType,
                            oldConvOutputType.getEncoding());
  auto quantConv = rewriter.create<mlir::tt::ttir::ConvTranspose2dOp>(
      getLoc(), newType, sourceOperands[0], sourceOperands[1], getBias(),
      getStrideAttr(), getPaddingAttr(), getOutputPaddingAttr(),
      getDilationAttr(), getGroupsAttr(), /*flattenedCompatInfo=*/nullptr);

  return quantConv.getOperation();
}

// ConvTranspose2dOp verification
mlir::LogicalResult mlir::tt::ttir::ConvTranspose2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getType();
  std::optional<mlir::RankedTensorType> bias =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  auto flatInfo = getFlattenedCompatInfoAttr();
  if (flatInfo &&
      flatInfo.getBatchSize() * flatInfo.getInputHeight() *
              flatInfo.getInputWidth() !=
          getInput().getType().getDimSize(verification_utils::FLATTENED_DIM)) {
    int64_t expectedSize = flatInfo.getBatchSize() * flatInfo.getInputHeight() *
                           flatInfo.getInputWidth();
    int64_t actualSize =
        getInput().getType().getDimSize(verification_utils::FLATTENED_DIM);
    return emitOpError()
           << "The input tensor's flattened dimension (" << actualSize
           << ") does not match the product of batch_size * input_height * "
              "input_width from FlattenedCompatInfo ("
           << flatInfo.getBatchSize() << " * " << flatInfo.getInputHeight()
           << " * " << flatInfo.getInputWidth() << " = " << expectedSize
           << ").";
  }

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

  int64_t inputBatchSize = inputType.getDimSize(0);
  int64_t outputBatchSize = outputType.getDimSize(0);
  if (flatInfo) {
    inputBatchSize = flatInfo.getBatchSize();
    outputBatchSize = flatInfo.getBatchSize();
  }

  if (inputBatchSize != outputBatchSize) {
    return emitOpError("Batch size of input and output tensors must match");
  }

  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(getStride());
  if (auto error = stride.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for stride";
  }
  if (stride->first < 1 || stride->second < 1) {
    return emitOpError("Stride values must be greater than 0");
  }

  auto padding = ttmlir::utils::getQuadrupleOfInteger<int32_t>(getPadding());
  if (auto error = padding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for padding";
  }

  auto [paddingTop, paddingLeft, paddingBottom, paddingRight] = *padding;
  if (paddingTop < 0 || paddingBottom < 0 || paddingLeft < 0 ||
      paddingRight < 0) {
    return emitOpError("Padding values must be greater or equal than 0");
  }
  int32_t verticalPadding = paddingTop + paddingBottom;
  int32_t horizontalPadding = paddingLeft + paddingRight;

  auto outputPadding =
      ttmlir::utils::getPairOfInteger<int32_t>(getOutputPadding());
  if (auto error = outputPadding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error))
                         << " for output padding";
  }
  if (outputPadding->first < 0 || outputPadding->second < 0) {
    return emitOpError("Output padding values must be greater or equal than 0");
  }

  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(getDilation());
  if (auto error = dilation.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for dilation";
  }
  if (dilation->first < 1 || dilation->second < 1) {
    return emitOpError("Dilation values must be greater than 0");
  }

  llvm::ArrayRef<std::int64_t> kernelShape = weightType.getShape();

  int32_t inputChannels = inputType.getDimSize(getChannelDim());
  int32_t outputChannels = outputType.getDimSize(getChannelDim());
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
    if (bias->getDimSize(getChannelDim()) != outputChannels) {
      return emitOpError() << "Mismatch in bias tensor dimensions. "
                           << "Bias tensor has "
                           << bias->getDimSize(getChannelDim()) << " channels, "
                           << "but the output tensor has " << outputChannels
                           << " channels.";
    }
  }

  int32_t kernelHeight = kernelShape[2];
  int32_t kernelWidth = kernelShape[3];

  int32_t Hin = inputType.getDimSize(getHeightDim());
  int32_t Win = inputType.getDimSize(getWidthDim());
  if (flatInfo) {
    Hin = flatInfo.getInputHeight();
    Win = flatInfo.getInputWidth();
  }

  int32_t expectedHOut = (Hin - 1) * stride->first - verticalPadding +
                         dilation->first * (kernelHeight - 1) +
                         outputPadding->first + 1;
  int32_t expectedWOut = (Win - 1) * stride->second - horizontalPadding +
                         dilation->second * (kernelWidth - 1) +
                         outputPadding->second + 1;
  if (expectedHOut < 0 || expectedWOut < 0) {
    return emitOpError() << "Given input size per channel: (" << Hin << " x "
                         << Win << "). "
                         << "Calculated output size per channel: ("
                         << expectedHOut << " x " << expectedWOut << "). "
                         << "Output size is too small";
  }

  int32_t HOut = outputType.getDimSize(getHeightDim());
  int32_t WOut = outputType.getDimSize(getWidthDim());
  if (!flatInfo && (HOut != expectedHOut || WOut != expectedWOut)) {
    return emitOpError() << "Mismatch between expected output size per channel "
                            "and got output tensor dimensions. "
                         << "Expected: (" << expectedHOut << " x "
                         << expectedWOut << "), "
                         << "got: (" << HOut << " x " << WOut << ").";
  }

  if (flatInfo &&
      inputBatchSize * expectedHOut * expectedWOut !=
          outputType.getDimSize(verification_utils::FLATTENED_DIM)) {
    return emitOpError() << "Mismatch between expected flattened NHW dim size. "
                         << "Expected: "
                         << inputBatchSize * expectedHOut * expectedWOut << ", "
                         << "got: "
                         << outputType.getDimSize(
                                verification_utils::FLATTENED_DIM)
                         << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pooling helper functions
//===----------------------------------------------------------------------===//

// Checks if a AvgPool2dOp or MaxPool2dOp operation is an identity operation.
// Identity operations can be folded away when kernel=[1,1], stride=[1,1],
// dilation=[1,1], and padding=[0,0,0,0].
template <typename Pool2dOp>
static bool isIdentityPool2d(Pool2dOp op) {
  auto kernel = ttmlir::utils::getPairOfInteger<int32_t>(op.getKernel());
  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(op.getStride());
  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(op.getDilation());
  auto padding = ttmlir::utils::getQuadrupleOfInteger<int32_t>(op.getPadding());

  auto tupleToArray = [](const auto &t) {
    return std::apply([](auto... args) { return std::array{args...}; }, t);
  };

  return kernel && stride && dilation && padding &&
         llvm::all_of(tupleToArray(*kernel),
                      [](int32_t v) { return v == 1; }) &&
         llvm::all_of(tupleToArray(*stride),
                      [](int32_t v) { return v == 1; }) &&
         llvm::all_of(tupleToArray(*dilation),
                      [](int32_t v) { return v == 1; }) &&
         llvm::all_of(tupleToArray(*padding), [](int32_t v) { return v == 0; });
}

//===----------------------------------------------------------------------===//
// Generic Pool2dOp verification
//===----------------------------------------------------------------------===//

template <typename Pool2dOp>
static mlir::LogicalResult verifyPooling2dOp(Pool2dOp *op) {

  // Verify tensor ranks.
  if (verification_utils::verifyTensorRanks(op).failed()) {
    return mlir::failure();
  }

  // Verify flattened compatibility info if present.
  if (mlir::failed(verification_utils::pool2d::verifyFlattenedCompatInfo(op))) {
    return mlir::failure();
  }

  // Get input and output dimensions with flattened support.
  verification_utils::InputTensorDims inputDims =
      verification_utils::pool2d::getPool2dInputDims(op);
  verification_utils::OutputTensorDims outputDims =
      verification_utils::pool2d::getPool2dOutputDims(op);
  auto expectedParams = verification_utils::pool2d::getPool2dParams(op);
  if (auto error = expectedParams.takeError()) {
    return op->emitOpError() << llvm::toString(std::move(error));
  }
  verification_utils::pool2d::Pool2dParams params = *expectedParams;

  // Verify pooling parameters.
  if (mlir::failed(
          verification_utils::pool2d::verifyPool2dParams(op, params))) {
    return mlir::failure();
  }

  // Verify input dimensions constraints.
  if (mlir::failed(verification_utils::pool2d::verifyPool2dInputDims(
          op, inputDims, params))) {
    return mlir::failure();
  }

  // Verify output dimensions match expected calculations.
  if (mlir::failed(verification_utils::pool2d::verifyPool2dOutputDims(
          op, inputDims, outputDims, params))) {
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AvgPool2dOp
//===----------------------------------------------------------------------===//

// AvgPool2dOp verification.
::mlir::LogicalResult mlir::tt::ttir::AvgPool2dOp::verify() {
  return verifyPooling2dOp(this);
}

// Folds AvgPool2dOp when it is an identity operation.
::mlir::OpFoldResult mlir::tt::ttir::AvgPool2dOp::fold(FoldAdaptor adaptor) {
  if (isIdentityPool2d(*this)) {
    return getInput();
  }
  return {};
}

// Rewrites operation with quantization, if possible.
::mlir::Operation *mlir::tt::ttir::AvgPool2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter,
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // Unlike MaxPool2dOp which only compares values (scale-invariant), AvgPool2d
  // performs a calculation which causes precision loss due to integer division
  // truncation and scale parameter mismatch.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

// MaxPool2dOp verification.
::mlir::LogicalResult mlir::tt::ttir::MaxPool2dOp::verify() {
  return verifyPooling2dOp(this);
}

// Folds MaxPool2dOp when it is an identity operation.
::mlir::OpFoldResult mlir::tt::ttir::MaxPool2dOp::fold(FoldAdaptor adaptor) {
  if (isIdentityPool2d(*this)) {
    return getInput();
  }
  return {};
}

// Rewrites operation with quantization, if possible.
::mlir::Operation *mlir::tt::ttir::MaxPool2dOp::rewriteWithQuantizedInputs(
    mlir::PatternRewriter &rewriter,
    mlir::ArrayRef<mlir::Value> sourceOperands) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  using mlir::quant::UniformQuantizedPerAxisType;

  mlir::Value input = sourceOperands[0];
  if (mlir::dyn_cast<UniformQuantizedPerAxisType>(input.getType())) {
    return nullptr;
  }

  RankedTensorType outType =
      mlir::cast<RankedTensorType>(getResult().getType());

  RankedTensorType newOutType = RankedTensorType::get(
      outType.getShape(),
      mlir::cast<RankedTensorType>(input.getType()).getElementType(),
      outType.getEncoding());

  return rewriter
      .create<mlir::tt::ttir::MaxPool2dOp>(
          getLoc(), newOutType, input, getKernelAttr(), getStrideAttr(),
          getDilationAttr(), getPaddingAttr(), getCeilModeAttr())
      .getOperation();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// MaxPool2dWithIndicesOp
//===----------------------------------------------------------------------===//

// MaxPool2dWithIndicesOp verification
::mlir::LogicalResult mlir::tt::ttir::MaxPool2dWithIndicesOp::verify() {
  // First verify the pooling operation itself
  if (mlir::failed(verifyPooling2dOp(this))) {
    return mlir::failure();
  }

  // Verify that both results have the same shape
  auto pooledShape = this->getResult().getType().getShape();
  auto indicesShape = this->getResultIndices().getType().getShape();

  if (pooledShape != indicesShape) {
    return emitOpError("Pooled values and indices must have the same shape");
  }

  // Verify that indices have integer element type
  auto indicesElementType = this->getResultIndices().getType().getElementType();
  if (!indicesElementType.isInteger()) {
    return emitOpError("Indices result must have integer element type");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// ConcatOp verification
::mlir::LogicalResult mlir::tt::ttir::ConcatOp::verify() {
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
  int64_t dimSizeSum = firstTensor.getDimSize(dim);
  for (auto input : inputs.drop_front()) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    // Check if all inputs have the same rank.
    if (inputType.getRank() != firstTensorRank) {
      return emitOpError("All input tensors must have the same rank.");
    }

    // Check that dimensions (except `dim`) are the same.
    for (int64_t i = 0; i < firstTensorRank; ++i) {
      if (i == dim) {
        dimSizeSum += inputType.getDimSize(i);
        continue;
      }
      if (inputType.getDimSize(i) != firstTensor.getDimSize(i)) {
        return emitOpError() << "All input tensors must have the same "
                                "dimensions, except for dimension "
                             << dim << ".";
      }
    }
  }

  auto outputType = getType();
  if (outputType.getDimSize(dim) != dimSizeSum) {
    return emitOpError()
           << "Output tensor dimension " << dim
           << " does not match the sum of input tensor dimensions: "
           << outputType.getDimSize(dim) << " vs. " << dimSizeSum << ".";
  }

  return success();
}

// ConcatOp with single input is a no-op.
// Replace the op with input.
mlir::OpFoldResult foldUnitConcatOp(ttir::ConcatOp op) {
  mlir::ValueRange inputs = op.getInputs();
  if (inputs.size() == 1) {
    return inputs.front();
  }
  return nullptr;
}

// Empty tensor(s) act as neutral/identity element for ConcatOp.
// Remove empty tensors from ConcatOp operands.
mlir::OpFoldResult foldEmptyTensorsConcatOp(ttir::ConcatOp op) {
  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  mlir::ValueRange inputs = op.getInputs();
  int32_t dim = op.getDim();
  int32_t rank = outputType.getRank();
  int32_t adjustedDim = dim < 0 ? (dim + rank) : dim;
  llvm::SmallVector<mlir::Value> nonEmptyInputs;

  for (auto input : inputs) {
    auto shape = mlir::cast<RankedTensorType>(input.getType()).getShape();
    if (shape[adjustedDim] == 0) {
      continue;
    }
    nonEmptyInputs.push_back(input);
  }

  // No empty tensors to remove; Folding not applicable.
  if (inputs.size() == nonEmptyInputs.size()) {
    return nullptr;
  }

  // All inputs are empty tensors; returning first input (it can be any input).
  if (nonEmptyInputs.empty()) {
    return inputs.front();
  }

  // Update the operands with non empty inputs.
  op.getInputsMutable().assign(nonEmptyInputs);

  return op.getResult();
}

// ConcatOp Folder
mlir::OpFoldResult mlir::tt::ttir::ConcatOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldUnitConcatOp(*this)) {
    return foldResult;
  }
  if (auto foldResult = foldEmptyTensorsConcatOp(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// PadOp verification
::mlir::LogicalResult mlir::tt::ttir::PadOp::verify() {

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
// ReshapeOp
//===----------------------------------------------------------------------===//

// ReshapeOp verification
::mlir::LogicalResult mlir::tt::ttir::ReshapeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
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

  // Check that all dimensions are positive except for at most one -1.
  // Check that the non-negative dimensions match the output tensor shape.
  // Calculate the product of the known dimensions.
  for (int64_t i = 0; i < shapeSize; i++) {
    int64_t dimValue = mlir::cast<IntegerAttr>(shape[i]).getInt();

    if (dimValue == -1) {
      if (hasNegative) {
        return emitOpError("Shape attribute must have at most one -1 element");
      }
      hasNegative = true;
    } else {
      if (dimValue < 0) {
        return emitOpError(
            "All dimensions must be >= 0 except the one with -1");
      }

      // Ensure that the non-negative dimensions match the output tensor shape.
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
static mlir::OpFoldResult foldIdentityReshape(mlir::tt::ttir::ReshapeOp op) {
  if (op.getType() == op.getInput().getType()) {
    return op.getInput();
  }
  return nullptr;
}

// Back to back reshapes can be replaced with the final reshape.
static mlir::OpFoldResult foldConsecutiveReshape(mlir::tt::ttir::ReshapeOp op) {
  if (auto reshapeOperand =
          op.getInput().getDefiningOp<mlir::tt::ttir::ReshapeOp>()) {
    op.getInputMutable().assign(reshapeOperand.getInput());
    return op.getResult();
  }
  return nullptr;
}

// ReshapeOp folder
::mlir::OpFoldResult mlir::tt::ttir::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityReshape(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldConsecutiveReshape(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// RearrangeOp
//===----------------------------------------------------------------------===//

// RearrangeOp verification
::mlir::LogicalResult mlir::tt::ttir::RearrangeOp::verify() {
  llvm::StringRef patternStr = getPattern();
  llvm::SmallVector<llvm::StringRef> parts;
  patternStr.split(parts, "->");

  if (parts.size() != 2) {
    return emitOpError() << "pattern must contain exactly one '->' separator.";
  }

  mlir::FailureOr<AffineMap> failureOrInvMap = getInvPatternMap();
  if (failed(failureOrInvMap)) {
    return emitOpError() << "failed to parse pattern " << patternStr << ".";
  }

  auto invMap = *failureOrInvMap;
  if (getInput().getType().getRank() != invMap.getNumResults()) {
    return emitOpError() << "number of dimensions in the pattern's input ("
                         << invMap.getNumResults()
                         << ") must match the rank of the input tensor ("
                         << getInput().getType().getRank() << ").";
  }

  SmallVector<int64_t> expectedInputShape =
      ttmlir::utils::evalShape(invMap, getResult().getType().getShape());
  if (getInput().getType().getShape() !=
      ArrayRef<int64_t>(expectedInputShape)) {
    return emitOpError() << "input tensor shape ("
                         << ttmlir::utils::join(
                                getResult().getType().getShape(), ",")
                         << ") does not match the expected shape ("
                         << ttmlir::utils::join(expectedInputShape, ",")
                         << ").";
  }

  return success();
}

mlir::FailureOr<::mlir::AffineMap>
mlir::tt::ttir::RearrangeOp::getInvPatternMap(mlir::MLIRContext *context,
                                              StringRef pattern,
                                              ArrayRef<int64_t> shape) {
  // We need to write a routine to convert the pattern string to an affine map.
  // Example patterns:
  // >>> rearrange(images, 'b h w c -> b h w c').shape
  // (32, 30, 40, 3)
  //
  // # stacked and reordered axes to "b c h w" format
  // >>> rearrange(images, 'b h w c -> b c h w').shape
  // (32, 3, 30, 40)
  //
  // # concatenate images along height (vertical axis), 960 = 32 * 30
  // >>> rearrange(images, 'b h w c -> (b h) w c').shape
  // (960, 40, 3)
  //
  // # concatenated images along horizontal axis, 1280 = 32 * 40
  // >>> rearrange(images, 'b h w c -> h (b w) c').shape
  // (30, 1280, 3)
  //
  // # flattened each image into a vector, 3600 = 30 * 40 * 3
  // >>> rearrange(images, 'b h w c -> b (c h w)').shape
  // (32, 3600)
  //
  // # split each image into 4 smaller (top-left, top-right, bottom-left,
  // bottom-right), 128 = 32 * 2 * 2
  // >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2,
  // w1=2).shape (128, 15, 20, 3)
  //
  // # space-to-depth operation
  // >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2,
  // w1=2).shape (32, 15, 20, 12)

  llvm::SmallVector<llvm::StringRef> parts;
  pattern.split(parts, "->");

  assert(parts.size() == 2 &&
         "RearrangeOp pattern must contain exactly one '->' separator.");

  llvm::StringRef inputPattern = parts[0].trim();
  llvm::StringRef outputPattern = parts[1].trim();

  // Helper lambda to parse dimension names from a pattern.
  auto parseDims = [](llvm::StringRef pattern)
      -> llvm::SmallVector<llvm::SmallVector<llvm::StringRef>> {
    llvm::SmallVector<llvm::SmallVector<llvm::StringRef>> result;
    llvm::SmallVector<llvm::StringRef> currentGroup;
    bool inParens = false;
    size_t i = 0;

    while (i < pattern.size()) {
      char c = pattern[i];

      if (c == '(') {
        inParens = true;
        i++;
        continue;
      }

      if (c == ')') {
        if (!currentGroup.empty()) {
          result.push_back(currentGroup);
          currentGroup.clear();
        }
        inParens = false;
        i++;
        continue;
      }

      if (c == ' ' && !inParens && !currentGroup.empty()) {
        result.push_back(currentGroup);
        currentGroup.clear();
        i++;
        continue;
      }

      if (c == ' ') {
        i++;
        continue;
      }

      // Parse dimension name.
      size_t start = i;
      while (i < pattern.size() && pattern[i] != ' ' && pattern[i] != ')' &&
             pattern[i] != '(') {
        i++;
      }

      if (i > start) {
        currentGroup.push_back(pattern.substr(start, i - start));
      }
    }

    if (!currentGroup.empty()) {
      result.push_back(currentGroup);
    }

    return result;
  };

  auto inputDims = parseDims(inputPattern);
  auto outputDims = parseDims(outputPattern);

  // Build a map from dimension name to input position.
  llvm::DenseMap<llvm::StringRef, unsigned> dimToInputPos;
  unsigned pos = 0;
  for (const auto &group : inputDims) {
    if (group.size() > 1) {
      // Input groups are currently unsupported.
      return failure();
    }

    if (pos >= shape.size()) {
      // OOB dimension position for provided shape.
      return failure();
    }

    for (llvm::StringRef dim : group) {
      dimToInputPos[dim] = pos;
    }

    pos++;
  }

  // Build the affine expressions for the output.
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.resize(inputDims.size(), nullptr);
  for (const auto [groupPos, group] : llvm::enumerate(outputDims)) {
    assert(!group.empty());
    // For flattening like b h -> (b h)@d, we create inverse map: (d / h_size, d
    // % h_size).
    int64_t stride = 1;
    mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, context);
    for (int64_t i = static_cast<int64_t>(group.size()) - 1; i >= 0; --i) {
      unsigned dimPos = dimToInputPos[group[i]];
      expr = mlir::getAffineDimExpr(groupPos, context).floorDiv(stride);
      if (i > 0) {
        expr = expr % shape[dimPos];
      }
      stride *= shape[dimPos];
      exprs[dimPos] = expr;
    }
  }

  return mlir::AffineMap::get(outputDims.size(), 0, exprs, context);
}

mlir::FailureOr<::mlir::AffineMap>
mlir::tt::ttir::RearrangeOp::getInvPatternMap() {
  return getInvPatternMap(
      getContext(), getPattern(),
      mlir::cast<ShapedType>(getInput().getType()).getShape());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// BroadcastOp verification
::mlir::LogicalResult mlir::tt::ttir::BroadcastOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Sanity check to make sure that input rank matches the rank of the output
  // tensor.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank of " << inputType.getRank()
                         << " does not match output tensor rank of "
                         << outputType.getRank();
  }

  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Verify that inputShape can be legally broadcasted to outputShape.
  llvm::SmallVector<int64_t> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError() << "Input tensor shape ("
                         << ttmlir::utils::join(inputShape, ",")
                         << ") is not broadcastable to output shape ("
                         << ttmlir::utils::join(outputShape, ",") << ")";
  }

  auto broadcastDimensions = getBroadcastDimensions();

  // Check that the shape size matches the rank of the output tensor.
  if (static_cast<int64_t>(broadcastDimensions.size()) != inputType.getRank()) {
    return emitOpError("Input tensor rank should match output tensor rank.");
  }

  // Verify that each dimension of the inputShape multiplied by corresponding
  // broadcast dimension is equal to the outputShape dimension.
  for (size_t i = 0; i < broadcastDimensions.size(); i++) {
    int64_t dimValue = broadcastDimensions[i];
    if (inputShape[i] * dimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",") << ") index "
                           << i << " does not broadcast to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using broadcast value " << dimValue;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SliceStaticOp
//===----------------------------------------------------------------------===//

// SliceStaticOp verification
::mlir::LogicalResult mlir::tt::ttir::SliceStaticOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::ArrayAttr begins = getBeginsAttr();
  ::mlir::ArrayAttr ends = getEndsAttr();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getType();

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

// back to back producer-consumer SliceStaticOp can be folded
// into a single SliceStaticOp.
static mlir::OpFoldResult
foldConsecutiveSliceStatic(mlir::tt::ttir::SliceStaticOp consumerOp) {

  if (auto producerSliceOp =
          consumerOp.getInput()
              .getDefiningOp<mlir::tt::ttir::SliceStaticOp>()) {

    // If producerOp has multiple uses, do not fold
    if (!producerSliceOp->hasOneUse()) {
      return nullptr;
    }

    mlir::RankedTensorType inputType = consumerOp.getInput().getType();
    size_t input_rank = static_cast<size_t>(inputType.getRank());

    // Producer begins and step arrays
    mlir::ArrayAttr begins1 = producerSliceOp.getBeginsAttr();
    mlir::ArrayAttr stepAttr1 = producerSliceOp.getStepAttr();

    // Consumer begins, end and step arrays
    mlir::ArrayAttr begins2 = consumerOp.getBeginsAttr();
    mlir::ArrayAttr ends2 = consumerOp.getEndsAttr();
    mlir::ArrayAttr stepAttr2 = consumerOp.getStepAttr();

    llvm::SmallVector<mlir::Attribute> begins, ends, step;
    mlir::MLIRContext *ctx = consumerOp->getContext();
    auto i32Ty = mlir::IntegerType::get(ctx, 32);
    for (size_t i = 0; i < input_rank; ++i) {
      // ith dimension in Producer begins and step arrays
      int32_t b1_i = mlir::cast<::mlir::IntegerAttr>(begins1[i]).getInt();
      int32_t s1_i = mlir::cast<::mlir::IntegerAttr>(stepAttr1[i]).getInt();

      // ith dimension in Consumer begins, end and step arrays
      int32_t b2_i = mlir::cast<::mlir::IntegerAttr>(begins2[i]).getInt();
      int32_t e2_i = mlir::cast<::mlir::IntegerAttr>(ends2[i]).getInt();
      int32_t s2_i = mlir::cast<::mlir::IntegerAttr>(stepAttr2[i]).getInt();

      int32_t b_i = b1_i + s1_i * b2_i;
      int32_t e_i = b1_i + s1_i * (e2_i - 1) + 1;
      int32_t s_i = s1_i * s2_i;

      begins.push_back(mlir::IntegerAttr::get(i32Ty, b_i));
      ends.push_back(mlir::IntegerAttr::get(i32Ty, e_i));
      step.push_back(mlir::IntegerAttr::get(i32Ty, s_i));
    }

    mlir::ArrayAttr beginsArrayAttr = mlir::ArrayAttr::get(ctx, begins);
    mlir::ArrayAttr endsArrayAttr = mlir::ArrayAttr::get(ctx, ends);
    mlir::ArrayAttr stepArrayAttr = mlir::ArrayAttr::get(ctx, step);

    consumerOp.setBeginsAttr(beginsArrayAttr);
    consumerOp.setEndsAttr(endsArrayAttr);
    consumerOp.setStepAttr(stepArrayAttr);
    consumerOp->setOperand(0, producerSliceOp.getInput());
    return consumerOp.getResult();
  }

  return nullptr;
}

// SliceStaticOp Folder
mlir::OpFoldResult mlir::tt::ttir::SliceStaticOp::fold(FoldAdaptor adaptor) {

  if (auto foldResult = foldConsecutiveSliceStatic(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// SliceDynamicOp
//===----------------------------------------------------------------------===//

// SliceDynamicOp verification
::mlir::LogicalResult mlir::tt::ttir::SliceDynamicOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType beginsType = getBegins().getType();
  ::llvm::ArrayRef<int64_t> beginsShape = beginsType.getShape();
  ::mlir::RankedTensorType endsType = getEnds().getType();
  ::llvm::ArrayRef<int64_t> endsShape = endsType.getShape();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getType();

  // Verify that the input is at least 1D tensor.
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Verify that begins and ends are 1D tensors.
  size_t beginsRank = static_cast<size_t>(beginsType.getRank());
  size_t endsRank = static_cast<size_t>(endsType.getRank());
  if (beginsRank != 1 || endsRank != 1) {
    return emitOpError("Begins and ends must be 1D tensors");
  }

  // Verify that the input rank matches number of elements in begins, ends, and
  // step.
  auto inputRank = inputType.getRank();

  if (inputRank != beginsShape[0] || inputRank != endsShape[0] ||
      (stepAttr && static_cast<size_t>(inputRank) != stepAttr.size())) {
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
    for (auto i = 0; i < inputRank; ++i) {
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
// IndexOp
//===----------------------------------------------------------------------===//

// ANCHOR: decomposing_an_op_index_ttir_verify
// IndexOp verification
::mlir::LogicalResult mlir::tt::ttir::IndexOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::RankedTensorType outputType = getType();
  int32_t dim = getDim();
  int32_t begin = getBegin();
  int32_t end = getEnd();
  int32_t step = getStep();

  // Verify that the input is at least 1D tensor
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
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

  // Verify that the dim attribute is within the bounds of the input tensor
  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension index " << dim
                         << ". Input tensor rank is " << inputType.getRank();
  }

  // Verify begin, end, step and the output tensor dimensions
  int64_t dimSize = inputShape[dim];

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

  if (adjustedBegin < 0 || adjustedBegin >= dimSize) {
    return emitOpError() << "Invalid begin index for dimension "
                         << std::to_string(dim) << ". Expected value in range ["
                         << std::to_string(-dimSize) << ", " << dimSize
                         << "), got " << begin
                         << ". Input shape: " << inputShapeStr;
  }
  if (adjustedEnd < 0 || adjustedEnd > dimSize) {
    return emitOpError() << "Invalid end index for dimension "
                         << std::to_string(dim) << ". Expected value in range ["
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
    return emitOpError("Step value for dimension " + std::to_string(dim) +
                       " cannot be zero");
  }

  if (step > 0 && adjustedBegin > adjustedEnd) {
    return emitOpError() << "For positive step, begin index must be less "
                            "than or equal to end index for dimension "
                         << dim << ". Got begin: " << beginValueMessage
                         << ", end: " << endValueMessage << ", step: " << step
                         << ", input shape: " << inputShapeStr;
  }

  if (step < 0 && adjustedBegin < adjustedEnd) {
    return emitOpError() << "For negative step, begin index must be greater "
                            "than or equal to end index for dimension "
                         << dim << ". Got begin: " << beginValueMessage
                         << ", end: " << endValueMessage << ", step: " << step
                         << ", input shape: " << inputShapeStr;
  }

  // Calculate the expected size of the output dimension
  int32_t expectedDimSize =
      (std::abs(adjustedEnd - adjustedBegin) + std::abs(step) - 1) /
      std::abs(step);
  if (outputType.getDimSize(dim) != expectedDimSize) {
    return emitOpError() << "Mismatch in dimension " << std::to_string(dim)
                         << " of the output tensor: expected size "
                         << expectedDimSize << ", but got "
                         << outputType.getDimSize(dim);
  }

  return success();
}
// ANCHOR_END: decomposing_an_op_index_ttir_verify

//===----------------------------------------------------------------------===//
// IndexSelectOp
//===----------------------------------------------------------------------===//

// IndexSelectOp verification
::mlir::LogicalResult mlir::tt::ttir::IndexSelectOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError("Input and output tensors must have the same rank.");
  }

  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input and output tensors must have the same element "
                       "type.");
  }

  int32_t dim = getDim();
  int32_t origDim = dim;
  if (dim < 0) {
    dim += inputType.getRank();
  }

  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension " << origDim
                         << " for select op with input tensor rank "
                         << inputType.getRank();
  }

  int32_t dimSize = inputType.getDimSize(dim);

  int32_t stride = getStride();
  if (stride == 0) {
    stride = dimSize;
  }

  if (stride < 0) {
    return emitOpError() << "Invalid stride " << stride << " for dimension "
                         << dim << ", stride must be non-negative";
  }

  if (stride > dimSize) {
    return emitOpError() << "Invalid stride " << stride << " for dimension "
                         << dim << " with size " << dimSize
                         << ". stride must be less than or equal to the "
                            "dimension size";
  }

  int32_t begin = getBegin();
  int32_t length = getLength();
  if (begin < 0 || begin >= dimSize) {
    return emitOpError() << "Invalid begin index " << begin << " for dimension "
                         << dim << " with size " << dimSize
                         << ". begin must be "
                            "in the range [0, dimSize)";
  }

  if (length < 1 || length > stride) {
    return emitOpError() << "Invalid length " << length << " for begin index "
                         << begin << " and stride " << stride
                         << " for dimension " << dim << " with size " << dimSize
                         << ". stride must be greater than or equal to length";
  }

  if (begin + length > dimSize) {
    return emitOpError() << "Invalid length " << length << " for begin index "
                         << begin << " and dimension " << dim << " with size "
                         << dimSize
                         << ". begin + length must be less than or "
                            "equal to the dimension size";
  }

  // Get the number of slices as the number of times the stride fits in the
  // dimension size starting from the begin index.
  int32_t numSlices = (dimSize - begin + stride - 1) / stride;
  int32_t totalLength = 0;
  for (int32_t i = 0; i < numSlices; i++) {
    int32_t newBegin = begin + i * stride;
    int32_t newEnd = std::min(newBegin + length, dimSize);
    totalLength += newEnd - newBegin;
  }

  if (totalLength != outputType.getDimSize(dim)) {
    return emitOpError() << "Sum of all slices must be equal to the output "
                            "dimension size for the given dimension. Expected "
                            "output dimension size: "
                         << outputType.getDimSize(dim) << ", but got "
                         << totalLength;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

// SqueezeOp verification
::mlir::LogicalResult mlir::tt::ttir::SqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  int32_t dim = getDim();

  if (dim < 0) {
    dim += inputType.getRank();
  }

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension " << dim << " for squeezing.";
  }

  // Check that the dimension `dim` is 1 in the input tensor.
  if (inputType.getDimSize(dim) != 1) {
    return emitOpError() << "Dimension " << dim
                         << " in the input tensor must be 1.";
  }

  if (outputType.getRank() == 0) {
    return emitOpError() << "Output tensor must have at least one dimension.";
  }

  // Check that the rank of the output tensor is one less than the input tensor.
  if (outputType.getRank() != inputType.getRank() - 1) {
    return emitOpError()
           << "Output tensor rank must be one less than the input tensor rank.";
  }

  // Check that the dimensions of the output tensor are the same as the input
  // tensor except for dimension `dim`.
  for (int64_t i = 0, j = 0; i < inputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }
    if (inputType.getDimSize(i) != outputType.getDimSize(j)) {
      return emitOpError() << "Dimensions of the output tensor must be the "
                              "same as the input tensor except for dimension "
                           << dim << ".";
    }
    ++j;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// TransposeOp verification
::mlir::LogicalResult mlir::tt::ttir::TransposeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
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

void mlir::tt::ttir::TransposeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {
  patterns.add(+[](TransposeOp op, mlir::PatternRewriter &rewriter) {
    SmallVector<int64_t> permutation;
    for (int64_t i = 0; i < op.getInput().getType().getRank(); ++i) {
      permutation.push_back(i);
    }
    int64_t dim0 = op.getDim0() < 0
                       ? op.getDim0() + op.getInput().getType().getRank()
                       : op.getDim0();
    int64_t dim1 = op.getDim1() < 0
                       ? op.getDim1() + op.getInput().getType().getRank()
                       : op.getDim1();
    std::swap(permutation[dim0], permutation[dim1]);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), op.getInput(),
                                           permutation);
    return success();
  });
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

// TypecastOp folder
mlir::OpFoldResult mlir::tt::ttir::TypecastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getInput().getType()) {
    return getInput();
  }
  return {};
}

static bool isNarrowingConversion(const ::mlir::tt::ttcore::DataType srcDtype,
                                  const ::mlir::tt::ttcore::DataType dstDtype) {
  const bool srcIsFloat = isFloat(srcDtype);
  const bool dstIsFloat = isFloat(dstDtype);
  const auto srcNumberOfBits = getNumberOfBits(srcDtype);
  const auto dstNumberOfBits = getNumberOfBits(dstDtype);

  if (srcIsFloat && !dstIsFloat) {
    return true;
  }

  if (srcIsFloat && dstIsFloat) {
    const auto srcExponentSize = getExponentSize(srcDtype);
    const auto dstExponentSize = getExponentSize(dstDtype);
    const auto srcMantissaSize = getMantissaSize(srcDtype);
    const auto dstMantissaSize = getMantissaSize(dstDtype);
    return srcExponentSize > dstExponentSize ||
           srcMantissaSize > dstMantissaSize;
  }

  // For integer to FP, it is narrowing if the FP type has fewer bits in its
  // mantissa than the integer type's magnitude bits.
  if (!srcIsFloat && dstIsFloat) {
    if (isSignedInteger(srcDtype)) {
      return srcNumberOfBits - 1 > getMantissaSize(dstDtype);
    }
    return srcNumberOfBits > getMantissaSize(dstDtype);
  }

  assert(!srcIsFloat && !dstIsFloat);
  const auto srcIsSigned = isSignedInteger(srcDtype);
  const auto dstIsSigned = isSignedInteger(dstDtype);
  // When signedness are the same, reducing the number of bits is narrowing.
  if (srcIsSigned == dstIsSigned) {
    return srcNumberOfBits > dstNumberOfBits;
  }
  // Unsigned->Signed is narrowing when the signed type can't hold the largest.
  // value of the unsigned type
  if (!srcIsSigned && dstIsSigned) {
    return srcNumberOfBits >= dstNumberOfBits;
  }
  // Signed->Unsigned is always narrowing.
  assert(srcIsSigned && !dstIsSigned);
  return true;
}

// TypecastOp canonicalization method
::llvm::LogicalResult
mlir::tt::ttir::TypecastOp::canonicalize(mlir::tt::ttir::TypecastOp op,
                                         ::mlir::PatternRewriter &rewriter) {
  // Fold two consecutive typecast ops into a single one.
  ::mlir::tt::ttir::TypecastOp producerOp =
      op.getInput().getDefiningOp<mlir::tt::ttir::TypecastOp>();

  if (!producerOp) {
    return mlir::failure();
  }

  const bool conservativeFolding =
      op.getConservativeFolding() || producerOp.getConservativeFolding();

  if (conservativeFolding) {
    // Disable folding if it has the potential to cause too much numerical
    // differences.
    auto dtypeIn = ttcore::elementTypeToDataType(
        producerOp.getInput().getType().getElementType());
    auto dtypeMid =
        ttcore::elementTypeToDataType(op.getInput().getType().getElementType());
    auto dtypeOut =
        ttcore::elementTypeToDataType(op.getType().getElementType());

    assert(dtypeMid == ttcore::elementTypeToDataType(
                           producerOp.getType().getElementType()));

    // If the 1st Op is narrowing and the 2nd Op is widening, we shouldn't fold.
    // FP->Int->FP is special and should never fold, due to its truncation
    // semantics and application in QDQ models.
    const bool isNarrowingProducer = isNarrowingConversion(dtypeIn, dtypeMid);
    const bool isNarrowingConsumer = isNarrowingConversion(dtypeMid, dtypeOut);
    const bool isFpIntFp =
        isFloat(dtypeIn) && !isFloat(dtypeMid) && isFloat(dtypeOut);
    if (isFpIntFp || (isNarrowingProducer && !isNarrowingConsumer)) {
      return mlir::failure();
    }
  }

  // The resulting Op is conservative iff both typecast ops were conservative.
  rewriter.replaceOpWithNewOp<ttir::TypecastOp>(
      op, op.getType(), producerOp.getInput(),
      op.getConservativeFolding() && producerOp.getConservativeFolding());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnsqueezeOp
//===----------------------------------------------------------------------===//

// UnsqueezeOp verification
::mlir::LogicalResult mlir::tt::ttir::UnsqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  int32_t dim = getDim();

  // Convert negative dim to its positive equivalent
  if (dim < 0) {
    dim += inputType.getRank() + 1;
  }

  // Check that the dim is within the bounds of the input tensor
  if (dim > inputType.getRank() || dim < 0) {
    return emitOpError(
        "Dimension attribute must be within the bounds of the input tensor");
  }

  // Check that the output tensor has one more dimension than the input tensor
  if (outputType.getRank() != inputType.getRank() + 1) {
    return emitOpError(
        "Output tensor must have one more dimension than the input tensor");
  }

  // and that the dimension added is of size 1
  if (outputType.getDimSize(dim) != 1) {
    return emitOpError("Dimension added must be of size 1");
  }

  // All dimensions of the input tensor must be the same as the output tensor
  // except for the dimension added
  for (int64_t i = 0, j = 0; i < outputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }

    if (inputType.getDimSize(j) != outputType.getDimSize(i)) {
      return emitOpError("All dimensions of the input tensor must be the same "
                         "as the output tensor except for the dimension added");
    }

    j++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

// EmbeddingOp verification
::mlir::LogicalResult mlir::tt::ttir::EmbeddingOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Input tensor must be at most 2D tensor.
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
::mlir::LogicalResult mlir::tt::ttir::EmbeddingBackwardOp::verify() {
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType inputGradType = getInGradient().getType();
  ::mlir::RankedTensorType outputType = getType();

  // weightType must have rank of 2: (dictionary_size, embedding_size).
  if (weightType.getRank() != 2) {
    return emitOpError("Input must be a 2D tensor");
  }

  // inputGradType checks.
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
// ToLayoutOp
//===----------------------------------------------------------------------===//

struct ToLayoutFoldRedundantPattern : public OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  ToLayoutFoldRedundantPattern(MLIRContext *context)
      : OpRewritePattern<ToLayoutOp>(context) {
    setDebugName("ttir.ToLayoutFoldRedundantPattern");
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    ToLayoutOp producerLayoutOp = op.getInput().getDefiningOp<ToLayoutOp>();
    if (!producerLayoutOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ToLayoutOp>(op, producerLayoutOp.getInput(),
                                            op.getOutput());
    return success();
  }
};

static ::mlir::LogicalResult
verifyLayoutOp(mlir::Operation *op, mlir::Type inputTensorOrMemrefTy,
               mlir::Type outputTensorOrMemrefTy, bool allowFormatChange,
               bool allowMemorySpaceChange, bool checkMemrefRank = false,
               bool checkMemrefGridShardForm = false,
               bool checkMemrefShardShape = false) {
  if (mlir::RankedTensorType inputTy =
          mlir::dyn_cast<mlir::RankedTensorType>(inputTensorOrMemrefTy)) {
    mlir::RankedTensorType outputTy =
        mlir::dyn_cast<mlir::RankedTensorType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    auto inputLayout =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
            inputTy.getEncoding());
    auto outputLayout =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
            outputTy.getEncoding());

    if (!inputLayout || !outputLayout) {
      // If the input/output tensor does not have a layout, we can early exit.
      return mlir::success();
    }

    const bool isFormatChange =
        inputTy.getElementType() != outputTy.getElementType();
    if (isFormatChange && !allowFormatChange) {
      return op->emitOpError(
          "Input and output tensor element types must be the same");
    }

    const bool isMemorySpaceChange =
        inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output layout memory spaces must be the same");
    }
    return mlir::success();
  }

  if (mlir::MemRefType inputTy =
          mlir::dyn_cast<mlir::MemRefType>(inputTensorOrMemrefTy)) {
    mlir::MemRefType outputTy =
        mlir::dyn_cast<mlir::MemRefType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    const bool isFormatChange =
        inputTy.getElementType() != outputTy.getElementType();
    if (!allowFormatChange && isFormatChange) {
      return op->emitOpError(
          "Input and output layout element types must be the same");
    }

    const bool isMemorySpaceChange =
        inputTy.getMemorySpace() != outputTy.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output memref memory spaces must be the same");
    }

    const bool sameRank = inputTy.getRank() == outputTy.getRank();
    if (checkMemrefRank && !sameRank) {
      return op->emitOpError("Input and output memref ranks must be the same");
    }

    auto inputDeviceLayout =
        mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
            inputTy.getLayout());
    if (checkMemrefGridShardForm && !inputDeviceLayout) {
      return op->emitOpError(
          "input memref must have device layout, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    auto outputDeviceLayout =
        mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
            outputTy.getLayout());
    if (checkMemrefGridShardForm && !outputDeviceLayout) {
      return op->emitOpError(
          "output memref must have device layout, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    return mlir::success();
  }

  return op->emitOpError("Unsupported input type for view");
}

// ToLayoutOp verification
::mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::verify() {
  return verifyLayoutOp(*this, getInput().getType(), getOutput().getType(),
                        /*allowFormatChange*/ true,
                        /*allowMemorySpaceChange*/ true);
}

// ToLayoutOp utility methods
mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::fold(
    FoldAdaptor, llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  mlir::RankedTensorType inputType =
      dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::RankedTensorType outputType =
      dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (inputType && outputType && inputType == outputType) {
    results.push_back(getInput());
    return mlir::success();
  }
  return mlir::failure();
}

bool mlir::tt::ttir::ToLayoutOp::isHostToDevice() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getInput().getType())
              .getEncoding());
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getOutput().getType())
              .getEncoding());
  return hostInput && !hostOutput;
}

bool mlir::tt::ttir::ToLayoutOp::isDeviceToHost() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getInput().getType())
              .getEncoding());
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getOutput().getType())
              .getEncoding());
  return !hostInput && hostOutput;
}

void mlir::tt::ttir::ToLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Fold into ttir.empty w/ desired layout
  patterns.add(+[](ToLayoutOp op, mlir::PatternRewriter &rewriter) {
    EmptyOp emptyOp = op.getInput().getDefiningOp<EmptyOp>();
    if (!emptyOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getOutput().getType());
    return success();
  });

  patterns.add(std::make_unique<ToLayoutFoldRedundantPattern>(context));
}

//===----------------------------------------------------------------------===//
// TTNNMetalLayoutCastOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttir::TTNNMetalLayoutCastOp::verify() {
  auto inputType = mlir::dyn_cast<mlir::ShapedType>(getInput().getType());
  auto outputType = mlir::dyn_cast<mlir::ShapedType>(getResult().getType());

  const bool inputIsMemref = mlir::isa<mlir::MemRefType>(inputType);
  const bool outputIsMemref = mlir::isa<mlir::MemRefType>(outputType);

  auto maybeInputTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType);
  auto maybeOutputTensor = mlir::dyn_cast<mlir::RankedTensorType>(outputType);

  auto maybeInputAttr =
      maybeInputTensor ? maybeInputTensor.getEncoding() : nullptr;
  auto maybeOutputAttr =
      maybeOutputTensor ? maybeOutputTensor.getEncoding() : nullptr;

  const bool inputIsTTNNTensor =
      maybeInputTensor &&
      (mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(maybeInputAttr) ||
       mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(maybeInputAttr));
  const bool outputIsTTNNTensor =
      maybeOutputTensor &&
      (mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(maybeOutputAttr) ||
       mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(maybeOutputAttr));

  const bool inputIsMetalTensor =
      maybeInputTensor &&
      mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(maybeInputAttr);
  const bool outputIsMetalTensor =
      maybeOutputTensor &&
      mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(maybeOutputAttr);

  if (inputIsTTNNTensor) {
    if (!outputIsMetalTensor && !outputIsMemref) {
      return emitOpError(
          "Input is ttnn tensor, output must be metal tensor or memref");
    }
  }

  if (inputIsMetalTensor || inputIsMemref) {
    if (!outputIsTTNNTensor) {
      return emitOpError(
          "Input is metal tensor or memref, output must be ttnn tensor");
    }
  }

  return success();
}

void mlir::tt::ttir::TTNNMetalLayoutCastOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cast");
}

void mlir::tt::ttir::TTNNMetalLayoutCastOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // Eliminate back-to-back casts of the same tensor.

  // Because of the verifier, we are guaranteed that casts are always
  // ttnn->metal or metal->ttnn. Therefore, if the producer of the input is a
  // cast, we necessarily have one of:
  //   1. metal_0->ttnn->metal_1
  //   2. ttnn_0->metal->ttnn_1
  // The cast is purely representational and no layout change is performed.
  // At op creation time it is guaranteed that the input and output are the same
  // tensor in the same underlying layout. Therefore we can be sure that:
  //   metal_0 == metal_1 in 1
  //   ttnn_0 == ttnn_1 in 2

  patterns.add(+[](TTNNMetalLayoutCastOp op, mlir::PatternRewriter &rewriter) {
    TTNNMetalLayoutCastOp producerOp =
        op.getInput().getDefiningOp<TTNNMetalLayoutCastOp>();

    if (!producerOp) {
      return failure();
    }

    // Bail if we are post bufferization.
    auto producerInputTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(producerOp.getInput().getType());
    auto producerOutputTensor = mlir::dyn_cast<mlir::RankedTensorType>(
        producerOp.getResult().getType());
    auto consumerInputTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    auto consumerOutputTensor =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());

    if (!producerInputTensor || !producerOutputTensor || !consumerInputTensor ||
        !consumerOutputTensor) {
      return failure();
    }

    rewriter.replaceOp(op, producerOp.getInput());
    return success();
  });
}

mlir::LogicalResult mlir::tt::ttir::TTNNMetalLayoutCastOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (mlir::isa<mlir::MemRefType>(resultType) ||
      mlir::isa<mlir::MemRefType>(inputType)) {
    return success();
  }

  auto inputTensor = mlir::cast<mlir::RankedTensorType>(inputType);
  auto outputTensor = mlir::cast<mlir::RankedTensorType>(resultType);

  auto inputEncoding = inputTensor.getEncoding();
  auto outputEncoding = outputTensor.getEncoding();

  if (mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(inputEncoding)) {
    // metal_layout -> ttnn_layout becomes memref -> ttnn_layout
    bool isTTNNLayout =
        mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(outputEncoding) ||
        mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(outputEncoding);
    TT_assertv(isTTNNLayout,
               "Output tensor must have ttnn_layout or ttnn_nd_layout");
    auto maybeInputBuf =
        mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
    if (failed(maybeInputBuf)) {
      return maybeInputBuf;
    }
    rewriter.replaceOpWithNewOp<TTNNMetalLayoutCastOp>(*this, outputTensor,
                                                       *maybeInputBuf);
  } else if (mlir::isa<mlir::tt::ttcore::MetalLayoutAttr>(outputEncoding)) {
    // ttnn_layout -> metal_layout becomes ttnn_layout -> memref
    bool isTTNNLayout =
        mlir::isa<mlir::tt::ttnn::TTNNLayoutAttr>(inputEncoding) ||
        mlir::isa<mlir::tt::ttnn::TTNNNDLayoutAttr>(inputEncoding);
    TT_assertv(isTTNNLayout,
               "Input tensor must have ttnn_layout or ttnn_nd_layout");
    ::llvm::SmallVector<mlir::Value> dummy;
    auto bufferType = getBufferType(getResult(), options, state, dummy);
    if (failed(bufferType)) {
      return bufferType;
    }
    MemRefType outputMemrefType = mlir::cast<mlir::MemRefType>(*bufferType);
    mlir::bufferization::replaceOpWithNewBufferizedOp<TTNNMetalLayoutCastOp>(
        rewriter, *this, outputMemrefType, getInput());

  } else {
    return emitOpError("Neither input or output uses metal_layout");
  }
  return success();
}

mlir::bufferization::AliasingValueList
mlir::tt::ttir::TTNNMetalLayoutCastOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
mlir::tt::ttir::TTNNMetalLayoutCastOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return mlir::tt::ttcore::getBufferType(value.getType(), /*isView=*/false);
}

bool mlir::tt::ttir::TTNNMetalLayoutCastOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // no-op
  return false;
}

bool mlir::tt::ttir::TTNNMetalLayoutCastOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // no-op
  return false;
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

// LinearOp verification
::mlir::LogicalResult mlir::tt::ttir::LinearOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  std::optional<::mlir::RankedTensorType> biasType =
      getBias() ? std::make_optional(getBias().getType()) : std::nullopt;
  ::mlir::RankedTensorType outputType = getType();

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
  // the purpose of the matrix multiplication. After the matrix
  // multiplication, the prepended dimension is removed. Otherwise, check if
  // the LHS needs to be transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards.
  // Otherwise, check if the RHS needs to be transposed.
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
  // Verify that the batch dimensions are broadcast compatible and construct
  // the expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            inputABatchDims, inputBBatchDims, broadcastedShape)) {

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
    llvm::SmallVector<int64_t> broadcastShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(expectedOutputShape,
                                                  biasShape, broadcastShape)) {
      return emitOpError("Bias shape(")
             << ttmlir::utils::join(biasShape, ",")
             << ") is not broadcast compatible with the matmul output shape("
             << ttmlir::utils::join(expectedOutputShape, ",") << ")";
    }

    // tt-metal uses a composite LinearOp where the bias is added after the
    // matmul, and ttnn.add supports broadcasting of both operands. Otherwise,
    // tt-metal lowers to a fused LinearOp, which uses the matmul result shape
    // as the output shape. The composite LinearOp requires that the bias
    // second-to-last dim (of padded shape) does not match the tile height.
    // Update the expected output shape to the fully broadcasted shape when:
    // 1) The matmul result is a scalar (vector x vector), so the inferred
    //    output shape is empty and must be derived from the bias via
    //    broadcasting, or
    // 2) The bias second-to-last dim (of padded shape) does not match the
    //    tile height, indicating the composite LinearOp is used.
    llvm::SmallVector<int64_t> paddedBiasShape =
        ttnn::utils::getTilePaddedShape(biasShape);
    if (expectedOutputShape.empty() ||
        (paddedBiasShape.size() > 1 &&
         paddedBiasShape[paddedBiasShape.size() - 2] != ttnn::TILE_HEIGHT)) {
      expectedOutputShape = broadcastShape;
    }
  }

  // Check the case of a vector-vector product. At this moment we don't
  // support scalars in IR, hence check that the output is at least 1D tensor
  // of size 1.
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

// LinearOp canonicalization
void mlir::tt::ttir::LinearOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // If bias is not provided, linear operation is equivalent to matmul.
  patterns.add(+[](ttir::LinearOp op, mlir::PatternRewriter &rewriter) {
    if (!op.getBias()) {
      rewriter.replaceOpWithNewOp<ttir::MatmulOp>(op, op.getType(), op.getA(),
                                                  op.getB(), op.getTransposeA(),
                                                  op.getTransposeB());
      return mlir::success();
    }
    return mlir::failure();
  });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_matmul_ttir_verify
// MatmulOp verification
::mlir::LogicalResult mlir::tt::ttir::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getType();

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
  // the purpose of the matrix multiplication. After the matrix
  // multiplication, the prepended dimension is removed. Otherwise, check if
  // the LHS needs to be transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards.
  // Otherwise, check if the RHS needs to be transposed.
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
  // Verify that the batch dimensions are broadcast compatible and construct
  // the expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            inputABatchDims, inputBBatchDims, broadcastedShape)) {

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

  // Check the case of a vector-vector product. At this moment we don't
  // support scalars in IR, hence check that the output is at least 1D tensor
  // of size 1.
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
// ANCHOR_END: adding_an_op_matmul_ttir_verify

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

// UpsampleOp verification
::mlir::LogicalResult mlir::tt::ttir::Upsample2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

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

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
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
// AllocOp
//===----------------------------------------------------------------------===//

// AllocOp verification
::mlir::LogicalResult mlir::tt::ttir::AllocOp::verify() {
  auto layout = mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
      getResult().getType().getEncoding());
  if (not layout) {
    return emitOpError("Result type missing layout attribute");
  }

  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto memspace = layout.getMemorySpace();
  if (memspace != getMemorySpace()) {
    return emitOpError(
        "Input tensor layout memory space must match alloc memory space");
  }

  if (isSystemMemorySpace(getMemorySpace()) and getAddress() != 0) {
    return emitOpError("Allocating from system memory space must have address "
                       "set to 0, implicitly allocated by the runtime");
  }

  if (isDeviceMemorySpace(memspace) and getAddress() == 0) {
    return emitOpError(
        "Allocating from a device memory space must have address "
        "set to a non-zero value, device addresses are statically allocated");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

// RepeatOp verification.
::mlir::LogicalResult mlir::tt::ttir::RepeatOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
  llvm::ArrayRef<int64_t> repeatDimensions = getRepeatDimensions();

  // Input tensor and repeat dimension argument must have same rank.
  if (inputType.getRank() != static_cast<int64_t>(repeatDimensions.size())) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the number of repeat dimensions "
                         << repeatDimensions.size() << ".";
  }

  // Input and output tensors must have the same rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the output tensor rank "
                         << outputType.getRank() << ".";
  }

  // Verify output shape based on input shape and repeat dimension argument.
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  for (size_t i = 0; i < inputShape.size(); i++) {
    // Verify that the repeat dimension is greater than 0.
    if (repeatDimensions[i] <= 0) {
      return emitOpError() << "Repeat dimension at index " << i
                           << " must be greater than 0.";
    }

    int64_t expectedDimValue = inputShape[i] * repeatDimensions[i];
    if (expectedDimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",")
                           << ") at index " << i
                           << " does not repeat to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using repeat value " << repeatDimensions[i]
                           << ".";
    }
  }

  return success();
}

// back to back producer-consumer RepeatOp can be folded
// into a single RepeatOp.
static mlir::OpFoldResult
foldConsecutiveRepeat(mlir::tt::ttir::RepeatOp consumerOp) {

  if (auto producerOp =
          consumerOp.getInput().getDefiningOp<mlir::tt::ttir::RepeatOp>()) {

    // If producerOp has multiple uses, do not fold
    if (!producerOp->hasOneUse()) {
      return nullptr;
    }

    mlir::RankedTensorType inputType = producerOp.getInput().getType();
    size_t inputRank = static_cast<size_t>(inputType.getRank());

    // Producer repeat dimensions
    llvm::ArrayRef<int64_t> producerRepeatDims =
        producerOp.getRepeatDimensions();

    // Consumer repeat dimensions
    llvm::ArrayRef<int64_t> consumerRepeatDims =
        consumerOp.getRepeatDimensions();

    llvm::SmallVector<int64_t> mergedRepeatDimensions(inputRank);
    for (size_t i = 0; i < inputRank; ++i) {
      mergedRepeatDimensions[i] = producerRepeatDims[i] * consumerRepeatDims[i];
    }
    llvm::ArrayRef<int64_t> mergedRepeatDimsRef(mergedRepeatDimensions);
    consumerOp.setRepeatDimensions(mergedRepeatDimsRef);
    consumerOp->setOperand(0, producerOp.getInput());
    return consumerOp.getResult();
  }

  return nullptr;
}

// Repeat op can be folded when repeat dimensions are all 1.
static mlir::OpFoldResult foldIdentityRepeat(mlir::tt::ttir::RepeatOp op) {
  if (llvm::all_of(op.getRepeatDimensions(),
                   [](int64_t dim) { return dim == 1; })) {
    return op.getInput();
  }
  return nullptr;
}

// RepeatOp Folder
mlir::OpFoldResult mlir::tt::ttir::RepeatOp::fold(FoldAdaptor fold) {

  if (auto foldResult = foldIdentityRepeat(*this)) {
    return foldResult;
  }
  if (auto foldResult = foldConsecutiveRepeat(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

// RepeatInterleaveOp verification
::mlir::LogicalResult mlir::tt::ttir::RepeatInterleaveOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();
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
::mlir::LogicalResult mlir::tt::ttir::SoftmaxOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

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
::mlir::LogicalResult mlir::tt::ttir::SortOp::verify() {
  auto dim = getDim();
  auto input = getInput();
  auto rank = input.getType().getRank();
  if (dim >= rank || dim < -rank) {
    return emitOpError("Dimension out of range (expected to be in range of [")
           << -rank << ", " << (rank - 1) << "], but got " << dim << ")";
  }

  auto indicesType =
      mlir::cast<RankedTensorType>(getResults().back().getType());
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
// AllGatherOp
//===----------------------------------------------------------------------===//

// AllGatherOp verification
::mlir::LogicalResult mlir::tt::ttir::AllGatherOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t gatherDim = getAllGatherDim();

  if (gatherDim >= inputType.getRank() || gatherDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid dimension for all gather op. Gather dimension must "
               "be "
               ">= to "
               "input tensor rank or < -input tensor rank, got gather_dim = ")
           << gatherDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

// AllReduceOp verification
::mlir::LogicalResult mlir::tt::ttir::AllReduceOp::verify() {
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();

  // Currently TTIR only supports the sum reduce types.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum) {
    return emitOpError("Invalid reduction op for all reduce op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

// ReduceScatterOp verification
::mlir::LogicalResult mlir::tt::ttir::ReduceScatterOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::tt::ttcore::ReduceType reduceType = getReduceType();
  int32_t scatterDim = getScatterDim();

  // Currently TTIR only supports the following reduce types.
  if (reduceType != ::mlir::tt::ttcore::ReduceType::Sum &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Max &&
      reduceType != ::mlir::tt::ttcore::ReduceType::Min) {
    return emitOpError("Invalid reduction op for reduce scatter op.");
  }

  if (scatterDim >= inputType.getRank() || scatterDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid dimension for reduce scatter op. Scatter dimension "
               "must be >= to input tensor rank or < -input tensor rank, got "
               "scatter_dim = ")
           << scatterDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

// CollectivePermuteOp verification
::mlir::LogicalResult mlir::tt::ttir::CollectivePermuteOp::verify() {
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
  unique. Given a 2D rank tensor of source target pairs eg. [['src',
  'target'],
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

//===----------------------------------------------------------------------===//
// MeshShardOp
//===----------------------------------------------------------------------===//

// MeshShardOp verification
mlir::LogicalResult mlir::tt::ttir::MeshShardOp::verify() {
  auto shardType = getShardType();

  // Currently, we are not supporting maximal from StableHLO.
  if (shardType == mlir::tt::ttcore::MeshShardType::Maximal) {
    return emitOpError("Invalid shard_type (maximal) for mesh_shard op.");
  }

  return success();
}

::mlir::OpFoldResult mlir::tt::ttir::MeshShardOp::fold(FoldAdaptor adaptor) {
  auto shardShapeArray = getShardShape();
  auto shardType = getShardType();
  if (shardType != mlir::tt::ttcore::MeshShardType::Replicate &&
      ttmlir::utils::volume(shardShapeArray) == 1) {
    return getInput();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ScatterOp::verify() {
  const ::mlir::RankedTensorType inputType = getInput().getType();
  const ::mlir::RankedTensorType indexType = getIndex().getType();
  const ::mlir::RankedTensorType sourceType = getSource().getType();

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> indexShape = indexType.getShape();
  llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();

  const size_t inputTypeRank = inputShape.size();
  const size_t indexTypeRank = indexShape.size();
  const size_t sourceTypeRank = sourceShape.size();

  if (inputTypeRank != indexTypeRank || inputTypeRank != sourceTypeRank ||
      indexTypeRank != sourceTypeRank) {
    return emitOpError() << "Input tensor, index tensor, and source tensor "
                            "must have the same rank. "
                         << "Got input rank = " << inputTypeRank
                         << ", index rank = " << indexTypeRank
                         << ", source rank = " << sourceTypeRank;
  }

  if (indexShape != sourceShape) {
    return emitOpError(
        "Index tensor must have the same shape as source tensor.");
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::UpdateCacheOp::verify() {
  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const ttcore::DataType cacheDataType =
      ttcore::elementTypeToDataType(cacheType.getElementType());
  const ttcore::DataType inputDataType =
      ttcore::elementTypeToDataType(inputType.getElementType());

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

  if (inputType.getShape()[0] != 1) {
    return emitOpError("Input tensor requires that dim 0 have size 1, got "
                       "input dim 0 size = " +
                       std::to_string(inputType.getShape()[0]));
  }

  if (cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "dims 1 and 3. Got cache shape (" +
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

void mlir::tt::ttir::UpdateCacheOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(
      +[](mlir::tt::ttir::UpdateCacheOp op, mlir::PatternRewriter &rewriter) {
        auto cacheShape = op.getCache().getType().getShape();
        auto inputShape = op.getInput().getType().getShape();
        auto updateIndexShape = op.getUpdateIndex().getType().getShape();

        auto numUsers = cacheShape[0];
        auto numHeads = cacheShape[1];
        auto headDim = cacheShape[3];

        TypedValue<RankedTensorType> newInput = op.getInput();

        // Permute input if in the format [1, num_heads, num_users, head_dim]
        if (inputShape[2] == numUsers && inputShape[1] == numHeads) {
          llvm::SmallVector<int64_t> newInputShape = {1, numUsers, numHeads,
                                                      headDim};
          auto newInputType = RankedTensorType::get(
              newInputShape, newInput.getType().getElementType(),
              newInput.getType().getEncoding());
          newInput = rewriter.create<PermuteOp>(
              op.getLoc(), newInputType, newInput,
              rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));
        }

        // If the update index shape is [1] then repeat to num users
        TypedValue<RankedTensorType> newUpdateIndex = op.getUpdateIndex();
        if (updateIndexShape[0] == 1) {
          auto newUpdateIndexShape = {numUsers};
          auto newUpdateIndexType = RankedTensorType::get(
              newUpdateIndexShape, newUpdateIndex.getType().getElementType(),
              newUpdateIndex.getType().getEncoding());
          auto repeatDims = rewriter.getDenseI64ArrayAttr({numUsers});
          newUpdateIndex = rewriter.create<RepeatOp>(
              op.getLoc(), newUpdateIndexType, newUpdateIndex, repeatDims);
        }

        rewriter.replaceOpWithNewOp<ttir::PagedUpdateCacheOp>(
            op, op.getType(), op.getCache(), newInput, newUpdateIndex, false,
            nullptr);

        return mlir::success();
      });
}

//===----------------------------------------------------------------------===//
// PagedUpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult PagedUpdateCacheOp::verify() {
  auto cacheType = getCache().getType();
  auto inputType = getInput().getType();
  auto updateIndexType = getUpdateIndex().getType();

  auto cacheShape = cacheType.getShape();
  auto inputShape = inputType.getShape();
  auto updateIndexShape = updateIndexType.getShape();

  bool usingStaticCache = getPageTable() == nullptr;

  if (cacheShape.size() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputShape.size() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (updateIndexShape.size() != 1) {
    return emitOpError("Update index tensor must be a 1D tensor");
  }

  int64_t blockSize = cacheShape[2];
  int64_t headDim = cacheShape[3];
  int64_t numUsers = updateIndexShape[0];

  if (!usingStaticCache && blockSize % ttnn::TILE_HEIGHT != 0) {
    return emitOpError("Block size must be divisible by 32, got " +
                       std::to_string(blockSize));
  }

  if (inputShape[0] != 1) {
    return emitOpError("Input tensor must have dim 0 be equal to 1, got " +
                       std::to_string(inputShape[0]));
  }

  if (inputShape[1] != numUsers) {
    return emitOpError("Input tensor must have shape equal to the number of "
                       "users (determined by update index shape): " +
                       std::to_string(numUsers) + ", got " +
                       std::to_string(inputShape[1]));
  }

  if (inputShape[3] != headDim) {
    return emitOpError("Input tensor must have dim 3 be equal to the head "
                       "dimension (determined by cache shape): " +
                       std::to_string(headDim) + ", got " +
                       std::to_string(inputShape[3]));
  }

  if (!usingStaticCache) {
    auto pageTableType = getPageTable().getType();
    auto pageTableShape = pageTableType.getShape();
    if (pageTableShape.size() != 2) {
      return emitOpError("Page table tensor must be a 2D tensor");
    }

    if (pageTableShape[0] != numUsers) {
      return emitOpError(
          "Page table tensor must have dim 0 be equal to the "
          "number of users (determined by update index shape): " +
          std::to_string(numUsers) + ", got " +
          std::to_string(pageTableShape[0]));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::FillCacheOp::verify() {
  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const ttcore::DataType cacheDataType =
      ttcore::elementTypeToDataType(cacheType.getElementType());
  const ttcore::DataType inputDataType =
      ttcore::elementTypeToDataType(inputType.getElementType());

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
        "equal to the size of dim 2 of the cache tensor. Got cache dim 2 "
        "size = " +
        std::to_string(cacheType.getShape()[2]) +
        ", input dim 2 size = " + std::to_string(inputType.getShape()[2]));
  }

  if (cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "dims 1 and 3. Got cache shape (" +
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
// PagedFillCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::PagedFillCacheOp::verify() {
  auto cacheType = getCache().getType();
  auto inputType = getInput().getType();
  auto pageTableType = getPageTable().getType();

  auto cacheShape = cacheType.getShape();
  auto inputShape = inputType.getShape();
  auto pageTableShape = pageTableType.getShape();

  if (cacheShape.size() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputShape.size() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (pageTableShape.size() != 2) {
    return emitOpError("Page table tensor must be a 2D tensor");
  }

  if (cacheType.getElementType() != inputType.getElementType()) {
    return emitOpError("Cache and input tensors must have the same dtype");
  }

  if (!cacheType.getElementType().isFloat()) {
    return emitOpError("Cache tensor must be a floating point type");
  }

  if (!inputType.getElementType().isFloat()) {
    return emitOpError("Input tensor must be a floating point type");
  }

  if (!pageTableType.getElementType().isInteger()) {
    return emitOpError("Page table tensor must be an integer type");
  }

  if (getBatchIdxTensor()) {
    auto batchIdxTensorType = getBatchIdxTensor().getType();
    if (batchIdxTensorType.getShape().size() != 1) {
      return emitOpError("Batch index tensor must be a 1D tensor");
    }
    if (batchIdxTensorType.getShape()[0] != 1) {
      return emitOpError(
          "Batch index tensor must have dim 0 be equal to 1, got " +
          std::to_string(batchIdxTensorType.getShape()[0]));
    }
    if (!batchIdxTensorType.getElementType().isInteger()) {
      return emitOpError("Batch index tensor must be an integer type");
    }
  }

  int64_t numCacheHeads = cacheShape[1];
  int64_t numInputHeads = inputShape[1];
  int64_t blockSize = cacheShape[2];
  int64_t headDim = cacheShape[3];

  if (blockSize % 32 != 0) {
    return emitOpError("Block size must be divisible by 32, got " +
                       std::to_string(blockSize));
  }

  if (numInputHeads % numCacheHeads != 0) {
    return emitOpError("Input must have a number of heads that is a multiple "
                       "of the number of heads in the cache.");
  }

  if (inputShape[3] != headDim) {
    return emitOpError("Input must have same head dimension as cache.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

// ReverseOp verification
::mlir::LogicalResult mlir::tt::ttir::ReverseOp::verify() {
  llvm::ArrayRef<int64_t> dimensions = getDimensions();

  // Check that all given dimensions are unique/not repeating.
  llvm::SmallDenseSet<int64_t> uniqueDims(dimensions.begin(), dimensions.end());

  if (uniqueDims.size() != dimensions.size()) {
    return emitOpError("dimensions should be unique. Got: ") << dimensions;
  }

  ::mlir::RankedTensorType operandTy = getInput().getType();

  // Check that each dimension is positive and within valid interval [0,
  // operandRank).
  for (int64_t dim : dimensions) {
    if (dim < 0) {
      return emitOpError(
                 "all dimensions should be non-negative. Got dimension: ")
             << dim;
    }

    if (dim >= operandTy.getRank()) {
      return emitOpError("all dimensions should be in interval [0, ")
             << operandTy.getRank() << "). Got dimension: " << dim;
    }
  }

  return success();
}

// ReverseOp canonicalization
void mlir::tt::ttir::ReverseOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // Reverse dimensions of two consecutive ReverseOps can be folded into a
  // single ReverseOp where the dimensions are the symmetric difference of the
  // two sets of dimensions.
  patterns.add(+[](mlir::tt::ttir::ReverseOp op,
                   mlir::PatternRewriter &rewriter) {
    auto producerOp = op.getInput().getDefiningOp<ttir::ReverseOp>();
    if (!producerOp) {
      return mlir::failure();
    }

    llvm::SmallBitVector reverseDimensions(op.getInput().getType().getRank());
    llvm::for_each(op.getDimensions(), [&reverseDimensions](int64_t dim) {
      reverseDimensions.flip(dim);
    });
    llvm::for_each(
        producerOp.getDimensions(),
        [&reverseDimensions](int64_t dim) { reverseDimensions.flip(dim); });

    llvm::SmallVector<int64_t> setIndices;
    llvm::copy_if(llvm::seq<int64_t>(reverseDimensions.size()),
                  std::back_inserter(setIndices),
                  [&](int64_t i) { return reverseDimensions.test(i); });

    rewriter.replaceOpWithNewOp<ttir::ReverseOp>(
        op, op.getType(), producerOp.getInput(), setIndices);
    return success();
  });

  // ReverseOp with empty reverse dimensions is a no-op.
  patterns.add(
      +[](mlir::tt::ttir::ReverseOp op, mlir::PatternRewriter &rewriter) {
        if (!op.getDimensions().empty()) {
          return mlir::failure();
        }

        rewriter.replaceAllOpUsesWith(op, op.getInput());
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

// PermuteOp verification
::mlir::LogicalResult mlir::tt::ttir::PermuteOp::verify() {
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
    return emitOpError("Expected result shape (")
           << ttmlir::utils::join(expectedResultShape, ", ") << "), got ("
           << ttmlir::utils::join(resultShape, ", ") << ")";
  }

  return success();
}

// PermuteOp with identity permutation is a no-op.
// The input can be used directly as the output.
// This includes:
// 1. Sorted permutations like [0, 1, 2, 3]
// 2. Permutations where all swapped dimensions have size 1
//    (e.g., [2, 0, 1, 3] on shape 1x1x1x64 is a no-op)
static mlir::OpFoldResult foldIdentityPermute(mlir::tt::ttir::PermuteOp op) {
  // Case 1: True identity permutation (sorted)
  if (llvm::is_sorted(op.getPermutation())) {
    return op.getInput();
  }

  // Case 2: All non-identity dimension swaps are between dims of size 1
  auto inputShape = op.getInput().getType().getShape();
  for (auto [index, permuteIndex] : llvm::enumerate(op.getPermutation())) {
    if (permuteIndex != static_cast<int64_t>(index)) {
      // This dim position is changed - check if both dims involved are size 1
      if (inputShape[index] != 1 || inputShape[permuteIndex] != 1) {
        return nullptr; // Non-trivial swap
      }
    }
  }
  return op.getInput();
}

// If the producer is a PermuteOp we can compose the permutation attributes
// into `op`, and set the input to the producers input.
static mlir::OpFoldResult foldConsecutivePermute(mlir::tt::ttir::PermuteOp op) {
  // Don't fold decomposed permutes - they were intentionally split.
  if (op->hasAttr("decomposed")) {
    return nullptr;
  }
  if (auto producerOp =
          op.getInput().getDefiningOp<mlir::tt::ttir::PermuteOp>()) {
    llvm::SmallVector<int64_t> composedPermutation =
        ttmlir::utils::applyPermutation(producerOp.getPermutation(),
                                        op.getPermutation());
    op.setPermutation(composedPermutation);
    op->setOperand(0, producerOp.getInput());
    return op.getResult();
  }
  return nullptr;
}

// PermuteOp folder
mlir::OpFoldResult mlir::tt::ttir::PermuteOp::fold(FoldAdaptor adaptor) {

  if (auto foldResult = foldIdentityPermute(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldConsecutivePermute(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

// FullOp verification
mlir::LogicalResult mlir::tt::ttir::FullOp::verify() {
  // Verify that the shape is the shape of the output.
  if (!llvm::equal(getShape(), getType().getShape())) {
    return emitOpError() << "expected shape (" << getType().getShape()
                         << "), got (" << getShape() << ")";
  }

  return mlir::success();
}

static std::optional<std::string>
verifyReplicaGroups(mlir::DenseIntElementsAttr replicaGroups) {
  if (replicaGroups.getType().getRank() != 2) {
    return "replica_groups must be a 2D array";
  }

  auto replicaIds = replicaGroups.getValues<int64_t>();
  int64_t maxId = replicaIds.size() - 1;
  llvm::SmallDenseSet<int64_t> seen;
  for (auto id : replicaIds) {
    if (id < 0) {
      return "replica_groups values must be positive";
    }
    if (id > maxId) {
      return llvm::formatv(
                 "replica_groups values must be in the range [0, {0}], got {1}",
                 maxId, id)
          .str();
    }
    if (!seen.insert(id).second) {
      return "replica_groups must not contain duplicate values";
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

// AllToAllOp verification
::mlir::LogicalResult mlir::tt::ttir::AllToAllOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  auto inShape = inputType.getShape();
  int64_t splitDim = getSplitDim();
  int64_t splitCount = getSplitCount();
  if (splitDim < 0 || splitDim >= inputType.getRank()) {
    return emitOpError("splitDim must be in the range [0, ")
           << inputType.getRank() - 1 << "], got " << splitDim;
  }
  if (splitCount <= 0) {
    return emitOpError("splitCount must be a positive integer");
  }
  if (inShape[splitDim] % splitCount != 0) {
    return emitOpError("splitDim size must be divisible by splitCount");
  }
  int64_t concatDim = getConcatDim();
  if (concatDim < 0 || concatDim >= inputType.getRank()) {
    return emitOpError("concatDim must be in the range [0, ")
           << inputType.getRank() - 1 << "], got " << concatDim;
  }
  ::llvm::SmallVector<int64_t> expectedShape(inShape.begin(), inShape.end());
  expectedShape[splitDim] = expectedShape[splitDim] / splitCount;
  expectedShape[concatDim] = expectedShape[concatDim] * splitCount;
  if (expectedShape != outputType.getShape()) {
    return emitOpError("Output shape mismatch: expected = <")
           << expectedShape << "> output = <" << outputType.getShape() << ">";
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input and output element types must match");
  }
  ::mlir::DenseIntElementsAttr replicaGroups = getReplicaGroups();

  if (auto errorMsg = verifyReplicaGroups(replicaGroups)) {
    return emitOpError() << *errorMsg;
  }
  auto replicaGroupsShape = replicaGroups.getType().getShape();
  if (replicaGroupsShape[1] != splitCount) {
    return emitOpError("replicaGroup count must match splitCount");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

// Common verifier for all Reduce ops.
static mlir::LogicalResult
verifyReduceOp(llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
               mlir::RankedTensorType inputType,
               const std::optional<mlir::ArrayAttr> &reduceDims, bool keepDim,
               ::llvm::ArrayRef<int64_t> specifiedOutputShape) {

  int64_t inputTensorRank = inputType.getRank();

  llvm::BitVector reduceDimsMask(inputTensorRank, false);
  if (!reduceDims) {
    reduceDimsMask.set();
  } else {
    llvm::SmallSet<int64_t, 4> uniqueReduceDims;
    for (mlir::Attribute reduceDim : *reduceDims) {
      int64_t reduceDimInt = mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
      if (reduceDimInt < -inputTensorRank || reduceDimInt >= inputTensorRank) {
        return emitOpError() << "Reduce dimension " << reduceDimInt
                             << " is out of range for input tensor of rank "
                             << inputTensorRank;
      }
      uniqueReduceDims.insert(reduceDimInt);
      reduceDimsMask.set((reduceDimInt + inputTensorRank) % inputTensorRank);
    }

    if (uniqueReduceDims.size() != reduceDims->size()) {
      return emitOpError() << "Reduce dimensions are not unique";
    }
  }

  // Check that the output shape is valid.
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

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

// MaxOp verification.
::mlir::LogicalResult mlir::tt::ttir::MaxOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// MeanOp verification.
::mlir::LogicalResult mlir::tt::ttir::MeanOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

// SumOp verification.
::mlir::LogicalResult mlir::tt::ttir::SumOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce MinOp
//===----------------------------------------------------------------------===//

// MinOp verification.
::mlir::LogicalResult mlir::tt::ttir::MinOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ProdOp
//===----------------------------------------------------------------------===//

// ProdOp verification.
::mlir::LogicalResult mlir::tt::ttir::ProdOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// ReduceAndOp
//===----------------------------------------------------------------------===//

// ReduceAndOp verification.
::mlir::LogicalResult mlir::tt::ttir::ReduceAndOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// ReduceOrOp
//===----------------------------------------------------------------------===//

// ReduceOrOp verification.
::mlir::LogicalResult mlir::tt::ttir::ReduceOrOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ArgMaxOp
//===----------------------------------------------------------------------===//

// ArgMaxOp verification.
::mlir::LogicalResult mlir::tt::ttir::ArgMaxOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

// CumSumOp verification
::mlir::LogicalResult mlir::tt::ttir::CumSumOp::verify() {
  int64_t dim = getDim();
  int64_t inputRank = getInput().getType().getRank();
  if (dim < -inputRank || dim >= inputRank) {
    return emitOpError() << "specified dimension should be between "
                         << -inputRank << " and " << (inputRank - 1)
                         << ", but got: " << dim;
  }

  return success();
}

// CumSumOp folding
::mlir::OpFoldResult mlir::tt::ttir::CumSumOp::fold(FoldAdaptor adaptor) {
  // Normalize `dim` to be in range [0, rank).
  int64_t dim = getDim();
  int64_t rank = getInput().getType().getRank();
  if (dim < 0) {
    setDim(dim + rank);
    return getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BatchNorm verification helpers
//===----------------------------------------------------------------------===//
namespace {
// Shared verification logic for BatchNorm operations
static ::mlir::LogicalResult verifyBatchNormOp(mlir::Operation *op,
                                               mlir::RankedTensorType inputType,
                                               int64_t dimension) {
  int64_t inputRank = inputType.getRank();

  // Input must be 2D to 5D
  if (inputRank < 2 || inputRank > 5) {
    return op->emitOpError(
               "input tensor must have rank between 2 and 5, got rank ")
           << inputRank;
  }

  // Dimension attribute must be within bounds
  if (dimension < 0 || dimension >= inputRank) {
    return op->emitOpError(
               "dimension attribute must be within input rank bounds, "
               "got dimension ")
           << dimension << " for rank " << inputRank;
  }

  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::BatchNormInferenceOp::verify() {
  return verifyBatchNormOp(getOperation(), getOperand().getType(),
                           getDimension());
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::BatchNormTrainingOp::verify() {
  return verifyBatchNormOp(getOperation(), getOperand().getType(),
                           getDimension());
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::CollectiveBroadcastOp::verify() {
  // Check input/output/result types are RankedTensorType
  auto inputType = getInput().getType();
  auto resultType = getType();

  // Check input == output type
  if (inputType != resultType) {
    return emitOpError("input and output must have the same type");
  }

  ::mlir::DenseIntElementsAttr replicaGroups = getReplicaGroups();
  if (auto errorMsg = verifyReplicaGroups(replicaGroups)) {
    return emitOpError() << *errorMsg;
  }

  return success();
}

mlir::OpFoldResult
mlir::tt::ttir::CollectiveBroadcastOp::fold(FoldAdaptor adaptor) {
  auto groupsType = getReplicaGroups().getType();
  // If there is no group, the broadcast is a no-op.
  if (groupsType.getShape()[0] < 1) {
    return getInput();
  }
  // If there is only one device in a group, the broadcast is a no-op.
  if (groupsType.getShape()[1] <= 1) {
    return getInput();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

// ConcatenateHeadsOp verification
::mlir::LogicalResult mlir::tt::ttir::ConcatenateHeadsOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Input tensor must be 4D tensor
  if (inputType.getRank() != 4) {
    return emitOpError() << "expected rank of input tensor is 4, got rank "
                         << inputType.getRank();
  }

  // Output tensor must be 3D tensor.
  if (outputType.getRank() != 3) {
    return emitOpError() << "expected rank of output tensor is 3, got rank "
                         << outputType.getRank();
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Input tensor dimensions [batch_size, num_heads, sequence_size, head_size].
  // Output tensor dimensions [batch_size, sequence_size, num_heads *
  // head_size].
  using namespace ttmlir::utils::transformer;

  // Verify batch_size dimension matches.
  if (inputShape[INPUT_BATCH] != outputShape[OUTPUT_BATCH]) {
    return emitOpError() << "expected output batch dimension to be "
                         << inputShape[INPUT_BATCH] << ", got "
                         << outputShape[OUTPUT_BATCH];
  }

  // Verify sequence_size dimension matches.
  if (inputShape[INPUT_SEQ] != outputShape[OUTPUT_SEQ]) {
    return emitOpError() << "expected output sequence dimension to be "
                         << inputShape[INPUT_SEQ] << ", got "
                         << outputShape[OUTPUT_SEQ];
  }

  // Verify that num_heads * head_size equals the output hidden dimension.
  int64_t expectedHiddenSize =
      inputShape[INPUT_NUM_HEADS] * inputShape[INPUT_HEAD_SIZE];
  if (expectedHiddenSize != outputShape[OUTPUT_HIDDEN]) {
    return emitOpError()
           << "expected output hidden dimension to be num_heads * "
              "head_size = "
           << expectedHiddenSize << ", got " << outputShape[OUTPUT_HIDDEN];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::SplitQueryKeyValueAndSplitHeadsOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();

  // Input tensor must be 3D tensor
  if (inputType.getRank() != 3) {
    return emitOpError() << "expected rank of input tensor is 3, got rank "
                         << inputType.getRank();
  }

  ::mlir::RankedTensorType queryOutputType = getQuery().getType();
  ::mlir::RankedTensorType keyOutputType = getKey().getType();
  ::mlir::RankedTensorType valueOutputType = getValue().getType();

  // Output tensors must be 4D tensors
  if (queryOutputType.getRank() != 4 || keyOutputType.getRank() != 4 ||
      valueOutputType.getRank() != 4) {
    return emitOpError() << "expected rank of query/key/value output tensor is "
                            "4, got query rank: "
                         << queryOutputType.getRank()
                         << ", key rank: " << keyOutputType.getRank()
                         << ", value rank: " << valueOutputType.getRank();
  }

  const uint32_t BATCH_DIM = 0;
  const uint32_t SEQUENCE_LENGTH_DIM = 1;
  const uint32_t HIDDEN_DIMENSION = 2;

  auto inputShape = inputType.getShape();
  auto kvInputShape = getKvInputTensor()
                          ? getKvInputTensor().getType().getShape()
                          : inputType.getShape();

  int64_t numHeads = getNumHeads();
  int64_t numKVHeads = getNumKvHeads() ? *getNumKvHeads() : numHeads;

  int64_t batchSizeQuery = inputShape[BATCH_DIM];
  int64_t sequenceLengthQuery = inputShape[SEQUENCE_LENGTH_DIM];
  int64_t headSizeQuery = 0;

  int64_t batchSizeKeyValue = kvInputShape[BATCH_DIM];
  int64_t sequenceLengthKeyValue = kvInputShape[SEQUENCE_LENGTH_DIM];
  int64_t headSizeKeyValue = 0;

  if (getKvInputTensor()) {
    headSizeQuery = inputShape[HIDDEN_DIMENSION] / numHeads;
    headSizeKeyValue = kvInputShape[HIDDEN_DIMENSION] / (2 * numKVHeads);
  } else {
    headSizeQuery = inputShape[HIDDEN_DIMENSION] / (numHeads + 2 * numKVHeads);
    headSizeKeyValue = headSizeQuery;
  }

  llvm::SmallVector<int64_t, 4> expectedQueryShape = {
      batchSizeQuery, numHeads, sequenceLengthQuery, headSizeQuery};

  if (!llvm::equal(expectedQueryShape, queryOutputType.getShape())) {
    return emitOpError() << "expected query output shape ("
                         << ttmlir::utils::join(expectedQueryShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(queryOutputType.getShape(),
                                                ", ")
                         << ")";
  }

  llvm::SmallVector<int64_t, 4> expectedKeyShape = {
      batchSizeKeyValue, numKVHeads, sequenceLengthKeyValue, headSizeKeyValue};

  if (getTransposeKey()) {
    std::swap(expectedKeyShape[2], expectedKeyShape[3]);
  }

  if (!llvm::equal(expectedKeyShape, keyOutputType.getShape())) {
    return emitOpError() << "expected key output shape ("
                         << ttmlir::utils::join(expectedKeyShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(keyOutputType.getShape(), ", ")
                         << ")";
  }

  llvm::SmallVector<int64_t, 4> expectedValueShape = {
      batchSizeKeyValue, numKVHeads, sequenceLengthKeyValue, headSizeKeyValue};

  if (!llvm::equal(expectedValueShape, valueOutputType.getShape())) {
    return emitOpError() << "expected value output shape ("
                         << ttmlir::utils::join(expectedValueShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(valueOutputType.getShape(),
                                                ", ")
                         << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::RMSNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  // Input and output must have the same shape.
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  // Verify normalized_shape is valid for the input tensor.
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> normalizedShape = getNormalizedShape();

  // Check that normalized_shape is not empty.
  if (normalizedShape.empty()) {
    return emitOpError("normalized_shape cannot be empty");
  }

  // Check that normalized_shape is not larger than input tensor shape.
  if (normalizedShape.size() > inputShape.size()) {
    return emitOpError(
        "normalized_shape has more dimensions than input tensor");
  }

  // Check that the trailing dimensions of input match normalized_shape.
  // For example, if input shape is [2, 3, 4, 5] and normalized_shape is [4, 5],
  // then we check that input shape's last two dimensions (4, 5) match
  // normalized_shape.
  size_t offset = inputShape.size() - normalizedShape.size();
  for (size_t i = 0; i < normalizedShape.size(); ++i) {
    if (inputShape[offset + i] != normalizedShape[i]) {
      return emitOpError("normalized_shape dimensions must match trailing "
                         "dimensions of input tensor");
    }
  }

  // Verify weight tensor shape if present.
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getShape() != normalizedShape) {
      return emitOpError("weight tensor shape must match normalized_shape");
    }
  }

  // Verify bias tensor shape if present.
  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getShape() != normalizedShape) {
      return emitOpError("bias tensor shape must match normalized_shape");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::LayerNormOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> normalizedShape = getNormalizedShape();

  if (normalizedShape.empty()) {
    return emitOpError("normalized_shape cannot be empty");
  }

  if (normalizedShape.size() > inputShape.size()) {
    return emitOpError(
        "normalized_shape has more dimensions than input tensor");
  }

  // Check that the trailing dimensions of input match normalized_shape.
  // For example, if input shape is [2, 3, 4, 5] and normalized_shape is [4, 5],
  // then we check that input shape's last two dimensions (4, 5) match
  // normalized_shape.
  size_t offset = inputShape.size() - normalizedShape.size();
  for (size_t i = 0; i < normalizedShape.size(); ++i) {
    if (inputShape[offset + i] != normalizedShape[i]) {
      return emitOpError("normalized_shape dimensions must match trailing "
                         "dimensions of input tensor");
    }
  }

  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    if (weightType.getShape() != normalizedShape) {
      return emitOpError("weight tensor shape must match normalized_shape");
    }
  }

  if (getBias()) {
    RankedTensorType biasType = getBias().getType();
    if (biasType.getShape() != normalizedShape) {
      return emitOpError("bias tensor shape must match normalized_shape");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::ScaledDotProductAttentionDecodeOp::verify() {

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
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PagedScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult
mlir::tt::ttir::PagedScaledDotProductAttentionDecodeOp::verify() {

  RankedTensorType queryType = getQuery().getType();
  RankedTensorType keyType = getKey().getType();
  RankedTensorType valueType = getValue().getType();
  RankedTensorType pageTableType = getPageTable().getType();

  auto queryShape = queryType.getShape();
  auto keyShape = keyType.getShape();
  auto pageTableShape = pageTableType.getShape();

  auto numUsers = queryShape[1];
  auto numQueryHeads = queryShape[2];
  auto headSize = queryShape[3];
  auto numKVHeads = keyShape[1];
  auto blockSize = keyShape[2];

  // Verify element types.
  if (queryType.getElementType() != keyType.getElementType() ||
      queryType.getElementType() != valueType.getElementType()) {
    return emitOpError(
        "Query, key, and value must have the same element type.");
  }

  if (!queryType.getElementType().isFloat()) {
    return emitOpError("Query, key, and value must be float tensors.");
  }

  if (!pageTableType.getElementType().isInteger()) {
    return emitOpError("Page table must be an integer tensor.");
  }

  // Verify key and value are identical shapes/dtypes
  if (keyType != valueType) {
    return emitOpError("Key and value must have the same shape and data type.");
  }

  // Verify ranks.
  if (queryType.getShape().size() != 4) {
    return emitOpError("Query must be a 4D tensor.");
  }
  if (keyType.getShape().size() != 4) {
    return emitOpError("Key/Value tensor must be a 4D tensor.");
  }
  if (pageTableType.getShape().size() != 2) {
    return emitOpError("Page table tensor must be a 2D tensor.");
  }

  // Verify shapes.
  if (headSize != keyShape[3]) {
    return emitOpError("Query head size must match key/value head size.");
  }
  if (numQueryHeads % numKVHeads != 0) {
    return emitOpError(
        "Query num heads must be divisible by key/value num heads.");
  }
  if (blockSize % 32 != 0) {
    return emitOpError("Block size must be divisible by 32.");
  }

  if (pageTableShape[0] != numUsers) {
    return emitOpError(
        "Page table number of users must match query number of users.");
  }

  bool isCausal = getIsCausal();
  if (isCausal) {
    if (getAttentionMask()) {
      return emitOpError(
          "Attention mask is not allowed when is_causal is true.");
    }
  } else {
    if (!getAttentionMask()) {
      return emitOpError("Attention mask is required when is_causal is false.");
    }
  }

  if (getCurPosTensor()) {
    if (!getCurPosTensor().getType().getElementType().isInteger()) {
      return emitOpError("Cur pos tensor must be an integer tensor.");
    }
    if (getCurPosTensor().getType().getShape().size() != 1) {
      return emitOpError("Cur pos tensor must be a 1D tensor.");
    }
    if (getCurPosTensor().getType().getShape()[0] != numUsers) {
      return emitOpError(
          "Cur pos tensor number of users must match query number of users.");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ScaledDotProductAttentionOp::verify() {

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
  }

  if (getIsCausal()) {
    if (seqLen != maxSeqLen) {
      return emitOpError("Sequence length must match key/value sequence length "
                         "when is_causal is true");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalAvgPool2dOp
//===----------------------------------------------------------------------===//

// GlobalAvgPool2dOp verification
::mlir::LogicalResult mlir::tt::ttir::GlobalAvgPool2dOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t rank = inputType.getRank();
  if (rank < 2) {
    return emitOpError("input tensor must have at least 2 dimensions for "
                       "global average pooling over 2 spatial dimensions");
  }

  if (outputType.getRank() != rank) {
    return emitOpError("output tensor must have the same rank as input tensor");
  }

  if (inputShape[0] != outputShape[0]) {
    return emitOpError(
        "batch dimension must remain the same between input and output");
  }

  if (outputShape[rank - 2] != 1 || outputShape[rank - 3] != 1) {
    return emitOpError("spatial dimensions must be reduced to 1");
  }

  if (inputShape[rank - 1] != outputShape[rank - 1]) {
    return emitOpError(
        "channel dimension must remain the same between input and output");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GeluBackwardOp
//===----------------------------------------------------------------------===//

// GeluBackwardOp verification
::mlir::LogicalResult mlir::tt::ttir::GeluBackwardOp::verify() {
  llvm::StringRef approximate = getApproximate();

  if (approximate != "none" && approximate != "tanh") {
    return emitOpError("approximate attribute must be either 'none' or 'tanh', "
                       "but got '")
           << approximate << "'";
  }

  RankedTensorType lhsType = getLhs().getType();
  RankedTensorType rhsType = getRhs().getType();

  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();

  if (lhsRank < 2 || lhsRank > 4) {
    return emitOpError(
               "gradient tensor (lhs) must have rank 2, 3, or 4, but got rank ")
           << lhsRank;
  }

  if (rhsRank < 2 || rhsRank > 4) {
    return emitOpError(
               "input tensor (rhs) must have rank 2, 3, or 4, but got rank ")
           << rhsRank;
  }

  if (lhsRank != rhsRank) {
    return emitOpError("gradient tensor (lhs) and input tensor (rhs) must have "
                       "the same rank, "
                       "but got lhs rank ")
           << lhsRank << " and rhs rank " << rhsRank;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseMatmulOp
//===----------------------------------------------------------------------===//

// SparseMatmulOp verification
::mlir::LogicalResult mlir::tt::ttir::SparseMatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType sparsityType = getSparsity().getType();
  ::mlir::RankedTensorType outputType = getType();

  // Verify that input A is at least 4D tensor
  if (inputAType.getRank() < 4) {
    return emitOpError("Input A must be at least a 4D tensor");
  }

  // Verify that input B is exactly 4D tensor [1, E, K, N]
  if (inputBType.getRank() != 4) {
    return emitOpError("Input B must be a 4D tensor [1, E, K, N]");
  }

  // Verify that sparsity is a 4D tensor
  if (sparsityType.getRank() != 4) {
    return emitOpError("Sparsity tensor must be a 4D tensor");
  }

  bool isInputASparse = getIsInputASparse();
  bool isInputBSparse = getIsInputBSparse();

  // Verify that at least one input is sparse
  if (!isInputASparse && !isInputBSparse) {
    return emitOpError("At least one of is_input_a_sparse or is_input_b_sparse "
                       "must be true");
  }

  // Get dimensions
  int64_t numExperts = inputBType.getShape()[1];
  int64_t K = inputBType.getShape()[2];
  (void)inputBType.getShape()[3]; // N - used for output shape validation

  // Verify input B first dimension is 1
  if (inputBType.getShape()[0] != 1) {
    return emitOpError("Input B first dimension must be 1");
  }

  // Verify inner dimensions match
  if (inputAType.getShape()[inputAType.getRank() - 1] != K) {
    return emitOpError(
        "Input A inner dimension must match Input B K dimension");
  }

  // Verify sparsity tensor has correct number of experts
  if (sparsityType.getShape()[sparsityType.getRank() - 1] != numExperts) {
    return emitOpError("Sparsity tensor last dimension must match number of "
                       "experts in Input B");
  }

  // Verify output shape based on sparse mode
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  if (isInputASparse && isInputBSparse) {
    // [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
    if (outputShape.size() != 4) {
      return emitOpError("Output must be 4D for sparse-sparse mode");
    }
  } else if (!isInputASparse && isInputBSparse) {
    // [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
    if (outputShape.size() != 6) {
      return emitOpError("Output must be 6D for dense-sparse mode");
    }
  } else if (isInputASparse && !isInputBSparse) {
    // [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
    if (outputShape.size() != 4) {
      return emitOpError("Output must be 4D for sparse-dense mode");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllDispatchOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::AllToAllDispatchOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();
  ::mlir::RankedTensorType indicesType = getExpertIndices().getType();
  ::mlir::RankedTensorType mappingType = getExpertMapping().getType();
  ::mlir::RankedTensorType dispatchedType = getDispatched().getType();
  ::mlir::RankedTensorType metadataType = getMetadata().getType();

  // All tensors must be 4D
  if (inputType.getRank() != 4) {
    return emitOpError("input_tensor must be a 4D tensor [B, S, 1, H]");
  }
  if (indicesType.getRank() != 4) {
    return emitOpError("expert_indices must be a 4D tensor [B, S, 1, K]");
  }
  if (mappingType.getRank() != 4) {
    return emitOpError("expert_mapping must be a 4D tensor [1, 1, E, D]");
  }
  if (dispatchedType.getRank() != 4) {
    return emitOpError("dispatched output must be a 4D tensor [1, B*D, S, H]");
  }
  if (metadataType.getRank() != 4) {
    return emitOpError("metadata output must be a 4D tensor [1, B*D, S, K]");
  }

  // Verify num_devices > 0
  if (getNumDevices() <= 0) {
    return emitOpError("num_devices must be positive");
  }

  // Verify cluster_axis is 0, 1, or -1 (all devices)
  int64_t clusterAxis = static_cast<int64_t>(getClusterAxis());
  if (clusterAxis > 1 || clusterAxis < -1) {
    return emitOpError("cluster_axis must be 0, 1, or -1 (all devices)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllCombineOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::AllToAllCombineOp::verify() {
  ::mlir::RankedTensorType inputType = getInputTensor().getType();
  ::mlir::RankedTensorType metadataType = getExpertMetadata().getType();
  ::mlir::RankedTensorType mappingType = getExpertMapping().getType();
  ::mlir::RankedTensorType resultType = getResult().getType();

  // All tensors must be 4D
  if (inputType.getRank() != 4) {
    return emitOpError("input_tensor must be a 4D tensor [E_local, B*D, S, H]");
  }
  if (metadataType.getRank() != 4) {
    return emitOpError("expert_metadata must be a 4D tensor [1, B*D, S, K]");
  }
  if (mappingType.getRank() != 4) {
    return emitOpError("expert_mapping must be a 4D tensor [1, 1, E, D]");
  }
  if (resultType.getRank() != 4) {
    return emitOpError("result must be a 4D tensor [K, B, S, H]");
  }

  // Verify num_devices > 0
  if (getNumDevices() <= 0) {
    return emitOpError("num_devices must be positive");
  }

  // Verify cluster_axis is 0, 1, or -1 (all devices)
  int64_t clusterAxis = static_cast<int64_t>(getClusterAxis());
  if (clusterAxis > 1 || clusterAxis < -1) {
    return emitOpError("cluster_axis must be 0, 1, or -1 (all devices)");
  }

  // Verify num_experts_per_tok > 0
  if (getNumExpertsPerTok() <= 0) {
    return emitOpError("num_experts_per_tok must be positive");
  }

  return success();
}

} // namespace mlir::tt::ttir
