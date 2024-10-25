// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <stablehlo/dialect/StablehloOps.h>

using namespace mlir;
using namespace mlir::tt;

namespace {

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIROpDefaultConversionPattern
    : public OpConversionPattern<SrcOp> {

  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp,
        TypeRange(
            this->getTypeConverter()->convertType(outputTensor.getType())),
        adaptor.getOperands(), ValueRange(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }
};

class StableHLOToTTIRReduceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceOp> {

  using OpConversionPattern<mlir::stablehlo::ReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceOp srcOp,
                  mlir::stablehlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    const mlir::Operation &innerOp = srcOp.getBody().front().front();

    if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::SumOp>(srcOp, adaptor,
                                                            rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MaxOp>(srcOp, adaptor,
                                                            rewriter);
    }

    return failure();
  }

private:
  LogicalResult checkBasicLegality(mlir::stablehlo::ReduceOp &srcOp,
                                   mlir::stablehlo::ReduceOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    if (!srcOp.getBody().hasOneBlock()) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Expecting StableHLO Reduce OP to have one block inside its body.");
    }

    if (srcOp.getBody().front().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Expecting StableHLO Reduce OP to have a body operation defined.");
    }

    return success();
  }

  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::ReduceOp &srcOp,
                          mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes().front()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    mlir::ArrayAttr dimArg = rewriter.getArrayAttr(SmallVector<Attribute>(
        1, rewriter.getI32IntegerAttr(adaptor.getDimensionsAttr()[0])));

    // If someone changes definition of TTIR_ReductionOp this constant will
    // become outdated, but I currently see no way to get this info (without
    // manually constructing the adaptor for dest OP).
    const std::size_t ttirReduceOpOperandsCount = 2;
    mlir::ArrayAttr operandConstraints =
        rewriter.getArrayAttr(SmallVector<Attribute>(
            ttirReduceOpOperandsCount, rewriter.getAttr<OperandConstraintAttr>(
                                           OperandConstraint::AnyDeviceTile)));

    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp, outputType, adaptor.getInputs().front(), outputTensor,
        false /* keep_dim */, dimArg, operandConstraints);

    return success();
  }
};

class StableHLOToTTIRTransposeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern<mlir::stablehlo::TransposeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::TransposeOp srcOp,
                  mlir::stablehlo::TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::TransposeOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        Value(adaptor.getOperand()), Value(outputTensor),
        adaptor.getPermutation()[0], adaptor.getPermutation()[1],
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

  LogicalResult
  checkBasicLegality(mlir::stablehlo::TransposeOp &srcOp,
                     mlir::stablehlo::TransposeOp::Adaptor &adaptor,
                     ConversionPatternRewriter &rewriter) const {

    if (adaptor.getPermutation().size() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp, "TTIR supports only two dimensional transposeOp.");
    }

    return success();
  }
};

class StableHLOToTTIRReshapeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern<mlir::stablehlo::ReshapeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReshapeOp srcOp,
                  mlir::stablehlo::ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    std::vector<int32_t> new_shape_i32;
    for (int64_t dim : outputType.getShape()) {
      new_shape_i32.push_back(static_cast<int32_t>(dim));
    }
    ArrayAttr new_shape_attr = rewriter.getI32ArrayAttr(new_shape_i32);
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ReshapeOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        srcOp->getOperand(0), outputTensor, new_shape_attr,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

  LogicalResult
  checkBasicLegality(mlir::stablehlo::TransposeOp &srcOp,
                     mlir::stablehlo::TransposeOp::Adaptor &adaptor,
                     ConversionPatternRewriter &rewriter) const {

    if (adaptor.getPermutation().size() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp, "TTIR supports only two dimensional transposeOp.");
    }

    return success();
  }
};

class StableHLOToTTIRDotGeneralOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern<mlir::stablehlo::DotGeneralOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp srcOp,
                  mlir::stablehlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This is a basic version that can only work for cases that can be directly
    // converted to matmul. The op should be extended as other ops such as
    // ttir.permute and ttir.broadcast_in_dim become available.

    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::MatmulOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        adaptor.getLhs(), adaptor.getRhs(), Value(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::DotGeneralOp &srcOp,
                     mlir::stablehlo::DotGeneralOp::Adaptor &adaptor,
                     ConversionPatternRewriter &rewriter) const {

    ::mlir::stablehlo::DotDimensionNumbersAttr dimensions =
        adaptor.getDotDimensionNumbers();

    if (dimensions.getLhsContractingDimensions().empty() ||
        dimensions.getRhsContractingDimensions().empty()) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Contracting dimension is missing.");
    }

    if (dimensions.getLhsContractingDimensions()[0] != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    if (dimensions.getRhsContractingDimensions()[0] != 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    if (!dimensions.getLhsBatchingDimensions().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    if (!dimensions.getRhsBatchingDimensions().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only non-transposed matmul is currently supported in TTIR.");
    }

    return success();
  }
};

class StableHLOToTTIRConstantOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {

  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp srcOp,
                  mlir::stablehlo::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    mlir::ElementsAttr valueAttr = getValueAttr(srcOp.getValue());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(srcOp, outputType,
                                                            valueAttr);
    return success();
  }

private:
  LogicalResult checkBasicLegality(mlir::stablehlo::ConstantOp &srcOp,
                                   ConversionPatternRewriter &rewriter) const {
    if (srcOp.getValue().getShapedType().getShape().empty() &&
        !srcOp.getValue().getElementType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(srcOp, "Unsupported element type.");
    }

    return success();
  }

  // Rebuilding value of constant op for following cases.
  // 1. Scalar values: TTNN does not support scalar types. So they are converted
  //    1-D tensors.
  // 2. Boolean tensor: TTNN does not support boolean data. So they are
  //    converted to bfloat16 tensors.
  // 3. Integer tensor: TTNN does not support 64 bit integer. So they are
  //    converted to 32 bit tensor.
  mlir::ElementsAttr getValueAttr(mlir::ElementsAttr valueAttr) const {
    Type elementType = valueAttr.getElementType();
    size_t bitWidth = elementType.getIntOrFloatBitWidth();
    bool isTensor = !valueAttr.getShapedType().getShape().empty();
    bool isIntTensor = isTensor && isa<IntegerType>(elementType) &&
                       bitWidth != 1 && bitWidth != 64;
    bool isFloatTensor = isTensor && isa<FloatType>(elementType);

    if (isTensor && (isIntTensor || isFloatTensor)) {
      return valueAttr;
    }

    mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(
        getTypeConverter()->convertType(valueAttr.getShapedType()));
    if (isa<IntegerType>(elementType)) {
      switch (bitWidth) {
      case 1: {
        return rebuildValueAttr<bool>(valueAttr, 1);
      }
      case 8: {
        return rebuildValueAttr<int8_t>(valueAttr, 8);
      }
      case 16: {
        return rebuildValueAttr<int16_t>(valueAttr, 16);
      }
      case 32: {
        return rebuildValueAttr<int32_t>(valueAttr, 32);
      }
      case 64: {
        return rebuildValueAttr<int64_t>(valueAttr, 32);
      }
      default: {
        assert(false && "Unsupported integer type.");
      }
      }
    }
    if (isa<FloatType>(elementType)) {
      // In case of float values llvm has a bug where not all float types are
      // supported for iterating in DenseElementsAttr, so we have to use a
      // different constructor.
      std::vector<mlir::APFloat> floatValues(
          valueAttr.getValues<mlir::APFloat>().begin(),
          valueAttr.getValues<mlir::APFloat>().end());
      return mlir::DenseElementsAttr::get(valueType, floatValues);
    }
    assert(false && "Unsupported data type.");
  }

  // Extract the values (using the given ElementType) and create new data
  // structure. This is used to convert scalars (of type boolean, int8, int16,
  // int32, and int64) and tensors (of type boolean and int64).
  template <typename ElementType>
  mlir::ElementsAttr rebuildValueAttr(mlir::ElementsAttr valueAttr,
                                      size_t bitWidth) const {
    mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(
        getTypeConverter()->convertType(valueAttr.getShapedType()));

    // Create data structure for boolean type with bfloat16.
    if (bitWidth == 1) {
      std::vector<mlir::APFloat> booleanValue = {};
      for (ElementType value : valueAttr.getValues<ElementType>()) {
        mlir::APFloat input(mlir::APFloat::BFloat(), value);
        booleanValue.emplace_back(input);
      }
      return mlir::DenseElementsAttr::get(valueType, booleanValue);
    }

    // Create data structure for other types.
    std::vector<mlir::APInt> IntegerValue = {};
    for (ElementType value : valueAttr.getValues<ElementType>()) {
      mlir::APInt input(bitWidth, value);
      IntegerValue.emplace_back(input);
    }
    return mlir::DenseElementsAttr::get(valueType, IntegerValue);
  }
};

class StableHLOToTTIRConvolutionOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using OpConversionPattern<
      mlir::stablehlo::ConvolutionOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp srcOp,
                  mlir::stablehlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    std::vector<int64_t> strides =
        adaptor.getWindowStrides().value_or(ArrayRef<int64_t>({1, 1})).vec();
    IntegerAttr stride_height_attr =
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(strides[0]));
    IntegerAttr stride_width_attr =
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(strides[1]));

    std::vector<int64_t> dilation =
        adaptor.getLhsDilation().value_or(ArrayRef<int64_t>({1, 1})).vec();

    IntegerAttr dilation_height_attr =
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(dilation[0]));
    IntegerAttr dilation_width_attr =
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(dilation[1]));

    IntegerAttr groups_attr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(adaptor.getFeatureGroupCount()));

    std::vector<int32_t> padding;
    if (!adaptor.getPadding().has_value()) {
      padding = {0, 0, 0, 0};
    } else {
      for (auto iter = adaptor.getPadding()->value_begin<int64_t>();
           iter < adaptor.getPadding()->value_end<int64_t>(); iter++) {
        padding.push_back(static_cast<int32_t>(*iter));
      }
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::Conv2dOp>(
        srcOp, outputType, adaptor.getLhs(), adaptor.getRhs(),
        mlir::Value(nullptr), outputTensor, stride_height_attr,
        stride_width_attr, dilation_height_attr, dilation_width_attr,
        groups_attr, rewriter.getSI32IntegerAttr(padding[0]),
        rewriter.getSI32IntegerAttr(padding[1]),
        rewriter.getSI32IntegerAttr(padding[2]),
        rewriter.getSI32IntegerAttr(padding[3]),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }
};

class StableHLOToTTIRReduceWindowOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern<
      mlir::stablehlo::ReduceWindowOp>::OpConversionPattern;

public:
  void recursiveErase(ConversionPatternRewriter &rewriter,
                      Operation *op) const {
    for (auto &operand : op->getOpOperands()) {
      recursiveErase(rewriter, operand.get().getDefiningOp());
    }
    rewriter.eraseOp(op);
  }

  bool isMaxPool2d(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    if (srcOp.getBody().getBlocks().size() != 1) {
      return false;
    }

    Block &block = *srcOp.getBody().getBlocks().begin();
    uint32_t op_idx = 0;
    for (Operation &op : block) {
      if (op_idx == 0 && !isa<mlir::stablehlo::MaxOp>(op)) {
        return false;
      }
      if (op_idx == 1 && !isa<mlir::stablehlo::ReturnOp>(op)) {
        return false;
      }
      if (op_idx >= 2) {
        return false; // More than two ops in the block
      }
      op_idx++;
    }

    return true;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp srcOp,
                  mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (isMaxPool2d(srcOp)) {

      RankedTensorType outputType = mlir::cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp.getResult(0).getType()));

      tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

      // The generalized ReduceWindow allows for kernel_size, strides, dilation,
      // and padding to act on all 4 input dimensions. Since we only support
      // channel-last pooling, we select the middle two values for H and W.
      // And fail if the others are not 1 (or 0 in the case of padding).
      std::vector<int64_t> window_dimensions = adaptor.getWindowDimensions();
      if (window_dimensions[0] != 1 || window_dimensions[3] != 1) {
        return failure();
      }
      IntegerAttr kernel_height_attr = rewriter.getSI32IntegerAttr(
          static_cast<int32_t>(window_dimensions[1]));
      IntegerAttr kernel_width_attr = rewriter.getSI32IntegerAttr(
          static_cast<int32_t>(window_dimensions[2]));

      std::vector<int64_t> strides =
          adaptor.getWindowStrides()
              .value_or(ArrayRef<int64_t>({1, 1, 1, 1}))
              .vec();

      if (strides[0] != 1 || strides[3] != 1) {
        return failure();
      }
      IntegerAttr stride_height_attr =
          rewriter.getSI32IntegerAttr(static_cast<int32_t>(strides[1]));
      IntegerAttr stride_width_attr =
          rewriter.getSI32IntegerAttr(static_cast<int32_t>(strides[2]));

      std::vector<int64_t> dilation =
          adaptor.getBaseDilations()
              .value_or(ArrayRef<int64_t>({1, 1, 1, 1}))
              .vec();

      if (dilation[0] != 1 || dilation[3] != 1) {
        return failure();
      }
      IntegerAttr dilation_height_attr =
          rewriter.getSI32IntegerAttr(static_cast<int32_t>(dilation[1]));
      IntegerAttr dilation_width_attr =
          rewriter.getSI32IntegerAttr(static_cast<int32_t>(dilation[2]));

      // Padding here is in the form ((., .), (top, bottom), (left, right), (.,
      // .)) one for each of (N, H, W, C). Since we only support maxpool2d, the
      // first and last padding tuples must be zero to be valid. This list is
      // flattened so we can use a single iterator to get the values.
      std::vector<int32_t> padding = {0, 0, 0, 0};
      if (adaptor.getPadding().has_value()) {
        uint32_t pad_idx = 0;
        for (auto iter = adaptor.getPadding()->value_begin<int64_t>();
             iter < adaptor.getPadding()->value_end<int64_t>(); iter++) {

          // TTIR requires left, right, top, bottom
          if (pad_idx == 2) {
            padding[2] = *iter;
          } else if (pad_idx == 3) {
            padding[3] = *iter;
          } else if (pad_idx == 4) {
            padding[0] = *iter;
          } else if (pad_idx == 5) {
            padding[1] = *iter;
          } else if (*iter != 0) {
            // Padding on the channel or batch is > 1. TTIR/TTNN does not
            // support this.
            return failure();
          }
          pad_idx++;
        }
      }
      ::llvm::ArrayRef<int64_t> input_shape =
          mlir::cast<mlir::RankedTensorType>(adaptor.getInputs()[0].getType())
              .getShape();

      // Dead ttir.constant sticks around and fails verification. Removing it
      // like so since its behind another op
      recursiveErase(rewriter, adaptor.getInitValues()[0].getDefiningOp());
      rewriter.replaceOpWithNewOp<mlir::tt::ttir::MaxPool2dOp>(
          srcOp, outputType, srcOp.getInputs()[0], outputTensor,
          kernel_height_attr, kernel_width_attr, stride_height_attr,
          stride_width_attr, dilation_height_attr, dilation_width_attr,
          rewriter.getBoolAttr(false), rewriter.getSI32IntegerAttr(padding[0]),
          rewriter.getSI32IntegerAttr(padding[1]),
          rewriter.getSI32IntegerAttr(padding[2]),
          rewriter.getSI32IntegerAttr(padding[3]),
          rewriter.getArrayAttr(
              SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                     rewriter.getAttr<OperandConstraintAttr>(
                                         OperandConstraint::AnyDeviceTile))),
          rewriter.getSI32IntegerAttr(input_shape[1]),
          rewriter.getSI32IntegerAttr(input_shape[2]));

      return success();
    }
    return failure();
  }
};

class StableHLOToTTIRBroadcastInDimOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp srcOp,
                  mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    mlir::ArrayAttr dimArg =
        rewriter.getI64ArrayAttr(adaptor.getBroadcastDimensions());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::BroadcastOp>(
        srcOp, getTypeConverter()->convertType(outputTensor.getType()),
        Value(adaptor.getOperand()), Value(outputTensor), dimArg,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::BroadcastInDimOp &srcOp,
                     mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    llvm::SmallVector<int64_t, 4> broadcastedShape;
    auto srcType =
        getTypeConverter()->convertType(srcOp.getOperand().getType());
    auto inputShape = mlir::cast<mlir::RankedTensorType>(srcType).getShape();
    auto outputShape = mlir::cast<mlir::RankedTensorType>(srcType).getShape();

    if (!OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Input cannot be broadcasted to provided dimensions.");
    }

    return success();
  }
};

class StableHLOToTTIRCompareOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompareOp> {
  using OpConversionPattern<mlir::stablehlo::CompareOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompareOp srcOp,
                  mlir::stablehlo::CompareOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // StableHLO has one 'compare' op to do all type of comparison (EQ, NE, GE,
    // GT, LE, and LT) and use direction to determine the type of comparison.
    mlir::stablehlo::ComparisonDirection direction =
        srcOp.getComparisonDirection();

    switch (direction) {
    case mlir::stablehlo::ComparisonDirection::EQ: {
      return matchAndRewriteInternal<mlir::tt::ttir::EqualOp>(srcOp, adaptor,
                                                              rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::NE: {
      return matchAndRewriteInternal<mlir::tt::ttir::NotEqualOp>(srcOp, adaptor,
                                                                 rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::GE: {
      return matchAndRewriteInternal<mlir::tt::ttir::GreaterEqualOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::GT: {
      return matchAndRewriteInternal<mlir::tt::ttir::GreaterThanOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::LE: {
      return matchAndRewriteInternal<mlir::tt::ttir::LessEqualOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::LT: {
      return matchAndRewriteInternal<mlir::tt::ttir::LessThanOp>(srcOp, adaptor,
                                                                 rewriter);
      break;
    }
    }
    return success();
  }

private:
  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::CompareOp srcOp,
                          mlir::stablehlo::CompareOp::Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    mlir::RankedTensorType outputType =
        mlir::cast<RankedTensorType>(this->getTypeConverter()->convertType(
            srcOp->getResults()[0].getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp,
        TypeRange(
            this->getTypeConverter()->convertType(outputTensor.getType())),
        adaptor.getOperands(), ValueRange(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));

    return success();
  }
};

class StableHLOToTTIRConcatOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConcatenateOp> {

  using OpConversionPattern<
      mlir::stablehlo::ConcatenateOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConcatenateOp srcOp,
                  mlir::stablehlo::ConcatenateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // Create an empty output tensor with the computed shape
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    // Replace the original ConcatOp with the destination operation
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConcatOp>(
        srcOp,
        outputType,          // result type
        adaptor.getInputs(), // input values
        Value(outputTensor), // output value
        rewriter.getSI32IntegerAttr(
            static_cast<int32_t>(adaptor.getDimension())), // dimension
        rewriter.getArrayAttr( // operand constraints
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::ConcatenateOp &srcOp,
                     mlir::stablehlo::ConcatenateOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (srcOp.getInputs().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "ConcatOp must have at least one input.");
    }
    if (adaptor.getDimension() >=
        INT32_MAX) { // stablehlo.concatenate dimension is i64,
                     // ttir.concat dimension is si32
      return rewriter.notifyMatchFailure(srcOp,
                                         "ConcatOp dimension is too large.");
    }

    auto rankedTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(srcOp.getOperand(0).getType());
    if (static_cast<int64_t>(adaptor.getDimension()) >=
        rankedTensorType.getRank()) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid concatenation dimension.");
    }

    return success();
  }
};

template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIROpLogicalOpConversionPattern
    : public OpConversionPattern<SrcOp> {

  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp,
        TypeRange(
            this->getTypeConverter()->convertType(outputTensor.getType())),
        adaptor.getOperands(), ValueRange(outputTensor),
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }

private:
  LogicalResult checkBasicLegality(SrcOp srcOp, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    if (mlir::cast<RankedTensorType>(srcOp->getOperand(0).getType())
                .getElementTypeBitWidth() > 1 &&
        mlir::cast<RankedTensorType>(srcOp->getOperand(1).getType())
                .getElementTypeBitWidth() > 1) {
      llvm::errs()
          << "error: TTIR does not support bitwise logical operation.\n";
      return rewriter.notifyMatchFailure(
          srcOp, "TTIR does not support bitwise logical operation.");
    }

    return success();
  }
};

class StableHLOToTTIRSliceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::SliceOp> {

  using OpConversionPattern<mlir::stablehlo::SliceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::SliceOp srcOp,
                  mlir::stablehlo::SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // Create an empty output tensor with the computed shape
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    std::vector<int32_t> start_indices(adaptor.getStartIndices().begin(),
                                       adaptor.getStartIndices().end());
    std::vector<int32_t> end_indices(adaptor.getLimitIndices().begin(),
                                     adaptor.getLimitIndices().end());
    std::vector<int32_t> step(adaptor.getStrides().begin(),
                              adaptor.getStrides().end());

    // Replace the original ConcatOp with the destination operation
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::SliceOp>(
        srcOp,
        outputType,           // result type
        adaptor.getOperand(), // input values
        outputTensor,         // output value
        rewriter.getI32ArrayAttr(start_indices),
        rewriter.getI32ArrayAttr(end_indices), rewriter.getI32ArrayAttr(step),
        rewriter.getArrayAttr( // operand constraints
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }
};

void addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AbsOp, mlir::tt::ttir::AbsOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::CbrtOp, mlir::tt::ttir::CbrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ConvertOp, mlir::tt::ttir::TypecastOp>>(typeConverter,
                                                               ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::CeilOp, mlir::tt::ttir::CeilOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::CosineOp, mlir::tt::ttir::CosOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ExpOp, mlir::tt::ttir::ExpOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::NegOp, mlir::tt::ttir::NegOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SineOp, mlir::tt::ttir::SinOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SqrtOp, mlir::tt::ttir::SqrtOp>>(typeConverter, ctx);
}

void addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AddOp, mlir::tt::ttir::AddOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::DivOp, mlir::tt::ttir::DivOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MaxOp, mlir::tt::ttir::MaximumOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MinOp, mlir::tt::ttir::MinimumOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MulOp, mlir::tt::ttir::MultiplyOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SubtractOp, mlir::tt::ttir::SubtractOp>>(typeConverter,
                                                                ctx);
}

void addReduceOpsConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceOpConversionPattern>(typeConverter, ctx);
}

void addTransposeOpsConversionPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIRTransposeOpConversionPattern>(typeConverter, ctx);
}

void addMatmulOpsConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRDotGeneralOpConversionPattern>(typeConverter,
                                                             ctx);
}

void addTensorCreationOpsConversionPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConstantOpConversionPattern>(typeConverter, ctx);
}

void addBroadcastOpConversionPattern(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIRBroadcastInDimOpConversionPattern>(typeConverter,
                                                                 ctx);
}

void addConv2dOpConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConvolutionOpConversionPattern>(typeConverter,
                                                              ctx);
}

void addReduceWindowOpConversionPattern(MLIRContext *ctx,
                                        RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceWindowOpConversionPattern>(typeConverter,
                                                               ctx);
}

void addCompareOpsConversionPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRCompareOpConversionPattern>(typeConverter, ctx);
}

void addConcatOpsConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConcatOpConversionPattern>(typeConverter, ctx);
}

void addReshapeOpConversionPattern(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReshapeOpConversionPattern>(typeConverter, ctx);
}

void addLogicalOpConversionPattern(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpLogicalOpConversionPattern<
      mlir::stablehlo::AndOp, mlir::tt::ttir::LogicalAndOp>>(typeConverter,
                                                             ctx);
  patterns.add<StableHLOToTTIROpLogicalOpConversionPattern<
      mlir::stablehlo::NotOp, mlir::tt::ttir::LogicalNotOp>>(typeConverter,
                                                             ctx);
  patterns.add<StableHLOToTTIROpLogicalOpConversionPattern<
      mlir::stablehlo::OrOp, mlir::tt::ttir::LogicalOrOp>>(typeConverter, ctx);
}

void addSliceOpConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRSliceOpConversionPattern>(typeConverter, ctx);
}

} // namespace

namespace mlir::tt {

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addReduceOpsConversionPatterns(ctx, patterns, typeConverter);
  addTransposeOpsConversionPatterns(ctx, patterns, typeConverter);
  addMatmulOpsConversionPatterns(ctx, patterns, typeConverter);
  addTensorCreationOpsConversionPatterns(ctx, patterns, typeConverter);
  addBroadcastOpConversionPattern(ctx, patterns, typeConverter);
  addConv2dOpConversionPattern(ctx, patterns, typeConverter);
  addReduceWindowOpConversionPattern(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
  addConcatOpsConversionPatterns(ctx, patterns, typeConverter);
  addReshapeOpConversionPattern(ctx, patterns, typeConverter);
  addLogicalOpConversionPattern(ctx, patterns, typeConverter);
  addSliceOpConversionPattern(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
