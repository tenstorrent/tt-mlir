// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>
#include <vector>

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

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
        adaptor.getOperands(), ValueRange(outputTensor));
    return success();
  }
};
} // namespace

namespace {
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
    if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MinOp>(srcOp, adaptor,
                                                            rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::ProdOp>(srcOp, adaptor,
                                                             rewriter);
    }
    if (mlir::isa<mlir::stablehlo::AndOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::ReduceAndOp>(
          srcOp, adaptor, rewriter);
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

    mlir::Operation &innerOp = srcOp.getBody().front().front();
    if (mlir::isa<mlir::stablehlo::AndOp>(innerOp)) {
      bool allOperandsAreBoolean = std::all_of(
          srcOp->operand_begin(), srcOp->operand_end(), [](auto operand) {
            return mlir::cast<RankedTensorType>(operand.getType())
                       .getElementTypeBitWidth() == 1;
          });
      if (!allOperandsAreBoolean) {
        return rewriter.notifyMatchFailure(
            srcOp, "stablehlo.reduce for stablehlo.and operator is only "
                   "supported for logical and.");
      }
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

    // Can't reuse the original dimensions attribute because it uses i64 type.
    mlir::ArrayAttr dimArg = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(srcOp.getDimensions()));

    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp, outputType, adaptor.getInputs().front(), outputTensor,
        false /* keep_dim */, dimArg);

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRDotGeneralOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern<mlir::stablehlo::DotGeneralOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp srcOp,
                  mlir::stablehlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::DotGeneralOp>(
        srcOp, outputTensor.getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getDotDimensionNumbers().getLhsBatchingDimensions(),
        adaptor.getDotDimensionNumbers().getLhsContractingDimensions(),
        adaptor.getDotDimensionNumbers().getRhsBatchingDimensions(),
        adaptor.getDotDimensionNumbers().getRhsContractingDimensions());
    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRTransposeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern<mlir::stablehlo::TransposeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::TransposeOp srcOp,
                  mlir::stablehlo::TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ::mlir::RankedTensorType outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    // stablehlo.transpose and ttir.permute have the same semantics.
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::PermuteOp>(
        srcOp, getTypeConverter()->convertType(srcOp.getResult().getType()),
        adaptor.getOperand(), outputTensor, adaptor.getPermutation());
    return success();
  }
};
} // namespace

namespace {
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
        adaptor.getOperand(), outputTensor, new_shape_attr);
    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRGetDimensionSizeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::GetDimensionSizeOp> {

  using OpConversionPattern<
      mlir::stablehlo::GetDimensionSizeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::GetDimensionSizeOp srcOp,
                  mlir::stablehlo::GetDimensionSizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerType intType = IntegerType::get(getContext(), 32);
    RankedTensorType outputType = RankedTensorType::get({1}, intType);
    mlir::OpBuilder builder(getContext());
    IntegerAttr dimension_attr = builder.getIntegerAttr(
        intType, static_cast<int32_t>(srcOp.getDimension()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::GetDimensionSizeOp>(
        srcOp, outputType, adaptor.getOperand(), dimension_attr);

    return success();
  }
};
} // namespace

namespace {
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
  // 4. Float tensor: TTNN does not support 64 bit float. So they are converted
  //    to 32 bit tensor.
  mlir::ElementsAttr getValueAttr(mlir::ElementsAttr valueAttr) const {
    Type elementType = valueAttr.getElementType();
    size_t bitWidth = elementType.getIntOrFloatBitWidth();
    bool isTensor = !valueAttr.getShapedType().getShape().empty();
    bool isIntTensor = isTensor && isa<IntegerType>(elementType) &&
                       bitWidth != 1 && bitWidth != 64;
    bool isFloatTensor = isTensor && isa<FloatType>(elementType) &&
                         bitWidth != 1 && bitWidth != 64;

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
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint8_t>(valueAttr, 8)
                   : rebuildValueAttr<int8_t>(valueAttr, 8);
      }
      case 16: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint16_t>(valueAttr, 16)
                   : rebuildValueAttr<int16_t>(valueAttr, 16);
      }
      case 32: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint32_t>(valueAttr, 32)
                   : rebuildValueAttr<int32_t>(valueAttr, 32);
      }
      case 64: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint64_t>(valueAttr, 32)
                   : rebuildValueAttr<int64_t>(valueAttr, 32);
      }
      default: {
        assert(false && "Unsupported integer type.");
      }
      }
    }
    if (isa<FloatType>(elementType)) {
      // Convert 64 bit floating point numbers to 32 bit floating point numbers.
      if (bitWidth == 64) {
        std::vector<mlir::APFloat> floatValues;
        for (mlir::APFloat value : valueAttr.getValues<mlir::APFloat>()) {
          float fl = static_cast<float>(value.convertToDouble());
          mlir::APFloat input = mlir::APFloat(fl);
          floatValues.emplace_back(input);
        }
        return mlir::DenseElementsAttr::get(valueType, floatValues);
      }
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
  // int32, int64, uint8, uint16, uint32, uint64) and tensors (of type boolean
  // and int64).
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
} // namespace

namespace {
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

    auto dimNums = adaptor.getDimensionNumbers();
    uint64_t numSpatialDims = dimNums.getInputSpatialDimensions().size();

    // These are the defaults intended by stablehlo when the attrs are not
    // populated
    DenseI64ArrayAttr windowStridesAttr =
        adaptor.getWindowStridesAttr()
            ? adaptor.getWindowStridesAttr()
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseI64ArrayAttr paddingAttr =
        adaptor.getPaddingAttr()
            ? rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  adaptor.getPaddingAttr().getValues<int64_t>()))
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims * 2, 0));
    DenseI64ArrayAttr inputDilationAttr =
        adaptor.getLhsDilationAttr()
            ? adaptor.getLhsDilationAttr()
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseI64ArrayAttr kernelDilationAttr =
        adaptor.getRhsDilationAttr()
            ? adaptor.getRhsDilationAttr()
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseBoolArrayAttr windowReversalAttr =
        adaptor.getWindowReversalAttr()
            ? adaptor.getWindowReversalAttr()
            : rewriter.getDenseBoolArrayAttr(
                  SmallVector<bool>(numSpatialDims, false));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConvolutionOp>(
        srcOp, outputType, adaptor.getLhs(), adaptor.getRhs(),
        mlir::Value(nullptr), outputTensor, windowStridesAttr, paddingAttr,
        inputDilationAttr, kernelDilationAttr, windowReversalAttr,
        mlir::tt::ttir::ConvolutionLayoutAttr::get(
            getContext(), dimNums.getInputBatchDimension(),
            dimNums.getInputFeatureDimension(),
            dimNums.getInputSpatialDimensions(),
            dimNums.getKernelOutputFeatureDimension(),
            dimNums.getKernelInputFeatureDimension(),
            dimNums.getKernelSpatialDimensions(),
            dimNums.getOutputBatchDimension(),
            dimNums.getOutputFeatureDimension(),
            dimNums.getOutputSpatialDimensions()),
        adaptor.getFeatureGroupCountAttr(), adaptor.getBatchGroupCountAttr());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRReduceWindowOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern<
      mlir::stablehlo::ReduceWindowOp>::OpConversionPattern;

public:
  bool isMaxPool(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    if (srcOp.getBody().getBlocks().size() != 1) {
      return false;
    }

    // Find constant input(s)
    Operation *initValue;
    for (uint64_t i = 0; i < srcOp.getInitValues().size(); i++) {
      initValue = srcOp.getInitValues()[i].getDefiningOp();
      while (initValue->getOpOperands().size() == 1) {
        initValue = initValue->getOpOperand(0).get().getDefiningOp();
      }
      if (!isa<stablehlo::ConstantOp>(initValue)) {
        return false;
      }

      stablehlo::ConstantOp initValueOp =
          mlir::cast<stablehlo::ConstantOp>(initValue);

      if (!checkInitValue(initValueOp, TypicalInitReductionValue::NEG_INF)) {
        return false;
      }
    }

    Block &block = *srcOp.getBody().getBlocks().begin();
    uint32_t opIdx = 0;
    for (Operation &op : block) {
      if (opIdx == 0 && !isa<mlir::stablehlo::MaxOp>(op)) {
        return false;
      }
      if (opIdx == 1 && !isa<mlir::stablehlo::ReturnOp>(op)) {
        return false;
      }
      if (opIdx >= 2) {
        return false; // More than two ops in the block
      }
      opIdx++;
    }

    return true;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp srcOp,
                  mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    SmallVector<Value> outputsVec;
    for (uint32_t i = 0; i < srcOp.getResults().size(); i++) {
      tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
      outputsVec.push_back(outputTensor);
    }
    ValueRange outputs = outputsVec;

    auto windowDimensions = adaptor.getWindowDimensionsAttr();
    auto windowStrides = adaptor.getWindowStridesAttr();
    auto baseDilations = adaptor.getBaseDilationsAttr();
    auto window_dilations = adaptor.getWindowDilationsAttr();
    auto padding_ = adaptor.getPaddingAttr();

    // Generate defaults if they dont exist (these defaults are what the
    // stablehlo dialect intends when they are not provided)
    windowStrides = windowStrides
                        ? windowStrides
                        : rewriter.getDenseI64ArrayAttr(
                              SmallVector<int64_t>(windowDimensions.size(), 1));
    baseDilations = baseDilations
                        ? baseDilations
                        : rewriter.getDenseI64ArrayAttr(
                              SmallVector<int64_t>(windowDimensions.size(), 1));
    window_dilations = window_dilations
                           ? window_dilations
                           : rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(
                                 windowDimensions.size(), 1));
    auto padding =
        padding_ ? rewriter.getDenseI64ArrayAttr(
                       SmallVector<int64_t>(padding_.getValues<int64_t>()))
                 : rewriter.getDenseI64ArrayAttr(
                       SmallVector<int64_t>(windowDimensions.size() * 2, 0));

    mlir::tt::ttir::PoolingMethod poolingMethod;
    int64_t dimension = -1;
    if (isMaxPool(srcOp)) {
      poolingMethod = mlir::tt::ttir::PoolingMethod::Max;
    } else if (isCumSum(srcOp, adaptor, dimension)) {
      rewriter.replaceOpWithNewOp<ttir::CumSumOp>(
          srcOp, outputType, adaptor.getInputs()[0],
          rewriter.getI64IntegerAttr(dimension), outputs[0]);
      return success();
    } else {
      return rewriter.notifyMatchFailure(srcOp, "Unsupported pooling method");
    }

    rewriter.replaceOpWithNewOp<ttir::PoolingOp>(
        srcOp, outputType, adaptor.getInputs(), outputs, poolingMethod,
        windowDimensions, windowStrides, baseDilations, window_dilations,
        padding);

    return success();
  }

private:
  // This function verify all the required conditions to convert stablehlo
  // reduce_window op to TTIR cumsum op and also determine the dimension
  // attribute along which the cumulative sum will be computed.
  // The reduce_window op must satisfy the following conditions.
  // 1. One input / one output, one block in body and two ops with in block.
  // 2. Ops in the block must be 'add' and 'return'.
  // 3. InitValue must be zero.
  // 4. There are no strides or dilations for window-related attributes.
  // 5. The size of padding attribute is equal to two times input tensor rank.
  // 6. Padding value must be zero in case of splat vector. Window dimension
  //    attribute must have all elements equal to one in this case.
  // 7. Padding attribute have one non-zero element in case of non-splat vector
  //    and this non-zero element must be equal to size of specified dimension
  //    minus one.
  // The dimension attribute is determined in following two ways.
  // 1. (If padding is splat vector): First dimension in the input tensor shape,
  //    whose size is 1, is the required dimension.
  // 2. (If padding is non-splat vector): Window dimension attribute must have
  //    all elements equal to 1 except one; whose location is the required
  //    dimension and value must be qual to size of the required dimension.
  bool isCumSum(mlir::stablehlo::ReduceWindowOp &srcOp,
                mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                int64_t &dimension) const {

    // Check basic structure of the ReduceWindowOp
    if (!hasValidOpStructure(srcOp)) {
      return false;
    }

    // Verify operations in the block
    if (!hasValidOperationsInBlock(srcOp)) {
      return false;
    }

    // Check init values
    if (!hasValidInitValues(srcOp)) {
      return false;
    }

    // Verify window-related attributes (strides, dilations)
    if (!hasValidWindowAttributes(adaptor)) {
      return false;
    }

    // Check input tensor type and padding
    if (!hasValidInputAndPadding(srcOp, adaptor, dimension)) {
      return false;
    }

    return true;
  }

  // validate basic structure of the ReduceWindowOp.
  bool hasValidOpStructure(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    if (srcOp.getBody().getBlocks().size() != 1 ||
        srcOp.getBody().getBlocks().begin()->getOperations().size() != 2) {
      return false;
    }
    if (srcOp.getInputs().size() != 1 || srcOp->getResults().size() != 1) {
      return false;
    }
    return true;
  }

  // Check init values (must be constant and zero).
  bool hasValidInitValues(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    for (auto initValue : srcOp.getInitValues()) {
      auto *defOp = initValue.getDefiningOp();
      while (defOp->getOpOperands().size() == 1) {
        defOp = defOp->getOpOperand(0).get().getDefiningOp();
      }
      if (!isa<stablehlo::ConstantOp>(defOp)) {
        return false;
      }
      stablehlo::ConstantOp initValueOp =
          mlir::cast<stablehlo::ConstantOp>(defOp);
      if (!checkInitValue(initValueOp, TypicalInitReductionValue::ZERO)) {
        return false;
      }
    }
    return true;
  }

  // Verify operations inside the block (AddOp followed by ReturnOp).
  bool hasValidOperationsInBlock(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    Block &block = *srcOp.getBody().getBlocks().begin();
    auto &operations = block.getOperations();
    if (!isa<mlir::stablehlo::AddOp>(operations.front())) {
      return false;
    }
    if (!isa<mlir::stablehlo::ReturnOp>(operations.back())) {
      return false;
    }
    return true;
  }

  // Verify that window attributes (strides, dilations) are all set to 1.
  bool hasValidWindowAttributes(
      mlir::stablehlo::ReduceWindowOp::Adaptor adaptor) const {
    auto verifyAttributes = [](mlir::DenseI64ArrayAttr arrAttr) -> bool {
      if (!arrAttr) {
        return true;
      }
      return std::all_of(arrAttr.asArrayRef().begin(),
                         arrAttr.asArrayRef().end(),
                         [](int value) { return value == 1; });
    };
    return verifyAttributes(adaptor.getWindowStridesAttr()) &&
           verifyAttributes(adaptor.getBaseDilationsAttr()) &&
           verifyAttributes(adaptor.getWindowDilationsAttr());
  }

  // Check input tensor type and validate padding.
  bool hasValidInputAndPadding(mlir::stablehlo::ReduceWindowOp &srcOp,
                               mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                               int64_t &dimension) const {
    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getInputs()[0].getType()));
    int64_t inputRank = inputType.getRank();
    llvm::ArrayRef<int64_t> windowDimensions =
        adaptor.getWindowDimensionsAttr().asArrayRef();
    mlir::DenseIntElementsAttr padding = adaptor.getPaddingAttr();

    // Validate padding size
    if (padding.size() != (inputRank * 2)) {
      return false;
    }

    // Check for splat padding (all zeroes expected).
    if (padding.isSplat()) {
      if (padding.getSplatValue<int64_t>() != 0) {
        return false;
      }
      if (!std::all_of(windowDimensions.begin(), windowDimensions.end(),
                       [](int value) { return value == 1; })) {
        return false;
      }
      // Determine the dimension using input tensor shape.
      return findDimensionWithShape(inputType, dimension);
    }

    // Check non-splat padding and ensure the window dimensions and padding are
    // consistent and determine the dimension attribute.
    return validateNonSplatPadding(windowDimensions, padding, inputType,
                                   dimension);
  }

  // Find the dimension using input tensor shape.
  bool findDimensionWithShape(RankedTensorType inputType,
                              int64_t &dimension) const {
    dimension = -1;
    for (int64_t size : inputType.getShape()) {
      ++dimension;
      if (size == 1) {
        return true;
      }
    }
    return false;
  }

  // Determine and validate dimension attribute for non-splat padding attribute.
  bool validateNonSplatPadding(llvm::ArrayRef<int64_t> windowDimensions,
                               mlir::DenseIntElementsAttr padding,
                               RankedTensorType inputType,
                               int64_t &dimension) const {
    int64_t dimArgValue = -1;
    int64_t idx = -1;
    auto padding_values = padding.getValues<int64_t>();

    // Determine dimension attribute.
    for (int64_t windowDim : windowDimensions) {
      ++idx;
      if (windowDim == 1) {
        continue;
      }
      if (dimArgValue != -1) {
        return false; // Ensure only one non-1 element.
      }
      dimArgValue = windowDim;
      dimension = idx;
    }

    // Validate dimension attribute.
    if (dimArgValue != inputType.getShape()[dimension] || dimArgValue <= 1) {
      return false;
    }

    for (int64_t i = 0; i < padding.size(); ++i) {
      if (i == (dimension * 2)) {
        if (padding_values[i] != (dimArgValue - 1)) {
          return false;
        }
      } else if (padding_values[i] != 0) {
        return false;
      }
    }

    return true;
  }

  enum TypicalInitReductionValue {
    NEG_INF, // used for max pooling
    ZERO,    // used for sum pooling
  };

  // Using the value enum rather than actual values because of different data
  // types the init value could be
  bool checkInitValue(stablehlo::ConstantOp initValueOp,
                      TypicalInitReductionValue desired) const {
    if (initValueOp.getValueAttr().size() != 1) {
      return false;
    }

    float desiredF32;
    double desiredF64;
    uint16_t desiredBF16;
    int32_t desiredI32;
    int64_t desiredI64;
    if (desired == TypicalInitReductionValue::NEG_INF) {
      desiredF32 = -std::numeric_limits<float>::infinity();
      desiredF64 = -std::numeric_limits<double>::infinity();
      desiredBF16 = 0xff80; // This is -inf in bfloat16 raw bits
      desiredI32 = std::numeric_limits<int32_t>::min();
      desiredI64 = std::numeric_limits<int64_t>::min();
    } else if (desired == TypicalInitReductionValue::ZERO) {
      desiredF32 = 0.0;
      desiredF64 = 0.0;
      desiredBF16 = 0x0000; // This is 0 in bfloat16 raw bits
      desiredI32 = 0;
      desiredI64 = 0;
    } else {
      return false;
    }

    // Constant operand must be -inf if this is to be a max pool
    // since bfloat16 is not a type we actually have I must compare the raw
    // bits
    if (initValueOp.getResult().getType().getElementType().isBF16()) {
      // Collect the values into a vector
      std::vector<mlir::Attribute> values;
      for (int64_t i = 0; i < initValueOp.getValueAttr().size(); ++i) {
        values.push_back(
            initValueOp.getValueAttr().getValues<mlir::Attribute>()[i]);
      }

      auto denseValues = ::mlir::DenseElementsAttr::get(
          initValueOp.getValueAttr().getShapedType(), values);
      uint16_t bfloat_bits =
          static_cast<uint16_t>(*denseValues.getRawData().data());
      if (bfloat_bits != desiredBF16) { // This is -inf in bfloat16
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isF32()) {
      if (*initValueOp.getValue().value_begin<float>() != desiredF32) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isF64()) {
      if (*initValueOp.getValue().value_begin<double>() != desiredF64) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isInteger(
                   32)) {
      if (*initValueOp.getValue().value_begin<int32_t>() != desiredI32) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isInteger(
                   64)) {
      if (*initValueOp.getValue().value_begin<int64_t>() != desiredI64) {
        return false;
      }
    } else {
      return false;
    }

    return true;
  }
};
} // namespace

namespace {
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

    auto inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getOperand().getType()));

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    if (inputType.getRank() == outputType.getRank()) {
      // No unsqueeze is needed in this case and this broadcast can be
      // represented by broadcast op.
      tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

      ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
      ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      rewriter.replaceOpWithNewOp<mlir::tt::ttir::BroadcastOp>(
          srcOp, outputTensor.getType(), adaptor.getOperand(), outputTensor,
          broadcastShape);
    } else {
      // This stablehlo operation cannot be represented by a single TTIR
      // operation. It has to be split into ttir.reshape followed by a
      // ttir.broadcast op.
      SmallVector<int64_t> unsqueezeShape(outputType.getRank(), 1);
      ::llvm::ArrayRef<int64_t> broadcastInDim =
          adaptor.getBroadcastDimensions();

      // Since we convert scalars to 1D tensors as a special case,
      // so check input dimension is not empty.
      if (!broadcastInDim.empty()) {
        for (int64_t i = 0; i < inputType.getRank(); i++) {
          unsqueezeShape[broadcastInDim[i]] = inputType.getDimSize(i);
        }
      }

      RankedTensorType unsqueezeOutputType =
          RankedTensorType::get(unsqueezeShape, outputType.getElementType());

      tensor::EmptyOp reshapeOutputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), unsqueezeOutputType.getShape(),
          unsqueezeOutputType.getElementType());

      SmallVector<int32_t> reshapeDim(unsqueezeShape.begin(),
                                      unsqueezeShape.end());
      auto reshapeDimAttr = rewriter.getI32ArrayAttr(reshapeDim);

      mlir::tt::ttir::ReshapeOp reshape =
          rewriter.create<mlir::tt::ttir::ReshapeOp>(
              srcOp.getLoc(), unsqueezeOutputType, adaptor.getOperand(),
              reshapeOutputTensor, reshapeDimAttr);

      tensor::EmptyOp broadcastOutputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

      ::llvm::ArrayRef<int64_t> inputShape = unsqueezeShape;
      ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      rewriter.replaceOpWithNewOp<mlir::tt::ttir::BroadcastOp>(
          srcOp, broadcastOutputTensor.getType(), reshape.getResult(),
          broadcastOutputTensor, broadcastShape);
    }

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::BroadcastInDimOp &srcOp,
                     mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    llvm::SmallVector<int64_t, 4> broadcastedShape;
    auto srcType =
        getTypeConverter()->convertType(adaptor.getOperand().getType());
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
} // namespace

namespace {
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
        adaptor.getOperands(), ValueRange(outputTensor));

    return success();
  }
};
} // namespace

namespace {
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
            static_cast<int32_t>(adaptor.getDimension()))); // dimension
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

    auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
        adaptor.getOperands()[0].getType());
    if (static_cast<int64_t>(adaptor.getDimension()) >=
        rankedTensorType.getRank()) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid concatenation dimension.");
    }

    return success();
  }
};
} // namespace

// Class implementing conversion from StableHLO to TTIR logical and bitwise ops.
// StableHLO has AND, OR, XOR and NOT ops defined in such a way that they do two
// different things based on type of inputs. In case of booleans, they perform
// logical version of the op, and in case of integers they perform bitwise
// version of the op. We made a decision to make those two cases completely
// distinct ops in TTIR. Thus, a StableHLO `SrcOp` is rewritten to one of
// `DestOp`s based on operand types.
namespace {
template <typename SrcOp, typename LogicalDestOp, typename BitwiseDestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIRLogicalAndBitwiseOpConversionPattern
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

    if (getStableHLOOpType(srcOp) == StableHLOOpType::kLogical) {
      replaceOpWithNewOp<LogicalDestOp>(srcOp, adaptor, outputTensor, rewriter);
    } else {
      replaceOpWithNewOp<BitwiseDestOp>(srcOp, adaptor, outputTensor, rewriter);
    }

    return success();
  }

private:
  enum StableHLOOpType { kLogical = 0, kBitwise = 1 };

  // Determines stablehlo op type based on its operand types (i.e. their
  // bit width). This assumes boolean operands are modeled as 1bit wide ints.
  static StableHLOOpType getStableHLOOpType(const SrcOp &srcOp) {
    // Checks if all operands are boolean (have bit width equal to 1).
    bool allOperandsAreBoolean = std::all_of(
        srcOp->operand_begin(), srcOp->operand_end(), [](auto operand) {
          return mlir::cast<RankedTensorType>(operand.getType())
                     .getElementTypeBitWidth() == 1;
        });

    return allOperandsAreBoolean ? StableHLOOpType::kLogical
                                 : StableHLOOpType::kBitwise;
  }

  // Helper function to replace the operation with the new op to avoid code
  // duplication.
  template <typename DestOp>
  void replaceOpWithNewOp(SrcOp srcOp, Adaptor adaptor,
                          tensor::EmptyOp outputTensor,
                          ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp,
        TypeRange(
            this->getTypeConverter()->convertType(outputTensor.getType())),
        adaptor.getOperands(), ValueRange(outputTensor));
  }
};
} // namespace

template <typename SrcOpTy>
LogicalResult getReduceType(SrcOpTy &srcOp, ReduceType &reduceType) {
  if constexpr (!std::is_same<SrcOpTy, mlir::stablehlo::AllReduceOp>::value) {
    return failure();
  }
  // Check operations in the first block and determine reduce type for now
  // TODO(wooseoklee): This pattern matching mechanism may need to be updated as
  // we see complicated patterns of reduce block in the future.
  auto &block = srcOp.getRegion().front();
  for (Operation &op : block) {
    if (isa<mlir::stablehlo::AddOp>(op)) {
      reduceType = ReduceType::Sum;
      return success();
    }
    if (isa<mlir::stablehlo::MaxOp>(op)) {
      reduceType = ReduceType::Max;
      return success();
    }
    if (isa<mlir::stablehlo::MinOp>(op)) {
      reduceType = ReduceType::Min;
      return success();
    }
  }
  // Other reduce types are currently not supported
  return failure();
}

// StalbeHLO spec.md defines following channel type for ccl ops
enum StableHLOChannelType {
  // CHANNEL_TYPE_INVALID = 0 : Invalid primitive type to serve as
  // default.
  kChannelTypeInvalid = 0,
  // DEVICE_TO_DEVICE = 1 : A channel for sending data between
  // devices.
  kChannelTypeDeviceToDevice = 1,
  // DEVICE_TO_HOST = 2 : A channel for sending data from the
  // device to the host. Can only be used with a Send operation.
  kChannelTypeDeviceToHost = 2,
  // HOST_TO_DEVICE = 3 : A channel for sending data from the host to
  // the device. Can only be used with a Recv operation.
  kChannelTypeHostToDevice = 3,
};

namespace {
class StableHLOToTTIRAllReduceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::AllReduceOp> {

  using OpConversionPattern<mlir::stablehlo::AllReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::AllReduceOp srcOp,
                  mlir::stablehlo::AllReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    // Create an empty output tensor with the computed shape
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    SmallVector<Type> ttirTypes;
    if (failed(this->getTypeConverter()->convertTypes(srcOp->getResultTypes(),
                                                      ttirTypes))) {
      return failure();
    }

    auto ttirOperands = srcOp.getOperandsMutable();
    ttirOperands.append(ValueRange(outputTensor));

    SmallVector<NamedAttribute> srcAttrs = to_vector(srcOp->getAttrs());
    SmallVector<NamedAttribute> ttirAttrs;
    for (auto srcAttr : srcAttrs) {
      StringAttr srcName = srcAttr.getName();
      if (srcName == "channel_handle") {
        auto srcChannelHandleAttr =
            dyn_cast<mlir::stablehlo::ChannelHandleAttr>(srcAttr.getValue());
        if (!srcChannelHandleAttr) {
          return failure();
        }

        // channelType is supposed to be DEVICE_TO_DEVICE for CCL ops.
        // Currently, we ensure if it is DEVICE_TO_DEVICE commmuincaiton.
        // Consider preserving this information in the future if the attribute
        // is non-DEVICE_TO_DEVICE values.
        auto channelType = static_cast<int32_t>(srcChannelHandleAttr.getType());
        if (channelType != kChannelTypeDeviceToDevice) {
          return failure();
        }

        IntegerAttr channelHandleAttr = rewriter.getSI32IntegerAttr(
            static_cast<int32_t>(srcChannelHandleAttr.getHandle()));
        if (!channelHandleAttr) {
          return failure();
        }
        ttirAttrs.push_back({srcName, channelHandleAttr});
      } else {
        ttirAttrs.push_back(srcAttr);
      }
    }

    // Algorithm: search for first non-one working dimension from back
    auto replicaGroupsShape = adaptor.getReplicaGroups().getType().getShape();
    size_t dim = replicaGroupsShape.size() - 1;
    for (auto s = replicaGroupsShape.rbegin(); s != replicaGroupsShape.rend();
         ++s, --dim) {
      if (*s != 1) {
        break;
      }
    }
    if (dim < 0) {
      // all one shape, then select the fastest dim
      dim = replicaGroupsShape.size();
    }
    StringAttr dimName = StringAttr::get(this->getContext(), "dim");
    IntegerAttr dimAttr =
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(dim));
    ttirAttrs.push_back({dimName, dimAttr});

    // Parse computation in region and add it to ttirAttrs
    ReduceType reduceType;
    if (failed(getReduceType(srcOp, reduceType))) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllReduceOp cannot specify reduce type.");
    }
    StringAttr reduceTypeAttrName =
        StringAttr::get(this->getContext(), "reduce_type");
    Attribute reduceTypeAttr = rewriter.getAttr<ReduceTypeAttr>(reduceType);
    ttirAttrs.push_back({reduceTypeAttrName, reduceTypeAttr});

    auto ttirAllReduceOp = rewriter.create<mlir::tt::ttir::AllReduceOp>(
        srcOp.getLoc(), ttirTypes, ValueRange(ttirOperands.getAsOperandRange()),
        ttirAttrs);

    rewriter.replaceOp(srcOp, ttirAllReduceOp);

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::AllReduceOp &srcOp,
                     mlir::stablehlo::AllReduceOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (srcOp.getOperands().empty() || srcOp.getOperands().size() > 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllReduceOp must have one input/output for now.");
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRCustomCallOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    const std::string kShardingTarget = "Sharding";
    const std::string kSPMDFullToShardShapeTarget = "SPMDFullToShardShape";
    const std::string kSPMDShardToFullShapeTarget = "SPMDShardToFullShape";

    auto callTargetName = adaptor.getCallTargetNameAttr();

    // Currently stablehlo.custom_call with following functions from
    // jax/openxla are supported
    if (callTargetName != kShardingTarget &&
        callTargetName != kSPMDFullToShardShapeTarget &&
        callTargetName != kSPMDShardToFullShapeTarget) {
      return failure();
    }

    auto shardingAttr = dyn_cast_or_null<StringAttr>(
        adaptor.getAttributes().get("mhlo.sharding"));
    if (!shardingAttr) {
      return failure();
    }
    StringRef shardingStr = shardingAttr.getValue();
    if (!shardingStr.consume_front("{") || !shardingStr.consume_back("}")) {
      return failure();
    }
    SmallVector<StringRef> shardingStrAttrs;
    shardingStr.split(shardingStrAttrs, " ");
    struct ShardAttrValue shardAttrValue;
    if (failed(parseShardingAttr(rewriter, shardingStrAttrs, shardAttrValue))) {
      return failure();
    }

    if (callTargetName == kSPMDFullToShardShapeTarget) {
      Operation *shardingOp = srcOp->getOperand(0).getDefiningOp();
      if (!shardingOp) {
        return rewriter.notifyMatchFailure(
            srcOp, "requires operand to be defined by an op");
      }

      // TODO(wooseoklee): a bit rough approach here to match output dim
      shardingOp->getResult(0).setType(srcOp->getResult(0).getType());
      srcOp.getResult(0).replaceAllUsesWith(shardingOp->getResult(0));
      rewriter.eraseOp(srcOp);
    } else if (callTargetName == kSPMDShardToFullShapeTarget) {
      Operation *shardingOp = srcOp->getOperand(0).getDefiningOp();
      if (!shardingOp) {
        return rewriter.notifyMatchFailure(
            srcOp, "requires operand to be defined by an op");
      }

      // Create the output tensor type based on inputs
      auto outputType = mlir::cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp->getResult(0).getType()));

      // Create an empty output tensor with the computed shape
      tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

      SmallVector<Type> outputTypes;
      if (failed(this->getTypeConverter()->convertTypes(srcOp->getResultTypes(),
                                                        outputTypes))) {
        return failure();
      }

      shardAttrValue.shardDirection = mlir::tt::MeshShardDirection::ShardToFull;
      if (failed(createMeshShardOp(srcOp, adaptor, outputTensor, outputTypes,
                                   shardAttrValue, rewriter))) {
        return failure();
      }
    } else if (callTargetName == kShardingTarget) {
      if (shardAttrValue.shardType == mlir::tt::MeshShardType::Manual) {
        // "manual" sharding indicates match between input/output tensor shape
        // and no sharding is required.
        srcOp.getResult(0).replaceAllUsesWith(srcOp->getOperand(0));
        rewriter.eraseOp(srcOp);
      } else {
        auto *user = *srcOp.getResult(0).user_begin();
        auto userOp = dyn_cast_or_null<mlir::stablehlo::CustomCallOp>(user);
        if (!userOp) {
          return failure();
        }

        // Create the output tensor type based on inputs
        auto outputType = mlir::cast<RankedTensorType>(
            getTypeConverter()->convertType(userOp->getResult(0).getType()));

        // Create an empty output tensor with the computed shape
        tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
            srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

        SmallVector<Type> outputTypes;
        if (failed(this->getTypeConverter()->convertTypes(
                userOp->getResultTypes(), outputTypes))) {
          return failure();
        }

        shardAttrValue.shardDirection =
            mlir::tt::MeshShardDirection::FullToShard;
        if (failed(createMeshShardOp(srcOp, adaptor, outputTensor, outputTypes,
                                     shardAttrValue, rewriter))) {
          return failure();
        }
      }
    }
    return success();
  }

private:
  struct ShardAttrValue {
    mlir::tt::MeshShardDirection shardDirection;
    mlir::tt::MeshShardType shardType;
    bool lastTileDimReplicate;
    std::vector<int64_t> shardShape;
  };

  // OpenXLA has its own lexer, but we will use simple string-based parser here
  // This parsing is mainly based on "Sharding Attribute" section in
  // https://github.com/sdasgup3/stablehlo/blob/80082431d1af0933e6202ecc8a6f8801e039235b/docs/spec.md
  LogicalResult parseShardingAttr(ConversionPatternRewriter &rewriter,
                                  SmallVector<StringRef> shardingStrAttrs,
                                  struct ShardAttrValue &shardAttrValue) const {
    MeshShardType shardType = mlir::tt::MeshShardType::Manual;
    bool lastTileDimReplicate = false;
    for (auto str : shardingStrAttrs) {
      if (str.contains("replicated")) {
        assert(shardType == mlir::tt::MeshShardType::Manual &&
               "Fail to parse sharding info.");
        // replicated: all devices have whole data
        shardType = mlir::tt::MeshShardType::Replicate;
        shardAttrValue.shardShape.push_back(1);
      } else if (str.contains("maximal")) {
        assert(shardType == mlir::tt::MeshShardType::Manual &&
               "Fail to parse sharding info.");
        // maximal: one device has whole data
        shardType = mlir::tt::MeshShardType::Maximal;
        shardAttrValue.shardShape.push_back(1);
      } else if (str.contains("device=")) {
        // maximal should followed by "device" to put data on
        assert(shardType == mlir::tt::MeshShardType::Maximal &&
               "Fail to parse sharding info.");
        int64_t d;
        if (!str.consume_front("device=")) {
          return failure();
        }
        if (str.getAsInteger<int64_t>(10, d)) {
          return failure();
        }
        shardAttrValue.shardShape.push_back(d);
      } else if (str.contains("manual")) {
        assert(shardType == mlir::tt::MeshShardType::Manual &&
               "Fail to parse sharding info.");
        // manual: already sharded, so no action is needed
        assert(!lastTileDimReplicate &&
               "last time dim duplicate option shouldn't be set here.");
        shardAttrValue.shardShape.push_back(1);
      } else if (str.contains("devices=")) {
        // other: "devices" detail sharding plan
        assert(shardType == mlir::tt::MeshShardType::Manual &&
               "Fail to parse sharding info.");
        shardType = mlir::tt::MeshShardType::Devices;
        if (!str.consume_front("devices=")) {
          return failure();
        }
        auto [devicesStr, restStr] = str.split("<=");
        // parse devices ex. [4,2,1]
        if (!devicesStr.consume_front("[") || !devicesStr.consume_back("]")) {
          return failure();
        }
        SmallVector<StringRef> dimsStr;
        devicesStr.split(dimsStr, ",");
        for (auto dim : dimsStr) {
          int64_t d;
          if (dim.getAsInteger<int64_t>(10, d)) {
            return failure();
          }
          shardAttrValue.shardShape.push_back(d);
        }
      } else if (str.contains("last_tile_dim_replicate")) {
        assert(shardType == mlir::tt::MeshShardType::Devices &&
               "Fail to parse sharding info.");
        // other: replicate last tile dim
        lastTileDimReplicate = true;
      }
    }
    shardAttrValue.shardType = shardType;
    shardAttrValue.lastTileDimReplicate = lastTileDimReplicate;
    return success();
  }

  LogicalResult
  createMeshShardOp(mlir::stablehlo::CustomCallOp &srcOp,
                    mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                    tensor::EmptyOp &outputTensor,
                    SmallVector<Type> &outputTypes,
                    ShardAttrValue &shardAttrValue,
                    ConversionPatternRewriter &rewriter) const {

    auto meshShardOperands = srcOp.getInputsMutable();
    meshShardOperands.append(ValueRange(outputTensor));
    SmallVector<NamedAttribute> meshShardAttrs;

    StringAttr shardTypeAttrName = rewriter.getStringAttr("shard_type");
    Attribute shardTypeAttr =
        rewriter.getAttr<MeshShardTypeAttr>(shardAttrValue.shardType);
    meshShardAttrs.push_back({shardTypeAttrName, shardTypeAttr});

    StringAttr shardDirectionAttrName =
        rewriter.getStringAttr("shard_direction");
    Attribute shardDirectionAttr =
        rewriter.getAttr<MeshShardDirectionAttr>(shardAttrValue.shardDirection);
    meshShardAttrs.push_back({shardDirectionAttrName, shardDirectionAttr});

    StringAttr shardShapeAttrName = rewriter.getStringAttr("shard_shape");
    if (shardAttrValue.lastTileDimReplicate) {
      shardAttrValue.shardShape.pop_back();
    }
    GridAttr shardShape =
        GridAttr::get(this->getContext(), shardAttrValue.shardShape);
    meshShardAttrs.push_back({shardShapeAttrName, shardShape});

    auto meshShardOp = rewriter.create<mlir::tt::ttir::MeshShardOp>(
        srcOp.getLoc(), outputTypes,
        ValueRange(meshShardOperands.getAsOperandRange()), meshShardAttrs);
    rewriter.replaceOp(srcOp, meshShardOp);

    return success();
  }

  LogicalResult
  checkBasicLegality(mlir::stablehlo::CustomCallOp &srcOp,
                     mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    // Expect single input/output, otherwise do not convert
    if (adaptor.getInputs().size() != 1 && srcOp->getResults().size() != 1) {
      return failure();
    }

    return success();
  }
};
} // namespace

namespace {
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
        rewriter.getI32ArrayAttr(end_indices), rewriter.getI32ArrayAttr(step));
    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIROpClampOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ClampOp> {

  using OpConversionPattern<mlir::stablehlo::ClampOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ClampOp srcOp,
                  mlir::stablehlo::ClampOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    Value min = adaptor.getMin();
    Value max = adaptor.getMax();
    Operation *minDefiningOp = min.getDefiningOp();
    Operation *maxDefiningOp = max.getDefiningOp();
    if (minDefiningOp && maxDefiningOp &&
        isa<mlir::tt::ttir::ConstantOp>(minDefiningOp) &&
        isa<mlir::tt::ttir::ConstantOp>(maxDefiningOp)) {
      mlir::ElementsAttr minValAttr =
          mlir::cast<mlir::tt::ttir::ConstantOp>(minDefiningOp).getValueAttr();
      mlir::ElementsAttr maxValAttr =
          mlir::cast<mlir::tt::ttir::ConstantOp>(maxDefiningOp).getValueAttr();
      if (minValAttr.isSplat() && maxValAttr.isSplat()) {
        float minValue =
            minValAttr.getElementType().isInteger()
                ? static_cast<float>(minValAttr.getSplatValue<int>())
                : minValAttr.getSplatValue<float>();
        float maxValue =
            maxValAttr.getElementType().isInteger()
                ? static_cast<float>(maxValAttr.getSplatValue<int>())
                : maxValAttr.getSplatValue<float>();
        rewriter.replaceOpWithNewOp<mlir::tt::ttir::ClampOp>(
            srcOp,
            this->getTypeConverter()->convertType(outputTensor.getType()),
            Value(adaptor.getOperand()), Value(outputTensor),
            rewriter.getF32FloatAttr(minValue),
            rewriter.getF32FloatAttr(maxValue));

        return success();
      }
    }

    ttir::MaximumOp maximumOp = rewriter.create<mlir::tt::ttir::MaximumOp>(
        srcOp->getLoc(), min, adaptor.getOperand(), outputTensor);

    tensor::EmptyOp finalOutputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::MinimumOp>(
        srcOp, maximumOp->getResult(0), max, finalOutputTensor);
    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRGatherOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::GatherOp> {
  using OpConversionPattern<mlir::stablehlo::GatherOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp srcOp,
                  mlir::stablehlo::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    auto dimensionNumbers = srcOp.getDimensionNumbers();

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::GatherOp>(
        srcOp, outputType, adaptor.getOperands()[0],
        adaptor.getOperands()[1], // Start indices
        Value(outputTensor), dimensionNumbers.getOffsetDims(),
        dimensionNumbers.getCollapsedSliceDims(),
        dimensionNumbers.getOperandBatchingDims(),
        dimensionNumbers.getStartIndicesBatchingDims(),
        dimensionNumbers.getStartIndexMap(),
        dimensionNumbers.getIndexVectorDim(), srcOp.getSliceSizesAttr(), false);
    return success();
  }
};
} // namespace

namespace {
template <typename SrcIotaOp, typename Adaptor = typename SrcIotaOp::Adaptor>
class StableHLOToTTIROpIotaOpConversionPattern
    : public OpConversionPattern<SrcIotaOp> {

  using OpConversionPattern<SrcIotaOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcIotaOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    rewriter.replaceOpWithNewOp<ttir::ArangeOp>(
        srcOp, outputType, 0, outputType.getDimSize(adaptor.getIotaDimension()),
        1, adaptor.getIotaDimension());

    // Dynamic Iota has an output_shape attribute but the output shape is
    // already known by the result type This is to remove the operand that
    // will become dead code
    for (auto operand : adaptor.getOperands()) {
      if (operand.getDefiningOp()) {
        rewriter.eraseOp(operand.getDefiningOp());
      }
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRScatterOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ScatterOp> {

  using OpConversionPattern<mlir::stablehlo::ScatterOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp srcOp,
                  mlir::stablehlo::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResults()[0].getType()));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    Value operand = srcOp.getInputs()[0];
    Value scatterIndices = srcOp.getScatterIndices();
    Value update = srcOp.getUpdates()[0];
    auto updateWindowsDims =
        adaptor.getScatterDimensionNumbers().getUpdateWindowDims();
    auto insertedWindowDims =
        adaptor.getScatterDimensionNumbers().getInsertedWindowDims();
    auto inputBatchingDims =
        adaptor.getScatterDimensionNumbers().getInputBatchingDims();
    auto scatterIndicesBatchingDims =
        adaptor.getScatterDimensionNumbers().getScatterIndicesBatchingDims();
    auto scatterDimsToOperandDims =
        adaptor.getScatterDimensionNumbers().getScatterDimsToOperandDims();
    auto indexVectorDim =
        adaptor.getScatterDimensionNumbers().getIndexVectorDim();
    auto indicesAreSorted = adaptor.getIndicesAreSorted();
    auto uniqueIndices = adaptor.getUniqueIndices();

    auto newScatterOp = rewriter.create<mlir::tt::ttir::ScatterOp>(
        srcOp.getLoc(), outputType, operand, scatterIndices, update,
        llvm::ArrayRef<int32_t>(
            convertArrayRefToInt32vector(updateWindowsDims)),
        llvm::ArrayRef<int32_t>(
            convertArrayRefToInt32vector(insertedWindowDims)),
        llvm::ArrayRef<int32_t>(
            convertArrayRefToInt32vector(inputBatchingDims)),
        llvm::ArrayRef<int32_t>(
            convertArrayRefToInt32vector(scatterIndicesBatchingDims)),
        llvm::ArrayRef<int32_t>(
            convertArrayRefToInt32vector(scatterDimsToOperandDims)),
        indexVectorDim, indicesAreSorted, uniqueIndices, outputTensor);

    // Replaces with different types do not work and will fail silently, so we
    // manually set the second operand, since the type changes there from i32 to
    // i64.
    newScatterOp.setOperand(
        1, adaptor.getScatterIndices().getDefiningOp()->getResult(0));

    newScatterOp->getRegion(0).takeBody(adaptor.getUpdateComputation());
    changeRegionTypes(newScatterOp->getRegion(0), *getTypeConverter(),
                      rewriter);

    rewriter.replaceOp(srcOp, newScatterOp);

    return success();
  }

private:
  std::vector<int32_t>
  convertArrayRefToInt32vector(const llvm::ArrayRef<int64_t> &source) const {
    std::vector<int32_t> converted;
    converted.reserve(source.size());

    for (int64_t value : source) {
      converted.push_back(static_cast<int32_t>(value));
    }

    return converted;
  }

  void changeRegionTypes(mlir::Region &region,
                         const mlir::TypeConverter &typeConverter,
                         mlir::PatternRewriter &rewriter) const {
    Block &block = *region.getBlocks().begin();
    llvm::SmallVector<mlir::BlockArgument, 4> oldArguments(
        block.getArguments().begin(), block.getArguments().end());
    llvm::SmallVector<mlir::Value, 4> newArguments;

    // Add new arguments with updated types to the block.
    for (auto arg : oldArguments) {
      if (auto newType = typeConverter.convertType(arg.getType())) {
        mlir::BlockArgument newArg = block.addArgument(newType, arg.getLoc());
        newArguments.push_back(newArg);
      } else {
        newArguments.push_back(arg); // Type didn't change
      }
    }

    for (auto it : llvm::zip(oldArguments, newArguments)) {
      mlir::BlockArgument oldArg = std::get<0>(it);
      mlir::Value newArg = std::get<1>(it);
      if (oldArg != newArg) {
        oldArg.replaceAllUsesWith(newArg);
      }
    }

    for (auto arg : oldArguments) {
      if (!llvm::is_contained(newArguments, arg)) {
        block.eraseArgument(arg.getArgNumber());
      }
    }
  }
};
} // namespace

namespace {
class StableHLOToTTIRReturnOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReturnOp> {

  using OpConversionPattern<mlir::stablehlo::ReturnOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReturnOp srcOp,
                  mlir::stablehlo::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::YieldOp>(srcOp,
                                                         srcOp.getResults());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIROpReverseOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReverseOp> {

  using OpConversionPattern<mlir::stablehlo::ReverseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReverseOp srcOp,
                  mlir::stablehlo::ReverseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ReverseOp>(
        srcOp,
        outputType,                 // result type
        adaptor.getOperand(),       // input
        outputTensor,               // output
        adaptor.getDimensionsAttr() // dimensions
    );
    return success();
  }
};
} // namespace

static void
addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
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
      mlir::stablehlo::FloorOp, mlir::tt::ttir::FloorOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::IsFiniteOp, mlir::tt::ttir::IsFiniteOp>>(typeConverter,
                                                                ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::NegOp, mlir::tt::ttir::NegOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SineOp, mlir::tt::ttir::SinOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SqrtOp, mlir::tt::ttir::SqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::Log1pOp, mlir::tt::ttir::Log1pOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::Expm1Op, mlir::tt::ttir::Expm1Op>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SignOp, mlir::tt::ttir::SignOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::LogisticOp, mlir::tt::ttir::SigmoidOp>>(typeConverter,
                                                               ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::TanOp, mlir::tt::ttir::TanOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::TanhOp, mlir::tt::ttir::TanhOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::LogOp, mlir::tt::ttir::LogOp>>(typeConverter, ctx);
}

static void
addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
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
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RemOp, mlir::tt::ttir::RemainderOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SelectOp, mlir::tt::ttir::WhereOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::PowOp, mlir::tt::ttir::PowerOp>>(typeConverter, ctx);
}

static void addReduceOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceOpConversionPattern>(typeConverter, ctx);
}

static void addDotGeneralOpConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRDotGeneralOpConversionPattern>(typeConverter,
                                                             ctx);
}

static void
addGetDimensionSizeOpsConversionPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRGetDimensionSizeOpConversionPattern>(
      typeConverter, ctx);
}

static void
addTensorCreationOpsConversionPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConstantOpConversionPattern>(typeConverter, ctx);
}

static void addBroadcastOpConversionPattern(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIRBroadcastInDimOpConversionPattern>(typeConverter,
                                                                 ctx);
}

static void addConv2dOpConversionPattern(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConvolutionOpConversionPattern>(typeConverter,
                                                              ctx);
}

static void addReduceWindowOpConversionPattern(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceWindowOpConversionPattern>(typeConverter,
                                                               ctx);
}

static void addCompareOpsConversionPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRCompareOpConversionPattern>(typeConverter, ctx);
}

static void addConcatOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConcatOpConversionPattern>(typeConverter, ctx);
}

static void addTransposeOpConversionPattern(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRTransposeOpConversionPattern>(typeConverter, ctx);
}

static void addReshapeOpConversionPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReshapeOpConversionPattern>(typeConverter, ctx);
}

static void addCCLOpsConversionPattern(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRAllReduceOpConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRCustomCallOpConversionPattern>(typeConverter,
                                                             ctx);
}

static void
addLogicalAndBitwiseOpsConversionPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::AndOp, mlir::tt::ttir::LogicalAndOp,
      mlir::tt::ttir::BitwiseAndOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::OrOp, mlir::tt::ttir::LogicalOrOp,
      mlir::tt::ttir::BitwiseOrOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::XorOp, mlir::tt::ttir::LogicalXorOp,
      mlir::tt::ttir::BitwiseXorOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::NotOp, mlir::tt::ttir::LogicalNotOp,
      mlir::tt::ttir::BitwiseNotOp>>(typeConverter, ctx);
}

static void addSliceOpConversionPattern(MLIRContext *ctx,
                                        RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRSliceOpConversionPattern>(typeConverter, ctx);
}

static void addClampOpConversionPattern(MLIRContext *ctx,
                                        RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpClampOpConversionPattern>(typeConverter, ctx);
}

static void addGatherOpConversionPattern(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRGatherOpConversionPattern>(typeConverter, ctx);
}

static void addIotaOpConversionPattern(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpIotaOpConversionPattern<stablehlo::IotaOp>>(
      typeConverter, ctx);
  patterns
      .add<StableHLOToTTIROpIotaOpConversionPattern<stablehlo::DynamicIotaOp>>(
          typeConverter, ctx);
}

static void addScatterOpConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRScatterOpConversionPattern>(typeConverter, ctx);
}

static void addReturnOpConversionPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReturnOpConversionPattern>(typeConverter, ctx);
}

static void addReverseOpConversionPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpReverseOpConversionPattern>(typeConverter, ctx);
}

namespace mlir::tt {

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addReduceOpsConversionPatterns(ctx, patterns, typeConverter);
  addDotGeneralOpConversionPatterns(ctx, patterns, typeConverter);
  addGetDimensionSizeOpsConversionPatterns(ctx, patterns, typeConverter);
  addTensorCreationOpsConversionPatterns(ctx, patterns, typeConverter);
  addBroadcastOpConversionPattern(ctx, patterns, typeConverter);
  addConv2dOpConversionPattern(ctx, patterns, typeConverter);
  addReduceWindowOpConversionPattern(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
  addConcatOpsConversionPatterns(ctx, patterns, typeConverter);
  addTransposeOpConversionPattern(ctx, patterns, typeConverter);
  addReshapeOpConversionPattern(ctx, patterns, typeConverter);
  addCCLOpsConversionPattern(ctx, patterns, typeConverter);
  addLogicalAndBitwiseOpsConversionPatterns(ctx, patterns, typeConverter);
  addSliceOpConversionPattern(ctx, patterns, typeConverter);
  addClampOpConversionPattern(ctx, patterns, typeConverter);
  addGatherOpConversionPattern(ctx, patterns, typeConverter);
  addIotaOpConversionPattern(ctx, patterns, typeConverter);
  addScatterOpConversionPatterns(ctx, patterns, typeConverter);
  addReturnOpConversionPatterns(ctx, patterns, typeConverter);
  addReverseOpConversionPattern(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
