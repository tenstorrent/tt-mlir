// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>
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
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

    auto input = Value(adaptor.getOperand());
    auto transposes = getPermutationTransposes(adaptor.getPermutation().vec());

    for (auto transposeDims : transposes) {
      auto dim0 = std::get<0>(transposeDims);
      auto dim1 = std::get<1>(transposeDims);

      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      auto outputShape = inputType.getShape().vec();
      std::swap(outputShape[dim0], outputShape[dim1]);

      auto outputType = RankedTensorType::get(
          outputShape, inputType.getElementType(), inputType.getEncoding());

      auto outputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputShape, outputType.getElementType());

      input = rewriter.create<mlir::tt::ttir::TransposeOp>(
          srcOp.getLoc(), outputType, input, outputTensor,
          rewriter.getSI32IntegerAttr(dim0), rewriter.getSI32IntegerAttr(dim1),
          rewriter.getArrayAttr(
              SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                     rewriter.getAttr<OperandConstraintAttr>(
                                         OperandConstraint::AnyDeviceTile))));
    }
    rewriter.replaceOp(srcOp, input);
    return success();
  }

private:
  std::vector<std::tuple<int64_t, int64_t>>
  getPermutationTransposes(std::vector<int64_t> permutation) const {
    std::vector<std::tuple<int64_t, int64_t>> transposes;
    for (uint32_t i = 0; i < permutation.size(); i++) {
      while (i != permutation[i]) {
        transposes.push_back(
            std::make_tuple(permutation[i], permutation[permutation[i]]));
        std::swap(permutation[i], permutation[permutation[i]]);
      }
    }

    return transposes;
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
        adaptor.getOperand(), outputTensor, new_shape_attr,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    checkForArgumentsAndReplace(srcOp, new_reshape_op);
    rewriter.replaceOp(srcOp, new_reshape_op);
    if (new_reshape_op.getType() == new_reshape_op->getOperand(0).getType()) {
      rewriter.replaceOp(new_reshape_op, new_reshape_op->getOperand(0));
    }
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

private:
  void
  checkForArgumentsAndReplace(mlir::stablehlo::ReshapeOp &srcOp,
                              mlir::tt::ttir::ReshapeOp &newReshapeOp) const {
    if (auto blockArg = mlir::cast<BlockArgument>(srcOp->getOperand(0))) {
      newReshapeOp.setOperand(
          0, srcOp->getParentOfType<mlir::func::FuncOp>().getArgument(
                 blockArg.getArgNumber()));
    }
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
        adaptor.getFeatureGroupCountAttr(), adaptor.getBatchGroupCountAttr(),
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

    auto operandConstraints = rewriter.getArrayAttr(SmallVector<Attribute>(
        adaptor.getOperands().size(), rewriter.getAttr<OperandConstraintAttr>(
                                          OperandConstraint::AnyDeviceTile)));

    mlir::tt::ttir::PoolingMethod poolingMethod;
    if (isMaxPool(srcOp)) {
      poolingMethod = mlir::tt::ttir::PoolingMethod::Max;
    } else {
      return rewriter.notifyMatchFailure(srcOp, "Unsupported pooling method");
    }

    for (Value initValue : adaptor.getInitValues()) {
      eraseInitValueSubgraph(rewriter, initValue.getDefiningOp());
    }

    rewriter.replaceOpWithNewOp<ttir::PoolingOp>(
        srcOp, outputType, adaptor.getInputs(), outputs, poolingMethod,
        windowDimensions, windowStrides, baseDilations, window_dilations,
        padding, operandConstraints);

    return success();
  }

private:
  void eraseInitValueSubgraph(ConversionPatternRewriter &rewriter,
                              Operation *op) const {

    std::vector<Operation *> opsToErase;
    opsToErase.push_back(op);

    bool addedOps = true;
    while (addedOps) {
      addedOps = false;
      Operation *currentOp = opsToErase.back();

      for (auto &operand : currentOp->getOpOperands()) {
        Operation *definingOp = operand.get().getDefiningOp();
        if (definingOp->hasOneUse() || definingOp->use_empty()) {
          addedOps = true;
          opsToErase.push_back(definingOp);
        }
      }
    }

    for (auto &op : opsToErase) {
      rewriter.eraseOp(op);
    }
  }

  // Just to make the code more readable
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
    // since bfloat16 is not a type we acually have I must compare the raw
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
            rewriter.getF32FloatAttr(maxValue),
            rewriter.getArrayAttr(
                SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                       rewriter.getAttr<OperandConstraintAttr>(
                                           OperandConstraint::AnyDeviceTile))));

        return success();
      }
    }

    ttir::MaximumOp maximumOp = rewriter.create<mlir::tt::ttir::MaximumOp>(
        srcOp->getLoc(), min, adaptor.getOperand(), outputTensor,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));

    tensor::EmptyOp finalOutputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::MinimumOp>(
        srcOp, maximumOp->getResult(0), max, finalOutputTensor,
        rewriter.getArrayAttr(
            SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    return success();
  }
};

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
        srcOp, outputType, srcOp.getOperands()[0],
        srcOp.getOperands()[1], // Start indices
        Value(outputTensor), dimensionNumbers.getOffsetDims(),
        dimensionNumbers.getCollapsedSliceDims(),
        dimensionNumbers.getOperandBatchingDims(),
        dimensionNumbers.getStartIndicesBatchingDims(),
        dimensionNumbers.getStartIndexMap(),
        dimensionNumbers.getIndexVectorDim(), srcOp.getSliceSizesAttr(), false,
        rewriter.getArrayAttr(
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
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RemOp, mlir::tt::ttir::RemainderOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SelectOp, mlir::tt::ttir::WhereOp>>(typeConverter, ctx);
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

void addGetDimensionSizeOpsConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRGetDimensionSizeOpConversionPattern>(
      typeConverter, ctx);
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
  patterns.add<StableHLOToTTIROpLogicalOpConversionPattern<
      mlir::stablehlo::XorOp, mlir::tt::ttir::LogicalXorOp>>(typeConverter,
                                                             ctx);
}

void addSliceOpConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRSliceOpConversionPattern>(typeConverter, ctx);
}

void addClampOpConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpClampOpConversionPattern>(typeConverter, ctx);
}

void addGatherOpConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRGatherOpConversionPattern>(typeConverter, ctx);
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
  addGetDimensionSizeOpsConversionPatterns(ctx, patterns, typeConverter);
  addTensorCreationOpsConversionPatterns(ctx, patterns, typeConverter);
  addBroadcastOpConversionPattern(ctx, patterns, typeConverter);
  addConv2dOpConversionPattern(ctx, patterns, typeConverter);
  addReduceWindowOpConversionPattern(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
  addConcatOpsConversionPatterns(ctx, patterns, typeConverter);
  addReshapeOpConversionPattern(ctx, patterns, typeConverter);
  addLogicalOpConversionPattern(ctx, patterns, typeConverter);
  addSliceOpConversionPattern(ctx, patterns, typeConverter);
  addClampOpConversionPattern(ctx, patterns, typeConverter);
  addGatherOpConversionPattern(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
