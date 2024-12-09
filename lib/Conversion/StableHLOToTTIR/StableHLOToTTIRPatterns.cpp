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
        1, rewriter.getI32IntegerAttr(adaptor.getDimensionsAttr().size() > 0
                                          ? adaptor.getDimensionsAttr()[0]
                                          : 1)));

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

    if (dimensions.getLhsContractingDimensions().size() != 1 ||
        dimensions.getRhsContractingDimensions().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "LHS and RHS must have exactly 1 contracting dimension each. "
          "Received LHS contracting dims: " +
              std::to_string(dimensions.getLhsContractingDimensions().size()) +
              ", RHS contracting dims: " +
              std::to_string(dimensions.getRhsContractingDimensions().size()));
    }

    // Use negative indexing to determine if this is a valid matmul since math
    // is done over the final two dimensions.
    int64_t lhsContractingDim = dimensions.getLhsContractingDimensions()[0] -
                                srcOp.getLhs().getType().getRank();
    int64_t rhsContractingDim = dimensions.getRhsContractingDimensions()[0] -
                                srcOp.getRhs().getType().getRank();

    if (lhsContractingDim != -1) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only support contracting dimensions that correspond to valid "
                 "matmuls. LHS contracting dimension must be " +
                     std::to_string(srcOp.getLhs().getType().getRank() - 1) +
                     ". Got " + std::to_string(lhsContractingDim));
    }

    if (rhsContractingDim != -2) {
      return rewriter.notifyMatchFailure(
          srcOp, "Only support contracting dimensions that correspond to valid "
                 "matmuls. RHS contracting dimension must be " +
                     std::to_string(srcOp.getRhs().getType().getRank() - 2) +
                     ". Got " + std::to_string(rhsContractingDim));
    }

    if (dimensions.getLhsBatchingDimensions() !=
        dimensions.getRhsBatchingDimensions()) {
      return rewriter.notifyMatchFailure(
          srcOp, "LHS and RHS must have same batching dimensions.");
    }

    // For the RHS, all dimensions which are not the row and column dimensions
    // must be 1 OR they must be equal to the corresponding dimension in the
    // LHS. If the RHS has less dimensions than the LHS we will assume that the
    // missing dimensions are 1.

    auto lhsShape = srcOp.getLhs().getType().getShape().vec();
    auto rhsShape = srcOp.getRhs().getType().getShape().vec();

    if (rhsShape.size() > lhsShape.size()) {
      return rewriter.notifyMatchFailure(
          srcOp, "RHS must not be a higher rank than LHS.");
    }

    while (rhsShape.size() < lhsShape.size()) {
      rhsShape.insert(rhsShape.begin(), 1);
    }

    // Need only to check dims to the left of dim -2 on the RHS
    bool allOnes = true;
    bool mismatchedDims = false;
    for (int32_t i = rhsShape.size() - 3; i >= 0; i--) {
      if (rhsShape[i] != 1) {
        allOnes = false;
      }

      if (rhsShape[i] != lhsShape[i]) {
        mismatchedDims = true;
      }
    }

    if (mismatchedDims && !allOnes) {
      return rewriter.notifyMatchFailure(
          srcOp, "All dimensions in the RHS that are not the row and column "
                 "dimensions must be 1 OR they must all be equal to the "
                 "corresponding dimensions in the LHS.");
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

    // Algorithm here is to search for the first non-one working dimension
    auto replicaGroupsShape = adaptor.getReplicaGroups().getType().getShape();
    size_t dim = 0;
    for (auto s : replicaGroupsShape) {
      if (s != 1) {
        break;
      }
      ++dim;
    }
    if (dim > replicaGroupsShape.size()) {
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

    StringAttr operationConstraintAttrName =
        StringAttr::get(this->getContext(), "operand_constraints");
    Attribute operationConstraintAttr = rewriter.getArrayAttr(
        SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                               rewriter.getAttr<OperandConstraintAttr>(
                                   OperandConstraint::AnyDeviceTile)));
    ttirAttrs.push_back({operationConstraintAttrName, operationConstraintAttr});

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
}; // namespace

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

    StringAttr operationConstraintAttrName =
        StringAttr::get(this->getContext(), "operand_constraints");
    Attribute operationConstraintAttr = rewriter.getArrayAttr(
        SmallVector<Attribute>(adaptor.getOperands().size() + 1,
                               rewriter.getAttr<OperandConstraintAttr>(
                                   OperandConstraint::SystemScalar)));
    meshShardAttrs.push_back(
        {operationConstraintAttrName, operationConstraintAttr});

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
}; // namespace

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
        srcOp, outputType, adaptor.getOperands()[0],
        adaptor.getOperands()[1], // Start indices
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
    // already known by the result type This is to remove the operand that will
    // become dead code
    for (auto operand : adaptor.getOperands()) {
      if (operand.getDefiningOp()) {
        rewriter.eraseOp(operand.getDefiningOp());
      }
    }

    return success();
  }
};

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
    mlir::ArrayAttr binaryConstraints = rewriter.getArrayAttr(
        SmallVector<Attribute>(4, rewriter.getAttr<OperandConstraintAttr>(
                                      OperandConstraint::AnyDeviceTile)));
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
        indexVectorDim, indicesAreSorted, uniqueIndices, outputTensor,
        binaryConstraints);

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

void addCCLOpsConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRAllReduceOpConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRCustomCallOpConversionPattern>(typeConverter,
                                                             ctx);
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

void addIotaOpConversionPattern(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpIotaOpConversionPattern<stablehlo::IotaOp>>(
      typeConverter, ctx);
  patterns
      .add<StableHLOToTTIROpIotaOpConversionPattern<stablehlo::DynamicIotaOp>>(
          typeConverter, ctx);
}

void addScatterOpConversionPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRScatterOpConversionPattern>(typeConverter, ctx);
}

void addReturnOpConversionPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReturnOpConversionPattern>(typeConverter, ctx);
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
  addCCLOpsConversionPattern(ctx, patterns, typeConverter);
  addSliceOpConversionPattern(ctx, patterns, typeConverter);
  addClampOpConversionPattern(ctx, patterns, typeConverter);
  addGatherOpConversionPattern(ctx, patterns, typeConverter);
  addIotaOpConversionPattern(ctx, patterns, typeConverter);
  addScatterOpConversionPatterns(ctx, patterns, typeConverter);
  addReturnOpConversionPatterns(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
