// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

//===----------------------------------------------------------------------===//
// IndexOp decomposition
//===----------------------------------------------------------------------===//

// ANCHOR: decomposing_an_op_index_ttir_decompose_pattern
// This transformation adjusts IndexOp attributes so that `begin`, `end`, and
// `step` become arrays, where each array element corresponds to a dimension of
// the input tensor. For dimensions other than the sliced dimension, default
// values are used.
//
struct IndexToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexOp> {
  using OpConversionPattern<ttir::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType =
        ::mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!inputType || !inputType.hasRank()) {
      return failure();
    }

    int64_t rank = inputType.getRank();
    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;

    for (int64_t i = 0; i < rank; ++i) {
      if (i == op.getDim()) {
        begins.push_back(rewriter.getI32IntegerAttr(adaptor.getBegin()));
        ends.push_back(rewriter.getI32IntegerAttr(adaptor.getEnd()));
        steps.push_back(rewriter.getI32IntegerAttr(adaptor.getStep()));
      } else {
        begins.push_back(rewriter.getI32IntegerAttr(0));
        ends.push_back(rewriter.getI32IntegerAttr(inputType.getDimSize(i)));
        steps.push_back(rewriter.getI32IntegerAttr(1));
      }
    }

    auto newOp = rewriter.create<ttir::SliceOp>(
        op.getLoc(), op.getType(), adaptor.getInput(), adaptor.getOutput(),
        rewriter.getArrayAttr(begins), rewriter.getArrayAttr(ends),
        rewriter.getArrayAttr(steps));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};
// ANCHOR_END: decomposing_an_op_index_ttir_decompose_pattern

//===----------------------------------------------------------------------===//
// Convolution passes
//===----------------------------------------------------------------------===//

using TransposeDims = std::tuple<int64_t, int64_t>;

template <uint32_t NDims>
using PaddingMatrix = std::array<std::array<int64_t, 2>, NDims>;

template <uint32_t NDims>
static PaddingMatrix<NDims> getPaddingMatrix(ArrayRef<int64_t> padding) {
  PaddingMatrix<NDims> paddingMatrix;
  std::vector<int64_t> paddingFlattened = padding.vec();

  for (uint32_t i = 0; i < 2 * NDims; i += 2) {
    paddingMatrix[i / 2] = {paddingFlattened[i], paddingFlattened[i + 1]};
  }
  return paddingMatrix;
}
/*
 * The following functions are used to generate the transpose operations needed
 * to convert a convolution operation to the specific op definitions for a
 * ConvNdOp for any N spatial dimensions.
 *
 * All convolutions will have a batch and feature dimension, and the kernel will
 * have an input and output feature dimension. The spatial dimensions can be
 * represented by non-negative integers.
 */
enum ConvolutionDimension { BATCH = -1, FEATURE = -2, INVALID_DIM = -3 };

enum ConvolutionKernelDimension {
  INPUT_FEATURES = -1,
  OUTPUT_FEATURES = -2,
  INVALID_KERNEL_DIM = -3
};

/*
 * Generates a sequence of dims in which to transpose to make currentLayout
 * match desiredLayout
 *
 * Ex: if currentLayout = [0, 1, 2, 3] and desiredLayout = [0, 2, 3, 1]
 * then the function will return [(1, 2), (2, 3)] because when we swap
 * currentLayout[1] with currentLayout[2] we get [0, 2, 1, 3], and then when
 * we swap currentLayout[2] with currentLayout[3] we get [0, 2, 3, 1], which
 * is the desired layout
 */
static std::vector<TransposeDims>
generateTransposeIndices(std::vector<int64_t> currentLayout,
                         const std::vector<int64_t> desiredLayout) {
  std::vector<TransposeDims> transposeIndices;
  for (int64_t i = 0; i < static_cast<int64_t>(currentLayout.size()); i++) {
    if (currentLayout[i] != desiredLayout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(currentLayout.begin(), currentLayout.end(),
                               desiredLayout[i]) -
                     currentLayout.begin();
      transposeIndices.push_back(std::make_tuple(dim0, dim1));
      std::swap(currentLayout[dim0], currentLayout[dim1]);
    }
  }

  return transposeIndices;
}

/*
 * This function will use a sequence of transpose indices to
 * generate the actual transpose operations descrbibed by them.
 *
 * It takes an input to apply these transposes to and returns the
 * result at the end of the sequence
 */
static Value generateTransposeOps(Value input, PatternRewriter &rewriter,
                                  std::vector<TransposeDims> transposeIndices) {
  for (auto [dim0, dim1] : transposeIndices) {

    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto outputShape = inputType.getShape().vec();
    std::swap(outputShape[dim0], outputShape[dim1]);

    auto dim0Attr = rewriter.getSI32IntegerAttr(dim0);
    auto dim1Attr = rewriter.getSI32IntegerAttr(dim1);

    auto outputType = RankedTensorType::get(
        outputShape, inputType.getElementType(), inputType.getEncoding());

    auto dpsOutput = rewriter.create<tensor::EmptyOp>(
        input.getLoc(), outputShape, outputType.getElementType());
    input = rewriter
                .create<ttir::TransposeOp>(input.getLoc(), outputType, input,
                                           dpsOutput, dim0Attr, dim1Attr)
                .getResult();
  }

  return input;
}

/*
 * This function will generate the transpose indices needed to convert a
 * convolution input to a desired layout. The reason for the separate
 * function is to encapsulate the logic for constructuring the inputLayout
 */
static std::vector<TransposeDims>
generateConvTransposeIndices(ttir::ConvolutionOp op,
                             const std::vector<int64_t> ttnnConvolutionLayout) {

  std::vector<int64_t> inputLayout(ttnnConvolutionLayout.size(),
                                   ConvolutionDimension::INVALID_DIM);
  inputLayout[op.getConvolutionLayout().getInputBatchDimension()] =
      ConvolutionDimension::BATCH;
  inputLayout[op.getConvolutionLayout().getInputFeatureDimension()] =
      ConvolutionDimension::FEATURE;

  int64_t spatialCount = 0;
  for (int64_t spatialDim :
       op.getConvolutionLayout().getInputSpatialDimensions()) {
    inputLayout[spatialDim] = spatialCount;
    spatialCount++;
  }

  return generateTransposeIndices(inputLayout, ttnnConvolutionLayout);
}

/*
 * This function will generate the transpose indices needed to convert a
 * convolution input to a desired layout. The reason for the separate
 * function is to encapsulate the logic for constructuring the kernelLayout
 */
static std::vector<TransposeDims> generateConvKernelTransposeIndices(
    ttir::ConvolutionOp op,
    const std::vector<int64_t> ttnnConvolutionKernelLayout) {
  std::vector<TransposeDims> transposeIndices;

  std::vector<int64_t> kernelLayout(
      ttnnConvolutionKernelLayout.size(),
      ConvolutionKernelDimension::INVALID_KERNEL_DIM);
  kernelLayout[op.getConvolutionLayout().getKernelOutputFeatureDimension()] =
      ConvolutionKernelDimension::OUTPUT_FEATURES;
  kernelLayout[op.getConvolutionLayout().getKernelInputFeatureDimension()] =
      ConvolutionKernelDimension::INPUT_FEATURES;

  int64_t spatialCount = 0;
  for (int64_t spatialDim :
       op.getConvolutionLayout().getKernelSpatialDimensions()) {
    kernelLayout[spatialDim] = spatialCount;
    spatialCount++;
  }

  return generateTransposeIndices(kernelLayout, ttnnConvolutionKernelLayout);
}

struct ConvolutionDecompositionPattern
    : public OpConversionPattern<ttir::ConvolutionOp> {
public:
  using OpConversionPattern<ttir::ConvolutionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override = 0;

protected:
  bool isNDimensional(ttir::ConvolutionOp op, uint32_t numSpatialDims) const {
    return op.getConvolutionLayout().getInputSpatialDimensions().size() ==
           numSpatialDims;
  }

  bool isSupportedConv(ttir::ConvolutionOp op) const {
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getOutputSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getKernelSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");

    // Not currently supporting window reversal
    std::vector<bool> windowReversal(op.getWindowReversal().begin(),
                                     op.getWindowReversal().end());
    for (bool reversed : windowReversal) {
      if (reversed) {
        return false;
      }
    }

    // Not currently support batch groups
    if (op.getBatchGroupCount() != 1) {
      return false;
    }

    return true;
  }
};

// A decomposition pattern that matches to a ttir.convolution op that does 1D
// convolution. Since that is not supported in ttnn, we reshape the inputs and
// the output to match a 2D ttir.convolution op. The expectation is that the new
// ttir.convolution op will be picked up by the ConvolutionToConv2dPattern and
// translated into ttir.conv2d op.
struct Legalize1DConvolutionPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;
  constexpr static uint32_t numSpatialDims = 1;

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, numSpatialDims))) {
      return failure();
    }

    // Not currently supporting spatial dims other than 2 for the 1D case.
    if (op.getConvolutionLayout().getInputSpatialDimensions()[0] != 2) {
      return failure();
    }

    // The shapes that the convolution currently operates with have are 3D, and
    // we need to add another dimension for it to match the conv2d signature, so
    // adding a dimension of size 1 to the end of input and output shapes.
    auto outputType =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
    llvm::SmallVector<int64_t, 4> conv2dOutputShape(outputShape.begin(),
                                                    outputShape.end());
    conv2dOutputShape.push_back(1);
    auto DPSConv2dOutput = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), conv2dOutputShape, outputType.getElementType());
    auto conv2dOutputType =
        mlir::cast<RankedTensorType>(DPSConv2dOutput.getType());

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t, 4> reshapeInputShape(inputShape.begin(),
                                                    inputShape.end());
    reshapeInputShape.push_back(1);

    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();
    llvm::SmallVector<int64_t, 4> reshapeWeightShape(weightShape.begin(),
                                                     weightShape.end());
    reshapeWeightShape.push_back(1);

    ttir::ReshapeOp reshapeInput = createReshapeOp(
        op.getLoc(), adaptor.getInput(), reshapeInputShape, rewriter);
    ttir::ReshapeOp reshapeWeight = createReshapeOp(
        op.getLoc(), adaptor.getWeight(), reshapeWeightShape, rewriter);

    mlir::DenseI64ArrayAttr conv2dOpWindowsStridesAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getWindowStridesAttr(), 1);
    mlir::DenseI64ArrayAttr conv2dOpPaddingAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getPaddingAttr(), 0);
    conv2dOpPaddingAttr =
        addIntegerToDenseArrayAttr(rewriter, conv2dOpPaddingAttr, 0);
    mlir::DenseI64ArrayAttr conv2dOpInputDilationAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getInputDilationAttr(), 1);
    mlir::DenseI64ArrayAttr conv2dOpWeightDilationAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getWeightDilationAttr(),
                                   1);
    mlir::DenseBoolArrayAttr conv2dOpWindowReversalAttr =
        addBooleanToDenseArrayAttr(rewriter, adaptor.getWindowReversalAttr(),
                                   false);

    auto convolutionLayout = adaptor.getConvolutionLayoutAttr();

    // The additional spatial dimension is added at the and (3rd in 0 indexed
    // array).
    llvm::SmallVector<int64_t, 4> conv2dInputSpatialDimensions(
        convolutionLayout.getInputSpatialDimensions().begin(),
        convolutionLayout.getInputSpatialDimensions().end());
    conv2dInputSpatialDimensions.push_back(3);

    llvm::SmallVector<int64_t, 4> conv2dKernelSpatialDimensions(
        convolutionLayout.getKernelSpatialDimensions().begin(),
        convolutionLayout.getKernelSpatialDimensions().end());
    conv2dKernelSpatialDimensions.push_back(3);

    llvm::SmallVector<int64_t, 4> conv2dOutputSpatialDimensions(
        convolutionLayout.getOutputSpatialDimensions().begin(),
        convolutionLayout.getOutputSpatialDimensions().end());
    conv2dOutputSpatialDimensions.push_back(3);

    mlir::tt::ttir::ConvolutionOp new2dConvolutionOp =
        rewriter.create<mlir::tt::ttir::ConvolutionOp>(
            op.getLoc(), conv2dOutputType, reshapeInput, reshapeWeight,
            mlir::Value(nullptr), DPSConv2dOutput, conv2dOpWindowsStridesAttr,
            conv2dOpPaddingAttr, conv2dOpInputDilationAttr,
            conv2dOpWeightDilationAttr, conv2dOpWindowReversalAttr,
            mlir::tt::ttir::ConvolutionLayoutAttr::get(
                getContext(), convolutionLayout.getInputBatchDimension(),
                convolutionLayout.getInputFeatureDimension(),
                conv2dInputSpatialDimensions,
                convolutionLayout.getKernelOutputFeatureDimension(),
                convolutionLayout.getKernelInputFeatureDimension(),
                conv2dKernelSpatialDimensions,
                convolutionLayout.getOutputBatchDimension(),
                convolutionLayout.getOutputFeatureDimension(),
                conv2dOutputSpatialDimensions),
            adaptor.getFeatureGroupCountAttr(),
            adaptor.getBatchGroupCountAttr());
    ttir::ReshapeOp reshapeOutput =
        createReshapeOp(op.getLoc(), new2dConvolutionOp, outputShape, rewriter);

    rewriter.replaceOp(op, reshapeOutput);

    return success();
  }

private:
  ttir::ReshapeOp createReshapeOp(Location loc, Value tensor,
                                  llvm::ArrayRef<int64_t> target_input_shape,
                                  ConversionPatternRewriter &rewriter) const {
    auto inputType = mlir::cast<RankedTensorType>(tensor.getType());

    auto DPSReshapeOutput = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>(target_input_shape),
        inputType.getElementType());
    llvm::SmallVector<int32_t, 2> shapei32(target_input_shape.begin(),
                                           target_input_shape.end());
    auto shape_attr = rewriter.getI32ArrayAttr(shapei32);

    return rewriter.create<ttir::ReshapeOp>(
        loc,
        mlir::RankedTensorType::get(target_input_shape,
                                    inputType.getElementType()),
        tensor, DPSReshapeOutput, shape_attr);
  }

  mlir::DenseI64ArrayAttr
  addIntegerToDenseArrayAttr(ConversionPatternRewriter &rewriter,
                             mlir::DenseI64ArrayAttr denseArrayAttr,
                             uint64_t integerValue) const {
    llvm::SmallVector<int64_t, 4> newDenseArray(denseArrayAttr.asArrayRef());
    newDenseArray.push_back(integerValue);
    return rewriter.getDenseI64ArrayAttr(newDenseArray);
  }

  mlir::DenseBoolArrayAttr
  addBooleanToDenseArrayAttr(ConversionPatternRewriter &rewriter,
                             mlir::DenseBoolArrayAttr denseArrayAttr,
                             bool booleanValue) const {
    llvm::SmallVector<bool, 4> newDenseArray(denseArrayAttr.asArrayRef());
    newDenseArray.push_back(booleanValue);
    return rewriter.getDenseBoolArrayAttr(newDenseArray);
  }
};
struct ConvolutionToConv2dPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;

  constexpr static uint32_t numSpatialDims = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC
  static inline const std::vector<int64_t> conv2dLayout = {
      ConvolutionDimension::BATCH,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
      ConvolutionDimension::FEATURE,
  };
  // OIHW
  static inline const std::vector<int64_t> conv2dKernelLayout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, numSpatialDims))) {
      return failure();
    }

    auto strideHeightAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_HEIGHT]);
    auto strideWidthAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_WIDTH]);
    auto dilationHeightAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT]);
    auto dilationWidthAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH]);

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order For Conv2d the most
    // significant spatial dimension is the height, followed by the width.
    auto paddingMatrix = getPaddingMatrix<numSpatialDims>(adaptor.getPadding());
    auto paddingTopAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_HEIGHT][0]);
    auto paddingBottomAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_HEIGHT][1]);
    auto paddingLeftAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_WIDTH][0]);
    auto paddingRightAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_WIDTH][1]);

    auto groupsAttr =
        rewriter.getSI32IntegerAttr(adaptor.getFeatureGroupCount());

    auto outputShape = op.getResult().getType().getShape().vec();
    std::vector<int64_t> newOutputShape = {
        outputShape[adaptor.getConvolutionLayout().getOutputBatchDimension()],
        outputShape[adaptor.getConvolutionLayout()
                        .getOutputSpatialDimensions()[SPATIAL_DIM_HEIGHT]],
        outputShape[adaptor.getConvolutionLayout()
                        .getOutputSpatialDimensions()[SPATIAL_DIM_WIDTH]],
        outputShape[adaptor.getConvolutionLayout()
                        .getOutputFeatureDimension()]};

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType =
        inputType.cloneWith(newOutputShape, inputType.getElementType());

    auto convDPSOutput = rewriter.create<tensor::EmptyOp>(
        adaptor.getInput().getLoc(), newOutputShape,
        outputType.getElementType());

    auto transposeIndices = generateConvTransposeIndices(op, conv2dLayout);
    Value input =
        generateTransposeOps(adaptor.getInput(), rewriter, transposeIndices);

    auto kernelTransposeIndices =
        generateConvKernelTransposeIndices(op, conv2dKernelLayout);
    Value weight = generateTransposeOps(adaptor.getWeight(), rewriter,
                                        kernelTransposeIndices);
    ttir::Conv2dOp newConv = rewriter.create<ttir::Conv2dOp>(
        op.getLoc(), outputType, input, weight, adaptor.getBias(),
        convDPSOutput, strideHeightAttr, strideWidthAttr, dilationHeightAttr,
        dilationWidthAttr, groupsAttr, paddingLeftAttr, paddingRightAttr,
        paddingTopAttr, paddingBottomAttr);

    // Applying the transposes in reverse order to the output will restore the
    // tensor to the original layout
    std::reverse(transposeIndices.begin(), transposeIndices.end());
    Value output =
        generateTransposeOps(newConv.getResult(), rewriter, transposeIndices);

    rewriter.replaceOp(op, output);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Gather Pattern Matching
//===----------------------------------------------------------------------===//

struct GatherToEmbeddingConversionPattern
    : public OpConversionPattern<ttir::GatherOp> {
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  /**
   * Validates Gather op constraints for embedding conversion
   *
   * Enforces constraints on Gather operation to ensure valid embedding
   * transformation:
   * - Output tensor shape: Multi-dimensional with last dimension as embedding
   * size/ hiddenDim
   * - Slice sizes: Must be [1, hiddenDim], where hiddenDim matches last output
   * dimension
   * - Offset dimensions: Strictly [2]
   * - Collapsed slice dimensions: Strictly [0]
   * - Start indices shape: Must be compatible with output shape
   *   - startIndices.size() < output.size()
   *   - if startIndices.size() == output.size(), then startIndices[-1] == 1
   *   - Last dimension of start indices can be reduced by reshape op.
   *   - This is due to embedding weights requiring to have smaller size than
   * output shape
   *
   */

  LogicalResult checkBasicLegality(ttir::GatherOp op,
                                   PatternRewriter &rewriter) const {

    // variables for embedding pattern matching checks
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto shape = outputType.getShape();

    // start indices of the gather op:
    auto startIndices = op.getStartIndices();
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());

    // slice sizes of the gather op
    auto sliceSizes = op.getSliceSizes();
    auto offsetDims = op.getOffsetDims();
    // collapsed slice dims of the gather op
    auto collapsedSliceDims = op.getCollapsedSliceDims();

    RankedTensorType operandType =
        mlir::cast<RankedTensorType>(op->getOperand(0).getType());
    if (!operandType.getElementType().isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "only supports bfloat16 input tensor.");
    }

    if (shape.size() > 1) {
      auto hiddenDim = shape[shape.size() - 1];
      // check if sliceSizes has more than one element
      if (sliceSizes.size() <= 1) {
        return rewriter.notifyMatchFailure(op, "Did not satisfy sliceSizes");
      }
      // check if sliceSizes is [1, hiddenDim]
      if (sliceSizes[0] != 1 || sliceSizes[1] != hiddenDim) {
        return rewriter.notifyMatchFailure(op, "Did not satisfy sliceSizes");
      }
    }

    // check if offsetDims is [2]
    if (offsetDims.size() > 1 || offsetDims[0] != 2) {
      return rewriter.notifyMatchFailure(op, "Did not satisfy offsetDims");
    }

    // check if collapsedSliceDims is [0]
    if (collapsedSliceDims.size() > 1 || collapsedSliceDims[0] != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "Did not satisfy collapsedSliceDims");
    }

    // check if startIndices and output have same shape, if not, check if
    // reshape is possible can reshape startIndices to remove the last dimension
    // if it is 1
    if (shape.size() == startIndicesType.getShape().size() &&
        startIndicesType.getShape()[shape.size() - 1] != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "Did not satisfy startIndicesType");
    }

    return success();
  }

  ttir::ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                                  Value input,
                                  ::llvm::ArrayRef<int64_t> shapei64) const {

    // reshape start indices (input) to remove the last dimension
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>(shapei64), ty.getElementType());
    std::vector<int32_t> shapei32(shapei64.begin(), shapei64.end());
    auto shape_attr = rewriter.getI32ArrayAttr(shapei32);

    return rewriter.create<ttir::ReshapeOp>(
        loc, mlir::RankedTensorType::get(shapei64, ty.getElementType()), input,
        output, shape_attr);
  }

  /**
   * Lowers Gather Op into Embedding Op (and applies Reshape Op, if necessary)
   *
   * - There is no TTNN Gather support.
   *
   * - TTIR Gather Op is lowered into TTIR Embedding Op. Torch embeddings are
   * lowered into Gather Op. Most models use Gather Op to implement simple
   * embeddings.
   *
   * - If encountered more complicated Gather Op implementations, they can be
   * lowered into slice/ concat/ etc.
   *
   * - Start Indices of Gather Op are expected to match Weight for Embeddings.
   * startIndices.size() < weight.size() however, if
   * startIndices.size() == weight.size() && startIndices[-1] == 1,
   * we can apply Reshape Op to reduce the last dimension.
   *
   */

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // GatherOp can be used to implement embedding lookup, check for that case
    LogicalResult err = checkBasicLegality(op, rewriter);
    if (not err.succeeded()) {
      return err;
    }
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto shape = outputType.getShape();

    // start indices of the gather op
    auto startIndices = op.getStartIndices();
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());

    // check if start indices need to be reshaped
    ::mlir::Value input = op.getStartIndices();
    if (shape.size() == startIndicesType.getShape().size() &&
        startIndicesType.getShape()[shape.size() - 1] == 1) {
      // reduce weight tensor dimension
      // insert reshape op to remove the last dimension of start indices
      // before gather/ embedding op
      std::vector<int64_t> newShapeI64(startIndicesType.getShape().begin(),
                                       startIndicesType.getShape().end() - 1);

      ttir::ReshapeOp reshapeOp =
          createReshapeOp(rewriter, op.getLoc(), startIndices, newShapeI64);

      assert(reshapeOp && "Failed to create reshape op");
      reshapeOp->moveBefore(op);
      input = reshapeOp.getResult();
    }

    // convert gather to embedding, use reshaped input if needed
    ttir::EmbeddingOp embeddingOp = rewriter.create<ttir::EmbeddingOp>(
        op.getLoc(), op.getResult().getType(), input, op.getOperands()[0],
        op.getOutput());

    assert(embeddingOp != nullptr && "Failed to create embedding op");
    rewriter.replaceOp(op, embeddingOp);

    return success();
  }
};

struct PoolingToPool2dPattern : public OpConversionPattern<ttir::PoolingOp> {
public:
  using OpConversionPattern<ttir::PoolingOp>::OpConversionPattern;

  std::vector<int64_t> getIndicesOfSpatialDims(ttir::PoolingOp op) const {
    std::vector<int64_t> spatialDims;
    for (int64_t i = 0;
         i < static_cast<int64_t>(op.getWindowDimensions().size()); i++) {
      if (op.getWindowDimensions()[i] > 1) {
        spatialDims.push_back(i);
      }
    }
    return spatialDims;
  }

  LogicalResult canDecompose2DPoolingOp(ttir::PoolingOp op) const {

    // Window dimensions must be 4 in length
    if (op.getWindowDimensions().size() != 4) {
      return failure();
    }

    // Window strides must be 4 in length
    if (op.getWindowStrides().size() != 4) {
      return failure();
    }

    // Operand rank(s) must be 4
    for (Value operand : op.getInputs()) {
      auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
      if (operandType.getRank() != 4) {
        return failure();
      }
    }

    // Exactly two of the window dimensions must be greater than 1
    std::vector<int64_t> trueWindowDimensionsIndices =
        getIndicesOfSpatialDims(op);

    if (trueWindowDimensionsIndices.size() != 2) {
      return failure();
    }

    // Exactly two of the window strides must be greater than 1
    std::vector<int64_t> trueWindowStrideIndices;
    for (int64_t i = 0; i < static_cast<int64_t>(op.getWindowStrides().size());
         i++) {
      if (op.getWindowStrides()[i] > 1) {
        trueWindowStrideIndices.push_back(i);
      }
    }

    if (trueWindowStrideIndices.size() != 2) {
      return failure();
    }

    // The indices of the true window dimensions and strides must be the same
    if ((trueWindowDimensionsIndices[0] != trueWindowStrideIndices[0] ||
         trueWindowDimensionsIndices[1] != trueWindowStrideIndices[1]) &&
        (trueWindowDimensionsIndices[0] != trueWindowStrideIndices[1] ||
         trueWindowDimensionsIndices[1] != trueWindowStrideIndices[0])) {
      return failure();
    }

    // Padding must be 8 in length
    if (op.getPadding().size() != 8) {
      return failure();
    }

    return success();
  }

  template <typename PoolOpType>
  void rewritePool2d(ttir::PoolingOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    const int64_t SPATIAL_H = -3;
    const int64_t SPATIAL_W = -2;
    const int64_t NON_SPATIAL = -1;

    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getInputs()[0].getType());
    assert(inputType.getRank() == 4 && "Input must be 4D tensor");
    std::vector<int64_t> desiredLayout(inputType.getRank(), NON_SPATIAL);
    desiredLayout[inputType.getRank() - 3] = SPATIAL_H;
    desiredLayout[inputType.getRank() - 2] = SPATIAL_W;

    int64_t nonSpatialCount = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(desiredLayout.size()); i++) {
      if (desiredLayout[i] == NON_SPATIAL) {
        desiredLayout[i] = nonSpatialCount;
        nonSpatialCount++;
      }
    }

    std::vector<int64_t> spatialDims = getIndicesOfSpatialDims(op);

    std::vector<int64_t> currentLayout(inputType.getRank(), NON_SPATIAL);
    currentLayout[spatialDims[0]] = SPATIAL_H;
    currentLayout[spatialDims[1]] = SPATIAL_W;

    nonSpatialCount = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(currentLayout.size()); i++) {
      if (currentLayout[i] == NON_SPATIAL) {
        currentLayout[i] = nonSpatialCount;
        nonSpatialCount++;
      }
    }

    auto transposeIndices =
        generateTransposeIndices(currentLayout, desiredLayout);

    auto kernelHeightAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowDimensions()[spatialDims[0]]));
    auto kernelWidthAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowDimensions()[spatialDims[1]]));

    auto strideHeightAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowStrides()[spatialDims[0]]));

    auto strideWidthAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowStrides()[spatialDims[1]]));

    auto dilationHeightAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowDilations()[spatialDims[0]]);
    auto dilationWidthAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowDilations()[spatialDims[1]]);
    auto ceilModeAttr = rewriter.getBoolAttr(false);

    auto paddingTopAttr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatialDims[0]]);
    auto paddingBottomAttr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatialDims[0] + 1]);
    auto paddingLeftAttr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatialDims[1]]);
    auto paddingRightAttr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatialDims[1] + 1]);

    std::vector<Value> outputs;
    for (Value input : adaptor.getInputs()) {
      input = generateTransposeOps(input, rewriter, transposeIndices);

      auto outputType = mlir::cast<RankedTensorType>(op.getResult(0).getType());
      auto newOutputShape = outputType.getShape().vec();
      for (TransposeDims dims : transposeIndices) {
        std::swap(newOutputShape[std::get<0>(dims)],
                  newOutputShape[std::get<1>(dims)]);
      }
      auto newOutputType =
          outputType.cloneWith(newOutputShape, outputType.getElementType());
      auto outputTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), newOutputType.getShape(),
          newOutputType.getElementType());

      auto newPool = rewriter.create<PoolOpType>(
          op.getLoc(), newOutputType, input, outputTensor, kernelHeightAttr,
          kernelWidthAttr, strideHeightAttr, strideWidthAttr,
          dilationHeightAttr, dilationWidthAttr, ceilModeAttr, paddingTopAttr,
          paddingBottomAttr, paddingLeftAttr, paddingRightAttr);

      // Applying the transposes in reverse order to the output will restore the
      // tensor to the original layout
      std::reverse(transposeIndices.begin(), transposeIndices.end());
      Value output =
          generateTransposeOps(newPool.getResult(), rewriter, transposeIndices);

      // Reverse back so the proper input transposes are generated for the next
      // pool
      std::reverse(transposeIndices.begin(), transposeIndices.end());
      outputs.push_back(output);
    }

    rewriter.replaceOp(op, outputs);
  }

  uint32_t getNumSpatialDims(ttir::PoolingOp op) const {
    uint32_t numSpatialDims = 0;
    for (int64_t dim : op.getWindowDimensions()) {
      if (dim > 1) {
        numSpatialDims++;
      }
    }
    return numSpatialDims;
  }

  LogicalResult
  matchAndRewrite(ttir::PoolingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t numSpatialDims = getNumSpatialDims(op);
    if (numSpatialDims == 2) {
      if (failed(canDecompose2DPoolingOp(op))) {
        return rewriter.notifyMatchFailure(
            op, "2D pooling op with the given attributes is not supported "
                "currently");
      }

      switch (op.getPoolingMethod()) {
      case ttir::PoolingMethod::Max: {
        rewritePool2d<ttir::MaxPool2dOp>(op, adaptor, rewriter);
        return success();
      }
      default: {
        return rewriter.notifyMatchFailure(
            op, "Failed to match pooling method: " +
                    stringifyPoolingMethod(op.getPoolingMethod()));
      }
      }
    }
    return rewriter.notifyMatchFailure(
        op, "No decompositions for a pooling op with " +
                std::to_string(numSpatialDims) + " spatial dimensions");
  }
};

// SelectOp is converted to a series of SliceOp and potentially a ConcatOp if
// the sliced dimension is sliced multiple times. For example, if the input
// tensor is
//    [[[1, 2, 3],
//      [4, 5, 6],
//      [7, 8, 9],
//      [10, 11, 12],
//      [13, 14, 15],
//      [16, 17, 18]],
//     [[19, 20, 21],
//      [22, 23, 24],
//      [25, 26, 27],
//      [28, 29, 30],
//      [31, 32, 33],
//      [34, 35, 36]]],
//    shape = [2, 6, 3]
// and the SelectOp is dim=1, begin=0, length=2, stride=4, the output tensor
// will be
//    [[[1, 2, 3],
//      [4, 5, 6],
//      [13, 14, 15],
//      [16, 17, 18]],
//     [[19, 20, 21],
//      [22, 23, 24],
//      [31, 32, 33],
//      [34, 35, 36]]],
//    shape = [2, 4, 3]
// In this case 2 slices are created and concatenated to form the output tensor.
// First slice has begins=[0, 0, 0], ends=[2, 2, 3], steps=[1, 1, 1], and the
// second slice has begins=[0, 4, 0], ends=[2, 6, 3], steps=[1, 1, 1].
struct SelectToSliceConversionPattern
    : public OpConversionPattern<ttir::SelectOp> {
public:
  using OpConversionPattern<ttir::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    auto inputShape = inputType.getShape();

    int32_t dim =
        op.getDim() < 0 ? inputType.getRank() + op.getDim() : op.getDim();

    int32_t begin = op.getBegin();
    int32_t length = op.getLength();
    int32_t stride = op.getStride();

    int32_t inputDimSize = inputType.getShape()[dim];
    int32_t numSlices = (inputDimSize - begin + stride - 1) / stride;

    llvm::SmallVector<int32_t, 4> begins, ends, steps;
    for (int32_t i = 0; i < inputType.getRank(); ++i) {
      // Always slicing with step 1.
      steps.push_back(1);
      if (i == dim) {
        // Push placeholder values for now which will be updated later.
        begins.push_back(0);
        ends.push_back(0);
        continue;
      }

      // For non-sliced dimensions, begin=0, end=dimSize, step=1.
      begins.push_back(0);
      ends.push_back(inputType.getDimSize(i));
    }

    // Create a slice for each slice of the input tensor. The slices are then
    // concatenated. The slices are created by updating the begin and end values
    // for the sliced dimension.
    llvm::SmallVector<Value> slices;
    for (int32_t i = 0; i < numSlices; ++i) {
      int32_t newBegin = begin + i * stride;
      int32_t newEnd = std::min(newBegin + length, inputDimSize);

      // Make a copy of the input shape and update the dim size.
      llvm::SmallVector<int64_t> resultShape(inputShape);
      resultShape[dim] = newEnd - newBegin;
      auto resultType =
          RankedTensorType::get(resultShape, inputType.getElementType());

      auto sliceDpsResult = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), resultShape, inputType.getElementType());

      begins[dim] = newBegin;
      ends[dim] = newEnd;

      auto newOp = rewriter.create<ttir::SliceOp>(
          op.getLoc(), resultType, adaptor.getInput(), sliceDpsResult,
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
          rewriter.getI32ArrayAttr(steps));
      slices.push_back(newOp->getResult(0));
    }

    assert(!slices.empty());
    if (slices.size() > 1) {
      auto concatDpsResult = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), outputType.getShape(), outputType.getElementType());

      auto concatOp = rewriter.create<ttir::ConcatOp>(
          op.getLoc(), outputType, slices, concatDpsResult,
          rewriter.getSI32IntegerAttr(dim));

      rewriter.replaceOp(op, concatOp.getResult());
    } else {
      rewriter.replaceOp(op, slices[0]);
    }

    return success();
  }
};

/*
 * This pattern rewrites ArangeOp by forcing the arange_dimension to be
 * rightmost dimension of the output tensor. This is done by replacing the
 * ArangeOp with a new one that has this property, and then transposing out last
 * dimension to the dimension specified by the original ArangeOp, and also
 * inserting a reshape to match the rank of the intended output and broadcasts
 * to repeat the data along the other dimensions.
 *
 * The ArangeOp that is generated here will be equivalent to how ttnn::ArangeOp
 * behaves. The reason this pass is done in TTIR rather than generated when we
 * want to lower to TTNN is because in the future we will want to consteval the
 * ArangeOp, but have the option to not include repeated data in the constant
 * tensor and broadcast at runtime instead. Consteval will be implemented for
 * the TTIR dialect only and so this explication of the TMs implicit in ArangeOp
 * must be done in TTIR.
 */
struct ArangeForceLastDimensionPattern
    : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());

    int64_t arangeDimension = adaptor.getArangeDimension();
    int64_t arangeDimensionNegative = arangeDimension - outputType.getRank();
    int64_t start = adaptor.getStart();
    int64_t end = adaptor.getEnd();
    int64_t step = adaptor.getStep();

    int64_t arangeLength = (end - start) / step;

    const llvm::SmallVector<int64_t, 4> requiredShape{1, 1, 1, arangeLength};
    ArrayRef<int64_t> ttnnShape(requiredShape);
    if (ttnnShape == outputType.getShape()) {
      return success();
    }

    RankedTensorType arangeOutputType = RankedTensorType::get(
        requiredShape, outputType.getElementType(), outputType.getEncoding());

    Value output =
        rewriter
            .create<ttir::ArangeOp>( // perform arange on the last dimension to
                                     // match how ttnn behaves
                op.getLoc(), arangeOutputType, start, end, step, 3)
            .getResult();

    std::vector<int64_t> outputShape = arangeOutputType.getShape().vec();
    // Must transpose the output so that the data changes along the axis defined
    // by arangeDimension
    if (arangeDimensionNegative != -1) {
      std::vector<int64_t> transposeShape = outputShape;
      transposeShape[arangeDimensionNegative + transposeShape.size()] =
          arangeLength;
      transposeShape[arangeOutputType.getRank() - 1] = 1;
      RankedTensorType transposeType = RankedTensorType::get(
          transposeShape, arangeOutputType.getElementType(),
          arangeOutputType.getEncoding());

      tensor::EmptyOp dpsOutput = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), transposeShape, transposeType.getElementType());

      output = rewriter.create<ttir::TransposeOp>(
          op.getLoc(), transposeType, output, dpsOutput,
          arangeDimensionNegative + transposeShape.size(),
          arangeOutputType.getRank() - 1);

      outputShape = transposeShape;
    }

    // Must match up the rank of the output with the rank of the intended output
    // from the original arange, with the arangeDimension in the correct
    // position
    if (outputType.getRank() != static_cast<int64_t>(outputShape.size())) {
      std::vector<int32_t> reshapeShape;
      for (uint32_t i = 0; i < outputType.getRank(); i++) {
        i == arangeDimension ? reshapeShape.push_back(end)
                             : reshapeShape.push_back(1);
      }

      RankedTensorType reshapeType = RankedTensorType::get(
          SmallVector<int64_t>(reshapeShape.begin(), reshapeShape.end()),
          outputType.getElementType(), outputType.getEncoding());
      tensor::EmptyOp dpsOutput = rewriter.create<tensor::EmptyOp>(
          op.getLoc(),
          SmallVector<int64_t>(reshapeShape.begin(), reshapeShape.end()),
          reshapeType.getElementType());
      output = rewriter.create<ttir::ReshapeOp>(
          op.getLoc(), reshapeType, output, dpsOutput,
          rewriter.getI32ArrayAttr(reshapeShape));

      outputShape =
          std::vector<int64_t>(reshapeShape.begin(), reshapeShape.end());
    }

    // Must broadcast the rest of the dimensions
    SmallVector<Attribute> broadcastDims;
    for (uint32_t i = 0; i < outputShape.size(); i++) {
      if (i != arangeDimension && outputShape[i] != outputType.getShape()[i]) {
        outputShape[i] = outputType.getShape()[i];
        broadcastDims.push_back(rewriter.getI64IntegerAttr(i));
      }
    }
    if (!broadcastDims.empty()) {
      RankedTensorType broadcastType = RankedTensorType::get(
          outputShape, outputType.getElementType(), outputType.getEncoding());

      tensor::EmptyOp dpsOutput = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), outputShape, outputType.getElementType());

      output = rewriter.create<ttir::BroadcastOp>(
          op.getLoc(), broadcastType, output, dpsOutput,
          rewriter.getArrayAttr(broadcastDims));

      assert(mlir::cast<RankedTensorType>(output.getType()).getShape() ==
                 outputType.getShape() &&
             "Output shape must match the shape of the input tensor");
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<PoolingToPool2dPattern>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<Legalize1DConvolutionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<GatherToEmbeddingConversionPattern>(typeConverter, ctx);
  patterns.add<SelectToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ArangeForceLastDimensionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
