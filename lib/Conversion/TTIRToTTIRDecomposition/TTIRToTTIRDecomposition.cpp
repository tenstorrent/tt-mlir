// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

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
        rewriter.getArrayAttr(steps), adaptor.getOperandConstraints());

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
                                  std::vector<TransposeDims> transposeIndices,
                                  ::mlir::ArrayAttr operandConstraints) {
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
                                           dpsOutput, dim0Attr, dim1Attr,
                                           operandConstraints)
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

struct ConvolutionToConv2dPattern
    : public OpConversionPattern<ttir::ConvolutionOp> {
public:
  using OpConversionPattern<ttir::ConvolutionOp>::OpConversionPattern;

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

  LogicalResult isConv2d(ttir::ConvolutionOp op) const {

    // Conv2d will have 2 spatial dimensions

    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getOutputSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getKernelSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");

    if (op.getConvolutionLayout().getInputSpatialDimensions().size() !=
        numSpatialDims) {
      return failure();
    }

    // Not currently supporting window reversal
    std::vector<bool> windowReversal(op.getWindowReversal().begin(),
                                     op.getWindowReversal().end());
    for (bool reversed : windowReversal) {
      if (reversed) {
        return failure();
      }
    }

    // Not currently support batch groups
    if (op.getBatchGroupCount() != 1) {
      return failure();
    }

    return success();
  }

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(isConv2d(op))) {
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
        generateTransposeOps(adaptor.getInput(), rewriter, transposeIndices,
                             adaptor.getOperandConstraints());

    auto kernelTransposeIndices =
        generateConvKernelTransposeIndices(op, conv2dKernelLayout);
    Value weight = generateTransposeOps(adaptor.getWeight(), rewriter,
                                        kernelTransposeIndices,
                                        adaptor.getOperandConstraints());
    ttir::Conv2dOp newConv = rewriter.create<ttir::Conv2dOp>(
        op.getLoc(), outputType, input, weight, adaptor.getBias(),
        convDPSOutput, strideHeightAttr, strideWidthAttr, dilationHeightAttr,
        dilationWidthAttr, groupsAttr, paddingLeftAttr, paddingRightAttr,
        paddingTopAttr, paddingBottomAttr, adaptor.getOperandConstraints());

    // Applying the transposes in reverse order to the output will restore the
    // tensor to the original layout
    std::reverse(transposeIndices.begin(), transposeIndices.end());
    Value output =
        generateTransposeOps(newConv.getResult(), rewriter, transposeIndices,
                             adaptor.getOperandConstraints());

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
                                  ::llvm::ArrayRef<int64_t> shapei64,
                                  ::mlir::ArrayAttr operandConstraints) const {

    // reshape start indices (input) to remove the last dimension
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>(shapei64), ty.getElementType());
    std::vector<int32_t> shapei32(shapei64.begin(), shapei64.end());
    auto shape_attr = rewriter.getI32ArrayAttr(shapei32);

    return rewriter.create<ttir::ReshapeOp>(
        loc, mlir::RankedTensorType::get(shapei64, ty.getElementType()), input,
        output, shape_attr, operandConstraints);
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
          createReshapeOp(rewriter, op.getLoc(), startIndices, newShapeI64,
                          op.getOperandConstraints());

      assert(reshapeOp && "Failed to create reshape op");
      reshapeOp->moveBefore(op);
      input = reshapeOp.getResult();
    }

    // convert gather to embedding, use reshaped input if needed
    ttir::EmbeddingOp embeddingOp = rewriter.create<ttir::EmbeddingOp>(
        op.getLoc(), op.getResult().getType(), input, op.getOperands()[0],
        op.getOutput(),
        rewriter.getArrayAttr(SmallVector<Attribute>(
            op.getNumOperands() + 1, rewriter.getAttr<OperandConstraintAttr>(
                                         OperandConstraint::AnyDeviceTile))));

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
    auto operandConstraints = adaptor.getOperandConstraints();

    std::vector<Value> outputs;
    for (Value input : adaptor.getInputs()) {
      input = generateTransposeOps(input, rewriter, transposeIndices,
                                   operandConstraints);

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
          paddingBottomAttr, paddingLeftAttr, paddingRightAttr,
          operandConstraints);

      // Applying the transposes in reverse order to the output will restore the
      // tensor to the original layout
      std::reverse(transposeIndices.begin(), transposeIndices.end());
      Value output = generateTransposeOps(newPool.getResult(), rewriter,
                                          transposeIndices, operandConstraints);

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

class GetDimensionSizeToConstantConversionPattern
    : public OpConversionPattern<ttir::GetDimensionSizeOp> {
public:
  using OpConversionPattern<ttir::GetDimensionSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GetDimensionSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const RankedTensorType inputTensorType =
        mlir::cast<RankedTensorType>(op.getOperand().getType());

    int64_t dimensionIndex = op.getDimension();

    int32_t dimSize = inputTensorType.getShape()[dimensionIndex];

    mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(op.getType());

    mlir::ElementsAttr valueAttr =
        mlir::DenseElementsAttr::get<int>(valueType, dimSize);

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(op, valueType,
                                                            valueAttr);

    return success();
  }
};

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
    int32_t numSlices = (inputDimSize - begin) / stride;

    llvm::SmallVector<int32_t, 4> begins, ends, steps;
    for (int32_t i = 0; i < inputType.getRank(); ++i) {
      if (i == dim) {
        // Push placeholder values for now which will be updated later.
        begins.push_back(0);
        ends.push_back(0);
        steps.push_back(0);
        continue;
      }

      // For non-sliced dimensions, begin=0, end=dimSize, step=1.
      begins.push_back(0);
      ends.push_back(inputType.getDimSize(i));
      steps.push_back(1);
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
      steps[dim] = 1;

      auto newOp = rewriter.create<ttir::SliceOp>(
          op.getLoc(), resultType, adaptor.getInput(), sliceDpsResult,
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
          rewriter.getI32ArrayAttr(steps), adaptor.getOperandConstraints());
      slices.push_back(newOp->getResult(0));
    }

    assert(!slices.empty());
    if (slices.size() > 1) {
      auto concatDpsResult = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), outputType.getShape(), outputType.getElementType());
      auto concatOp = rewriter.create<ttir::ConcatOp>(
          op.getLoc(), outputType, slices, concatDpsResult,
          rewriter.getSI32IntegerAttr(dim), adaptor.getOperandConstraints());

      rewriter.replaceOp(op, concatOp.getResult());
    } else {
      rewriter.replaceOp(op, slices[0]);
    }

    return success();
  }
};

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<PoolingToPool2dPattern>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<GetDimensionSizeToConstantConversionPattern>(typeConverter, ctx);
  patterns.add<GatherToEmbeddingConversionPattern>(typeConverter, ctx);
  patterns.add<SelectToSliceConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
