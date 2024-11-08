// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

//===----------------------------------------------------------------------===//
// IndexOp decomposition
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_index_ttir
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
    if (!inputType || !inputType.hasRank())
      return failure();

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
// ANCHOR_END: adding_an_op_index_ttir

//===----------------------------------------------------------------------===//
// Convolution passes
//===----------------------------------------------------------------------===//

using TransposeDims = std::tuple<int64_t, int64_t>;

template <uint32_t NDims>
using PaddingMatrix = std::array<std::array<int64_t, 2>, NDims>;

template <uint32_t NDims>
static PaddingMatrix<NDims> getPaddingMatrix(DenseIntElementsAttr paddingAttr) {
  PaddingMatrix<NDims> paddingMatrix;
  std::vector<int64_t> paddingFlattened(paddingAttr.value_begin<int64_t>(),
                                        paddingAttr.value_end<int64_t>());

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

static tensor::EmptyOp generateTransposeDPSOutput(Value input, int64_t dim0,
                                                  int64_t dim1,
                                                  PatternRewriter &rewriter) {
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto output_shape = input_type.getShape().vec();
  std::swap(output_shape[dim0], output_shape[dim1]);

  auto output_type = RankedTensorType::get(
      output_shape, input_type.getElementType(), input_type.getEncoding());

  return rewriter.create<tensor::EmptyOp>(input.getLoc(), output_shape,
                                          output_type.getElementType());
}

static ttir::TransposeOp
generateTranspose(Value input, int64_t dim0, int64_t dim1,
                  PatternRewriter &rewriter,
                  ::mlir::ArrayAttr operandConstraints) {
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto output_shape = input_type.getShape().vec();
  std::swap(output_shape[dim0], output_shape[dim1]);

  auto dim0_attr = rewriter.getSI32IntegerAttr(dim0);
  auto dim1_attr = rewriter.getSI32IntegerAttr(dim1);

  auto dps_output = generateTransposeDPSOutput(input, dim0, dim1, rewriter);
  return rewriter.create<ttir::TransposeOp>(
      input.getLoc(), dps_output.getType(), input, dps_output, dim0_attr,
      dim1_attr, operandConstraints);
}

static std::vector<TransposeDims> generateKernelTransposeIndices(
    ttir::ConvolutionOp op,
    const std::vector<int64_t> ttnn_convolution_kernel_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> kernel_layout(
      ttnn_convolution_kernel_layout.size(),
      ConvolutionKernelDimension::INVALID_KERNEL_DIM);
  kernel_layout[op.getConvolutionLayout().getKernelOutputFeatureDimension()] =
      ConvolutionKernelDimension::OUTPUT_FEATURES;
  kernel_layout[op.getConvolutionLayout().getKernelInputFeatureDimension()] =
      ConvolutionKernelDimension::INPUT_FEATURES;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getKernelSpatialDimensions()) {
    kernel_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  const std::vector<int64_t> desired_kernel_layout =
      ttnn_convolution_kernel_layout;
  for (int64_t i = 0; i < static_cast<int64_t>(kernel_layout.size()); i++) {
    if (kernel_layout[i] != desired_kernel_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(kernel_layout.begin(), kernel_layout.end(),
                               desired_kernel_layout[i]) -
                     kernel_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(kernel_layout[dim0], kernel_layout[dim1]);
    }
  }

  return transpose_indices;
}

static std::vector<TransposeDims> generateInputTransposeIndices(
    ttir::ConvolutionOp op,
    const std::vector<int64_t> ttnn_convolution_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> input_layout(ttnn_convolution_layout.size(),
                                    ConvolutionDimension::INVALID_DIM);
  input_layout[op.getConvolutionLayout().getInputBatchDimension()] =
      ConvolutionDimension::BATCH;
  input_layout[op.getConvolutionLayout().getInputFeatureDimension()] =
      ConvolutionDimension::FEATURE;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getInputSpatialDimensions()) {
    input_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  const std::vector<int64_t> desired_input_layout = ttnn_convolution_layout;
  for (int64_t i = 0; i < static_cast<int64_t>(input_layout.size()); i++) {
    if (input_layout[i] != desired_input_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(input_layout.begin(), input_layout.end(),
                               desired_input_layout[i]) -
                     input_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(input_layout[dim0], input_layout[dim1]);
    }
  }

  return transpose_indices;
}

/**
 * Although this function is mostly a clone of generateInputTransposeIndices,
 * its slightly different in that if the original Convolution op had the same
 * input and output layout, this function will generate the same transposes,
 * that were applied to the input but in reverse order. This makes optimizing
 * away the inserted transposes easier.
 */
static std::vector<TransposeDims> generateOutputTransposeIndices(
    ttir::ConvolutionOp op,
    const std::vector<int64_t> ttnn_convolution_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> desired_output_layout(ttnn_convolution_layout.size(),
                                             ConvolutionDimension::INVALID_DIM);
  desired_output_layout[op.getConvolutionLayout().getOutputBatchDimension()] =
      ConvolutionDimension::BATCH;
  desired_output_layout[op.getConvolutionLayout().getOutputFeatureDimension()] =
      ConvolutionDimension::FEATURE;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getOutputSpatialDimensions()) {
    desired_output_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  std::vector<int64_t> output_layout = ttnn_convolution_layout;

  for (int64_t i = static_cast<int64_t>(desired_output_layout.size()) - 1;
       i >= 0; i--) {
    if (desired_output_layout[i] != output_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(output_layout.begin(), output_layout.end(),
                               desired_output_layout[i]) -
                     output_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(output_layout[dim0], output_layout[dim1]);
    }
  }

  return transpose_indices;
}

static Value
generateTransposeSequence(Value input, PatternRewriter &rewriter,
                          std::vector<TransposeDims> transpose_indices,
                          ::mlir::ArrayAttr operandConstraints) {
  for (auto [dim0, dim1] : transpose_indices) {
    input = generateTranspose(input, dim0, dim1, rewriter, operandConstraints)
                .getResult();
  }

  return input;
}

struct ConvolutionToConv2dPattern
    : public OpConversionPattern<ttir::ConvolutionOp> {
public:
  using OpConversionPattern<ttir::ConvolutionOp>::OpConversionPattern;

  constexpr static uint32_t numSpatialDims = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC
  const std::vector<int64_t> conv2d_layout = {
      ConvolutionDimension::BATCH, SPATIAL_DIM_HEIGHT, SPATIAL_DIM_WIDTH,
      ConvolutionDimension::FEATURE};
  // OIHW
  const std::vector<int64_t> conv2d_kernel_layout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES, SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH};
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
    std::vector<bool> window_reversal(op.getWindowReversal().begin(),
                                      op.getWindowReversal().end());
    for (bool reversed : window_reversal) {
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

    auto stride_height_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_HEIGHT]);
    auto stride_width_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_WIDTH]);
    auto dilation_height_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT]);
    auto dilation_width_attr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH]);

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order For Conv2d the most
    // significant spatial dimension is the height, followed by the width.
    auto padding_matrix =
        getPaddingMatrix<numSpatialDims>(adaptor.getPadding());
    auto padding_top_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_HEIGHT][0]);
    auto padding_bottom_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_HEIGHT][1]);
    auto padding_left_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_WIDTH][0]);
    auto padding_right_attr =
        rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_WIDTH][1]);

    auto groups_attr =
        rewriter.getSI32IntegerAttr(adaptor.getFeatureGroupCount());

    auto output_shape = op.getResult().getType().getShape().vec();
    std::vector<int64_t> new_output_shape = {
        output_shape[adaptor.getConvolutionLayout().getOutputBatchDimension()],
        output_shape[adaptor.getConvolutionLayout()
                         .getOutputSpatialDimensions()[SPATIAL_DIM_HEIGHT]],
        output_shape[adaptor.getConvolutionLayout()
                         .getOutputSpatialDimensions()[SPATIAL_DIM_WIDTH]],
        output_shape[adaptor.getConvolutionLayout()
                         .getOutputFeatureDimension()]};

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType =
        inputType.cloneWith(new_output_shape, inputType.getElementType());

    auto convDPSOutput = rewriter.create<tensor::EmptyOp>(
        adaptor.getInput().getLoc(), new_output_shape,
        outputType.getElementType());

    auto input_transpose_indices =
        generateInputTransposeIndices(op, conv2d_layout);
    Value input = generateTransposeSequence(adaptor.getInput(), rewriter,
                                            input_transpose_indices,
                                            adaptor.getOperandConstraints());

    auto kernel_transpose_indices =
        generateKernelTransposeIndices(op, conv2d_kernel_layout);
    Value weight = generateTransposeSequence(adaptor.getWeight(), rewriter,
                                             kernel_transpose_indices,
                                             adaptor.getOperandConstraints());
    ttir::Conv2dOp new_conv = rewriter.create<ttir::Conv2dOp>(
        op.getLoc(), outputType, input, weight, adaptor.getBias(),
        convDPSOutput, stride_height_attr, stride_width_attr,
        dilation_height_attr, dilation_width_attr, groups_attr,
        padding_left_attr, padding_right_attr, padding_top_attr,
        padding_bottom_attr, adaptor.getOperandConstraints());

    auto output_transpose_indices =
        generateOutputTransposeIndices(op, conv2d_layout);
    Value output = generateTransposeSequence(new_conv.getResult(), rewriter,
                                             output_transpose_indices,
                                             adaptor.getOperandConstraints());

    rewriter.replaceOp(op, output);

    return success();
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

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<GetDimensionSizeToConstantConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
