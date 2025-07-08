// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::tt;

mlir::DenseI64ArrayAttr toDenseI64ArrayAttr(llvm::ArrayRef<int32_t> input,
                                            mlir::Builder &builder) {
  SmallVector<int64_t> values;
  values.reserve(input.size());
  for (int32_t v : input) {
    values.push_back(static_cast<int64_t>(v));
  }
  return builder.getDenseI64ArrayAttr(values);
}

namespace {
class Conv2dOpConversionPattern : public OpConversionPattern<ttnn::Conv2dOp> {
public:
  using OpConversionPattern<ttnn::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // let arguments = (ins AnyRankedTensor:$input,
    //   AnyRankedTensor:$weight,
    //   Optional<AnyRankedTensor>:$bias,
    //   AnyRankedTensor:$output,
    //   AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
    //   AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
    //   AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
    //   I32Attr:$groups,
    //   DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr,
    //   "nullptr">:$flattened_compat_info);

    // nhwc

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    uint64_t numSpatialDims = 4;

    // // These are the defaults intended by stablehlo when the attrs are not
    // // populated
    DenseI64ArrayAttr windowStridesAttr =
        adaptor.getStrideAttr()
            ? toDenseI64ArrayAttr(adaptor.getStride(), rewriter)
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    SmallVector<int64_t, 4> padding = {
        adaptor.getPadding()[0], adaptor.getPadding()[1],
        adaptor.getPadding()[0], adaptor.getPadding()[1]};
    DenseI64ArrayAttr paddingAttr = rewriter.getDenseI64ArrayAttr(padding);
    DenseI64ArrayAttr inputDilationAttr =
        rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(numSpatialDims, 1));
    DenseI64ArrayAttr kernelDilationAttr =
        adaptor.getDilationAttr()
            ? toDenseI64ArrayAttr(adaptor.getDilation(), rewriter)
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseBoolArrayAttr windowReversalAttr = rewriter.getDenseBoolArrayAttr(
        SmallVector<bool>(numSpatialDims, false));

    // let arguments = (ins
    //   AnyRankedTensor:$input,
    //   AnyRankedTensor:$weight,
    //   Optional<AnyRankedTensor>:$bias,
    //   AnyRankedTensor:$output,
    //   DenseI64ArrayAttr:$window_strides,
    //   DenseI64ArrayAttr:$padding,
    //   DenseI64ArrayAttr:$input_dilation,
    //   DenseI64ArrayAttr:$weight_dilation,
    //   DenseBoolArrayAttr:$window_reversal,
    //   TTIR_ConvolutionLayoutAttr:$convolution_layout,
    //   ConfinedAttr<I64Attr, [IntPositive]>:$feature_group_count,
    //   ConfinedAttr<I64Attr, [IntPositive]>:$batch_group_count
    // );

    // let parameters = (ins
    //   "int64_t":$inputBatchDimension,
    //   "int64_t":$inputFeatureDimension,
    //   ArrayRefParameter<"int64_t">:$inputSpatialDimensions,
    //
    //   "int64_t":$kernelOutputFeatureDimension,
    //   "int64_t":$kernelInputFeatureDimension,
    //   ArrayRefParameter<"int64_t">:$kernelSpatialDimensions,
    //
    //   "int64_t":$outputBatchDimension,
    //   "int64_t":$outputFeatureDimension,
    //   ArrayRefParameter<"int64_t">:$outputSpatialDimensions
    // );

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ConvolutionOp>(
        rewriter, op, outputType, adaptor.getInput(), adaptor.getWeight(),
        Value(), windowStridesAttr, paddingAttr, inputDilationAttr,
        kernelDilationAttr, windowReversalAttr,
        mlir::tt::ttir::ConvolutionLayoutAttr::get(
            getContext(), 0, 3, llvm::SmallVector<int64_t, 2>{1, 2}, 0, 1,
            llvm::SmallVector<int64_t, 2>{2, 3}, 0, 3,
            llvm::SmallVector<int64_t, 2>{1, 2}),
        mlir::IntegerAttr::get(rewriter.getI64Type(), 1),
        mlir::IntegerAttr::get(rewriter.getI64Type(), 1));

    // rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
template <typename TTNNOpTy>
class PoolingOpConversionPattern : public OpConversionPattern<TTNNOpTy> {
public:
  using OpConversionPattern<TTNNOpTy>::OpConversionPattern;
  using OpAdaptor = typename TTNNOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(TTNNOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    uint64_t numSpatialDims = 4;

    SmallVector<int64_t, 4> kernelSize = {adaptor.getKernelSize()[0],
                                          adaptor.getKernelSize()[1]};
    SmallVector<int64_t, 4> stride = {
        adaptor.getStride()[0], adaptor.getStride()[1], adaptor.getStride()[0],
        adaptor.getStride()[1]};
    SmallVector<int64_t, 4> kernelDilations;
    if (adaptor.getDilation().size() == 2) {
      kernelDilations = {adaptor.getDilation()[0], adaptor.getDilation()[1],
                         adaptor.getDilation()[0], adaptor.getDilation()[1]};
    } else if (adaptor.getDilation().size() == 1) {
      kernelDilations = {adaptor.getDilation()[0], adaptor.getDilation()[0],
                         adaptor.getDilation()[0], adaptor.getDilation()[0]};
    } else {
      return failure();
    }
    SmallVector<int64_t, 4> padding = {
        adaptor.getPadding()[0], adaptor.getPadding()[0],
        adaptor.getPadding()[1], adaptor.getPadding()[1],
        adaptor.getPadding()[0], adaptor.getPadding()[0],
        adaptor.getPadding()[1], adaptor.getPadding()[1]};
    DenseI64ArrayAttr kernelSizeAttr =
        rewriter.getDenseI64ArrayAttr(kernelSize);
    DenseI64ArrayAttr strideAttr = rewriter.getDenseI64ArrayAttr(stride);
    DenseI64ArrayAttr kernelDilationAttr =
        rewriter.getDenseI64ArrayAttr(kernelDilations);
    DenseI64ArrayAttr paddingAttr = rewriter.getDenseI64ArrayAttr(padding);
    DenseI64ArrayAttr inputDilationAttr =
        rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(numSpatialDims, 1));

    mlir::tt::ttir::PoolingMethod method;
    if constexpr (std::is_same<TTNNOpTy, ttnn::MaxPool2dOp>::value) {
      method = mlir::tt::ttir::PoolingMethod::Max;
    } else if constexpr (std::is_same<TTNNOpTy, ttnn::AvgPool2dOp>::value) {
      method = mlir::tt::ttir::PoolingMethod::Average;
    } else {
      static_assert(false, "Unsupported TTNNOpTy for pooling conversion");
    }

    auto methodAttr =
        mlir::tt::ttir::PoolingMethodAttr::get(rewriter.getContext(), method);

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::PoolingOp>(
        rewriter, srcOp, outputType, adaptor.getInput(), methodAttr,
        kernelSizeAttr, strideAttr, inputDilationAttr, kernelDilationAttr,
        paddingAttr);

    return success();
  }
};
} // namespace

namespace {
class MaxPool2dOpConversionPattern
    : public OpConversionPattern<ttnn::MaxPool2dOp> {
public:
  using OpConversionPattern<ttnn::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::MaxPool2dOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // let arguments = (ins AnyRankedTensor:$input,
    //   AnyRankedTensor:$output,
    //   SI32Attr:$kernel_height,
    //   SI32Attr:$kernel_width,
    //   SI32Attr:$stride_height,
    //   SI32Attr:$stride_width,
    //   SI32Attr:$dilation_height,
    //   SI32Attr:$dilation_width,
    //   BoolAttr:$ceil_mode,
    //   SI32Attr:$padding_left,
    //   SI32Attr:$padding_right,
    //   SI32Attr:$padding_top,
    //   SI32Attr:$padding_bottom,
    //   DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr,
    //   "nullptr">:$flattened_compat_info);

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::MaxPool2dOp>(
        rewriter, srcOp, outputType, adaptor.getInput(),
        adaptor.getKernelSize()[0], adaptor.getKernelSize()[1],
        adaptor.getStride()[0], adaptor.getStride()[1],
        adaptor.getDilation()[0], adaptor.getDilation()[1],
        adaptor.getCeilMode(), adaptor.getPadding()[0], adaptor.getPadding()[0],
        adaptor.getPadding()[1], adaptor.getPadding()[1], nullptr);

    return success();
  }
};
} // namespace

namespace {
class ReluOpConversionPattern : public OpConversionPattern<ttnn::ReluOp> {
public:
  using OpConversionPattern<ttnn::ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ReluOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ReluOp>(
        rewriter, srcOp, outputType, adaptor.getInput());

    return success();
  }
};
} // namespace

namespace {
class AddOpConversionPattern : public OpConversionPattern<ttnn::AddOp> {
public:
  using OpConversionPattern<ttnn::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::AddOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::AddOp>(
        rewriter, srcOp, outputType, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};
} // namespace

namespace {
template <typename TTNNOpTy, typename TTIROpTy>
class TTNNToTTIRNamedFullConversionPattern
    : public OpConversionPattern<TTNNOpTy> {

  using OpConversionPattern<TTNNOpTy>::OpConversionPattern;
  using OpAdaptor = typename ttnn::OnesOp::Adaptor;

public:
  LogicalResult
  matchAndRewrite(TTNNOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // let arguments = (ins DenseI32ArrayAttr:$shape);

    // let results = (outs AnyRankedTensor:$result);

    auto convertedType =
        this->getTypeConverter()->convertType(srcOp.getResult().getType());
    auto outputType = cast<mlir::RankedTensorType>(convertedType);

    std::vector<int32_t> shp =
        std::vector<int32_t>(adaptor.getShapeAttr().getShape().begin(),
                             adaptor.getShapeAttr().getShape().end());

    rewriter.replaceOpWithNewOp<TTIROpTy>(srcOp, outputType, shp);

    return success();
  }
};
} // namespace

namespace {
class DeviceOpConversionPattern : public OpConversionPattern<ttcore::DeviceOp> {
public:
  using OpConversionPattern<ttcore::DeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttcore::DeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
class GetDeviceOpConversionPattern
    : public OpConversionPattern<ttnn::GetDeviceOp> {
public:
  using OpConversionPattern<ttnn::GetDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::GetDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace mlir::tt {

void populateTTNNToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // clang-format off
  patterns.add<
           Conv2dOpConversionPattern,
          //  PoolingOpConversionPattern<ttnn::MaxPool2dOp>,
           PoolingOpConversionPattern<ttnn::AvgPool2dOp>,
           MaxPool2dOpConversionPattern,
           ReluOpConversionPattern,
           AddOpConversionPattern,
           TTNNToTTIRNamedFullConversionPattern<ttnn::OnesOp, ttir::OnesOp>,
           DeviceOpConversionPattern,
           GetDeviceOpConversionPattern
           >(typeConverter, ctx);
  // clang-format on
}

} // namespace mlir::tt
