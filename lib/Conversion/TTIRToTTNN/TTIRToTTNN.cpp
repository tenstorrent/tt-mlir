// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

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

using namespace mlir;
using namespace mlir::tt;

namespace {

// Gets or inserts a GetDeviceOp at the top of the current block of the given
// operation.
static Value getOrInsertDevice(ConversionPatternRewriter &rewriter,
                               Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp.getResult();
    }
  }

  DeviceAttr deviceAttr = getCurrentScopeDevice(op);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  auto deviceOp = rewriter.create<ttnn::GetDeviceOp>(
      op->getLoc(), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), 1, 1));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp.getResult();
}

class TensorEmptyConversionPattern
    : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get tt::LayoutAttr of the result type
    //
    tt::LayoutAttr ttLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getResult().getType().getEncoding());

    // Get the shape of the tensor, tensor layout, and data type
    //
    mlir::MemRefType memref = ttLayoutAttr.getMemref();
    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(),
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape());
    Type elementType = memref.getElementType();
    DataType dtype = DataType::Float32;
    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
    if (llvm::isa<TileType>(elementType)) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
      auto tileType = mlir::cast<TileType>(elementType);
      dtype = tileType.getDataType();
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
      dtype = elementTypeToDataType(elementType);
    }
    DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    // If the tensor is not going to device, we can create the op without
    // device-specific attributes
    //
    tt::TensorMemoryLayout ttTensorMemoryLayout = ttLayoutAttr.getMemLayout();
    if (ttTensorMemoryLayout == TensorMemoryLayout::None) {
      rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
          op, this->getTypeConverter()->convertType(op.getType()), nullptr,
          shapeAttr, dTypeAttr, tensorLayoutAttr, nullptr);

      return success();
    }

    ttnn::BufferType bufferType =
        ttnn::utils::toTTNNBufferType(ttLayoutAttr.getMemorySpace());
    ttnn::TensorMemoryLayout tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(ttLayoutAttr.getMemLayout());

    // Create MemoryConfigAttr
    //
    auto device = getOrInsertDevice(rewriter, op);
    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        op.getContext(),
        ttnn::TensorMemoryLayoutAttr::get(op.getContext(), tensorMemoryLayout),
        ttnn::BufferTypeAttr::get(op.getContext(), bufferType));

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
        op, this->getTypeConverter()->convertType(op.getType()), device,
        shapeAttr, dTypeAttr, tensorLayoutAttr, memoryConfigAttr);

    return success();
  }
};

// TTIR::ToLayoutOp is a rather generic op that dictates how all the layout
// properties of a tensor should be set. However, in TTNN world, multiple APIs
// are required to achieve an arbitrary layout. There are two main distinct
// paths in this conversion pattern:
//
// 1. If the layout calls for device memory, we will call TTNN::ToLayoutOp and
//    TTNN::ToDeviceOp to achieve the desired layout.
//
// 2. If the layout calls for system memory, we will call TTNN::ToLayoutOp to
//    change the tensor to RowMajor layout, and then the TTNN::FromDeviceOp to
//    move to host memory
//
class ToLayoutOpConversionPattern
    : public OpConversionPattern<ttir::ToLayoutOp> {
public:
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;

  bool shouldForceRowMajor(ttir::ToLayoutOp op) const {
    for (mlir::Operation *user : op.getResult().getUsers()) {
      if (isa<ttir::Conv2dOp>(user) || isa<ttir::MaxPool2dOp>(user)) {
        return true;
      }
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Find device to be used for the tensor
    //
    auto device = getOrInsertDevice(rewriter, op);

    // Get tt::LayoutAttr of the result type
    //
    tt::LayoutAttr ttLayoutAttr =
        mlir::cast<tt::LayoutAttr>(op.getResult().getType().getEncoding());

    // Figure out if output tensor is in RowMajor layout or Tile layout
    // Figure out the data type of the output tensor
    //
    mlir::MemRefType memref = ttLayoutAttr.getMemref();
    Type elementType = memref.getElementType();
    DataType dtype = DataType::Float32;
    // TODO(bug #665):
    // Remove attribute once 665 is fixed
    //
    ttnn::Layout ttnnLayoutEnum __attribute__((unused)) =
        ttnn::Layout::RowMajor;
    if (llvm::isa<TileType>(elementType)) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
      auto tileType = mlir::cast<TileType>(elementType);
      dtype = tileType.getDataType();
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
      dtype = elementTypeToDataType(elementType);
    }

    // TODO(bug #875):
    // Remove the following code block once constraints modelling is implemented
    // on dialect level
    //
    // Default to Tile layout unless op supports only RowMajor layout
    //
    ttnnLayoutEnum =
        shouldForceRowMajor(op) ? ttnn::Layout::RowMajor : ttnn::Layout::Tile;

    // Map TT::MemorySpace to TTNN::BufferType
    //
    ttnn::BufferType bufferType =
        ttnn::utils::toTTNNBufferType(ttLayoutAttr.getMemorySpace());

    // If the ToLayoutOp is applied to empty tensor, we need to check whether
    // the empty tensor is going back to system memory; if so, we should not
    // call the ToDeviceOp
    //
    if (bufferType == ttnn::BufferType::SystemMemory) {
      rewriter.replaceOpWithNewOp<ttnn::ToMemoryConfigOp>(
          op, this->getTypeConverter()->convertType(op.getType()),
          op.getInput(), device);
      return success();
    }

    // Set the tensor memory layout
    //
    ttnn::TensorMemoryLayout tensorMemoryLayout =
        ttnn::utils::toTTNNTensorMemoryLayout(ttLayoutAttr.getMemLayout());

    // TODO(bug #621):
    // Add ttnn::Tensor(tensor, dtype) op call once tt-metal is updated
    //
    // Also update the function header comment to reflect this added op
    //
    (void)dtype;

    // Create ToLayoutOp
    //
    ttnn::ToLayoutOp toLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
        op.getLoc(), this->getTypeConverter()->convertType(op.getType()),
        op.getInput(), device,
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum));

    // Create MemoryConfigAttr
    //
    // TODO(bug #620):
    // Add support for ShardSpec
    //
    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        op.getContext(),
        ttnn::TensorMemoryLayoutAttr::get(op.getContext(), tensorMemoryLayout),
        ttnn::BufferTypeAttr::get(op.getContext(), bufferType));

    // Create ToDeviceOp
    //
    rewriter.replaceOpWithNewOp<ttnn::ToDeviceOp>(
        op, this->getTypeConverter()->convertType(op.getType()), toLayoutOp,
        device, memoryConfigAttr);
    return success();
  }
};

template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TTNNOpTy>(op, resultTypes, adaptor.getInputs(),
                                          adaptor.getOutputs());
    return success();
  }
};

template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ReductionOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getKeepDim(),
        adaptor.getDimArg().value_or(nullptr));
    return success();
  }
};

class EmbeddingOpConversionPattern
    : public OpConversionPattern<ttir::EmbeddingOp> {
public:
  using OpConversionPattern<ttir::EmbeddingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::EmbeddingOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight());

    return success();
  }
};

class SoftmaxOpConversionPattern : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern<ttir::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::SoftmaxOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getDimension());
    return success();
  }
};

class TransposeOpConversionPattern
    : public OpConversionPattern<ttir::TransposeOp> {
public:
  using OpConversionPattern<ttir::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::TransposeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getDim0(), adaptor.getDim1());
    return success();
  }
};

class ConcatOpConversionPattern : public OpConversionPattern<ttir::ConcatOp> {
public:
  using OpConversionPattern<ttir::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int dim = adaptor.getDim();
    if (dim < 0) {
      dim += cast<RankedTensorType>(adaptor.getInputs().front().getType())
                 .getRank();
    }
    rewriter.replaceOpWithNewOp<ttnn::ConcatOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInputs(), adaptor.getOutput(), dim);
    return success();
  }
};

class ReshapeOpConversionPattern : public OpConversionPattern<ttir::ReshapeOp> {
public:
  using OpConversionPattern<ttir::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getShape());
    return success();
  }
};

class SliceOpConversionPattern : public OpConversionPattern<ttir::SliceOp> {
public:
  using OpConversionPattern<ttir::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::SliceOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getOutput(), adaptor.getBegins(),
        adaptor.getEnds(), adaptor.getStep());
    return success();
  }
};

class SqueezeOpConversionPattern : public OpConversionPattern<ttir::SqueezeOp> {
public:
  using OpConversionPattern<ttir::SqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract input tensor type
    ::mlir::RankedTensorType inputType =
        mlir::cast<::mlir::RankedTensorType>(adaptor.getInput().getType());

    // Get the squeeze dimension
    int32_t dim = adaptor.getDim();

    if (dim < 0) {
      dim += inputType.getRank();
    }

    // Get the shape of the input tensor
    auto inputShape = inputType.getShape();
    llvm::SmallVector<int32_t, 4> newShape;

    // Build the new shape by removing the specified dimension
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (i == dim) {
        continue;
      }
      newShape.push_back(inputShape[i]);
    }

    // Create the new shape attribute
    auto shapeAttr = rewriter.getI32ArrayAttr(newShape);

    // Replace the SqueezeOp with a ReshapeOp
    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), shapeAttr);

    return success();
  }
};

class UnsqueezeOpConversionPattern
    : public OpConversionPattern<ttir::UnsqueezeOp> {
public:
  using OpConversionPattern<ttir::UnsqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract input tensor type
    ::mlir::RankedTensorType inputType =
        mlir::cast<::mlir::RankedTensorType>(adaptor.getInput().getType());

    // Get the unsqueeze dimension
    int32_t dim = adaptor.getDim();

    // Convert negative dim to its positive equivalent
    if (dim < 0) {
      dim += inputType.getRank() + 1;
    }

    // Get the shape of the input tensor
    auto inputShape = inputType.getShape();
    llvm::SmallVector<int32_t, 5> newShape;

    // Insert the new dimension
    for (int i = 0; i < inputType.getRank(); ++i) {
      if (i == dim) {
        newShape.push_back(1);
      }
      newShape.push_back(inputShape[i]);
    }

    // Create the new shape attribute
    auto shapeAttr = rewriter.getI32ArrayAttr(newShape);

    // Replace the UnsqueezeOp with a ReshapeOp
    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), shapeAttr);

    return success();
  }
};

class ConstantOpConversionPattern
    : public OpConversionPattern<ttir::ConstantOp> {
public:
  using OpConversionPattern<ttir::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ::mlir::ElementsAttr valueAttr = op.getValue();

    LogicalResult legalityResult = checkBasicLegality(op, valueAttr, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    if (valueAttr.isSplat()) {
      Value device = getOrInsertDevice(rewriter, op);
      float fillValue = valueAttr.getElementType().isInteger()
                            ? static_cast<float>(valueAttr.getSplatValue<int>())
                            : valueAttr.getSplatValue<float>();
      if (fillValue == 0) {
        rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
            op, this->getTypeConverter()->convertType(op.getType()), device);
      } else {
        ::mlir::FloatAttr fillValueAttr = rewriter.getF32FloatAttr(fillValue);
        rewriter.replaceOpWithNewOp<ttnn::FullOp>(
            op, this->getTypeConverter()->convertType(op.getType()), device,
            fillValueAttr);
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "TTNN doesn't currently support tensor creation from multiple "
              "given values (issue #685)");
    }

    return success();
  }

private:
  LogicalResult checkBasicLegality(ttir::ConstantOp &op,
                                   ::mlir::ElementsAttr &valueAttr,
                                   ConversionPatternRewriter &rewriter) const {
    if (!valueAttr.getElementType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "TTNN doesn't currently support tensor creation from values "
              "which are not integer or floating point numbers");
    }

    return success();
  }
};

} // namespace

// ANCHOR: adding_an_op_matmul_op_rewriter
class MatmulOpConversionPattern : public OpConversionPattern<ttir::MatmulOp> {
public:
  using OpConversionPattern<ttir::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::MatmulOp>(
        op, this->getTypeConverter()->convertType(op.getType()), adaptor.getA(),
        adaptor.getB(), adaptor.getOutput());
    return success();
  }
};
// ANCHOR_END: adding_an_op_matmul_op_rewriter

class Conv2dOpConversionPattern : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = getOrInsertDevice(rewriter, op);
    auto kernel_ty =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    llvm::ArrayRef<std::int64_t> kernel_shape = kernel_ty.getShape();

    auto input_ty = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> input_shape = input_ty.getShape();

    auto output_ty =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<std::int64_t> output_shape = output_ty.getShape();

    auto in_channels =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 1]);
    auto out_channels =
        rewriter.getI32IntegerAttr(output_shape[output_shape.size() - 1]);
    auto batch_size =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 4]);
    auto input_height =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 3]);
    auto input_width =
        rewriter.getI32IntegerAttr(input_shape[input_shape.size() - 2]);

    auto kernel_height =
        rewriter.getI32IntegerAttr(kernel_shape[kernel_shape.size() - 2]);
    auto kernel_width =
        rewriter.getI32IntegerAttr(kernel_shape[kernel_shape.size() - 1]);

    auto stride_height = rewriter.getI32IntegerAttr(adaptor.getStrideHeight());
    auto stride_width = rewriter.getI32IntegerAttr(adaptor.getStrideWidth());

    assert(
        adaptor.getPaddingBottom() == adaptor.getPaddingTop() &&
        "TTNN only supports padding height/width attributes. Thus, padding_top "
        "must equal padding_bottom for the op to execute as expected.");
    assert(adaptor.getPaddingLeft() == adaptor.getPaddingRight() &&
           "TTNN only supports padding height/width attributes. Thus, "
           "padding_left must equal padding_right for the op to execute as "
           "expected.");
    auto padding_height = rewriter.getI32IntegerAttr(adaptor.getPaddingTop());
    auto padding_width = rewriter.getI32IntegerAttr(adaptor.getPaddingRight());

    auto dilation_height =
        rewriter.getI32IntegerAttr(adaptor.getDilationHeight());
    auto dilation_width =
        rewriter.getI32IntegerAttr(adaptor.getDilationWidth());
    auto groups = rewriter.getI32IntegerAttr(adaptor.getGroups());
    rewriter.replaceOpWithNewOp<ttnn::Conv2dOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
        adaptor.getOutput(), device, in_channels, out_channels, batch_size,
        input_height, input_width, kernel_height, kernel_width, stride_height,
        stride_width, padding_height, padding_width, dilation_height,
        dilation_width, groups);
    return success();
  }
};

class MaxPool2dOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    assert(adaptor.getPaddingBottom() == adaptor.getPaddingTop() &&
           "TTNN max_pool2d does not support padding top/bottom/left/right "
           "separately");
    assert(adaptor.getPaddingLeft() == adaptor.getPaddingRight() &&
           "TTNN max_pool2d does not support padding top/bottom/left/right "
           "separately");

    auto device = getOrInsertDevice(rewriter, op);
    auto input_ty = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> input_shape = input_ty.getShape();

    auto batch_size =
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 4]);
    auto channels =
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 1]);

    assert(adaptor.getOriginalHeight().has_value() &&
           "ttir::MaxPool2dOp must have original_height set before translating "
           "to TTNN dialect.");
    assert(adaptor.getOriginalWidth().has_value() &&
           "ttir::MaxPool2dOp must have original_width set before translating "
           "to TTNN dialect.");

    rewriter.replaceOpWithNewOp<ttnn::MaxPool2dOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getOutput(), device, batch_size,
        adaptor.getOriginalHeightAttr(), adaptor.getOriginalWidthAttr(),
        channels, adaptor.getKernelHeightAttr(), adaptor.getKernelWidthAttr(),
        adaptor.getStrideHeightAttr(), adaptor.getStrideWidthAttr(),
        adaptor.getDilationHeightAttr(), adaptor.getDilationWidthAttr(),
        adaptor.getCeilModeAttr(), adaptor.getPaddingTopAttr(),
        adaptor.getPaddingRightAttr());
    return success();
  }
};

class BroadcastOpConversionPattern
    : public OpConversionPattern<ttir::BroadcastOp> {
  using OpConversionPattern<ttir::BroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::BroadcastOp srcOp, ttir::BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Fold this operation into all consumer ops. It will only work with TTNN
    // ops that support implicit broadcasting. We expect each Op's verify
    // function to assert their arguments to verify that they can broadcast.

    if (srcOp->getUsers().empty()) {
      // This broadcast chain has already been replaced.
      rewriter.eraseOp(srcOp);
      return success();
    }

    mlir::Value input = srcOp.getOperand(0);

    mlir::Operation *nextOp = srcOp;
    while (isa<ttir::BroadcastOp>(*nextOp->getUsers().begin())) {
      assert(nextOp->hasOneUse() &&
             "Broadcast with multiple uses are not supported");
      nextOp = *nextOp->getUsers().begin();
      if (nextOp->getUsers().empty()) {
        // This broadcast chain has already been replaced.
        rewriter.eraseOp(srcOp);
        return success();
      }
    }

    rewriter.replaceAllOpUsesWith(nextOp, input);
    rewriter.eraseOp(srcOp);

    return success();
  }
};

class SubtractOpConversionPattern
    : public OpConversionPattern<ttir::SubtractOp> {
  using OpConversionPattern<ttir::SubtractOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::SubtractOp srcOp, ttir::SubtractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType lhsType =
        mlir::cast<RankedTensorType>(adaptor.getInputs().front().getType());
    RankedTensorType rhsType =
        mlir::cast<RankedTensorType>(adaptor.getInputs().back().getType());

    if (lhsType.getShape() == rhsType.getShape()) {
      rewriter.replaceOpWithNewOp<ttnn::SubtractOp>(
          srcOp, adaptor.getInputs().front(), adaptor.getInputs().back(),
          adaptor.getOutputs().front());

      // Broadcast for rhs operand require the operation to be commutative to
      // allow switching the order of operands. To allow this conversion, the
      // following conversion is applied to SubtractOp: subtractOp(lhs,rhs) ->
      // addOp(lhs, negOp(rhs))

    } else {
      Value device = getOrInsertDevice(rewriter, srcOp);
      tensor::EmptyOp negEmptyOp = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), this->getTypeConverter()->convertType(rhsType),
          device);
      ttnn::NegOp negOp = rewriter.create<ttnn::NegOp>(
          srcOp.getLoc(), adaptor.getInputs().back(), negEmptyOp);

      rewriter.replaceOpWithNewOp<ttnn::AddOp>(
          srcOp, adaptor.getInputs().front(), negOp.getResults().front(),
          adaptor.getOutputs().front());
    }

    return success();
  }
};

namespace mlir::tt {

void populateTTIRToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // clang-format off
  // ANCHOR: op_rewriter_pattern_set
  patterns
      .add<TensorEmptyConversionPattern,
           ToLayoutOpConversionPattern,
           ElementwiseOpConversionPattern<ttir::AbsOp, ttnn::AbsOp>,
           ElementwiseOpConversionPattern<ttir::AddOp, ttnn::AddOp>,
           ElementwiseOpConversionPattern<ttir::LogicalAndOp, ttnn::LogicalAndOp>,
           ElementwiseOpConversionPattern<ttir::LogicalOrOp, ttnn::LogicalOrOp>,
           ElementwiseOpConversionPattern<ttir::LogicalNotOp, ttnn::LogicalNotOp>,
           ElementwiseOpConversionPattern<ttir::MultiplyOp, ttnn::MultiplyOp>,
           ElementwiseOpConversionPattern<ttir::EqualOp, ttnn::EqualOp>,
           ElementwiseOpConversionPattern<ttir::NotEqualOp, ttnn::NotEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterThanOp, ttnn::GreaterThanOp>,
           ElementwiseOpConversionPattern<ttir::LessEqualOp, ttnn::LessEqualOp>,
           ElementwiseOpConversionPattern<ttir::LessThanOp, ttnn::LessThanOp>,
           ElementwiseOpConversionPattern<ttir::MaximumOp, ttnn::MaximumOp>,
           ElementwiseOpConversionPattern<ttir::NegOp, ttnn::NegOp>,
           ElementwiseOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
           ElementwiseOpConversionPattern<ttir::SqrtOp, ttnn::SqrtOp>,
           ElementwiseOpConversionPattern<ttir::RsqrtOp, ttnn::RsqrtOp>,
           ElementwiseOpConversionPattern<ttir::SigmoidOp, ttnn::SigmoidOp>,
           ElementwiseOpConversionPattern<ttir::TypecastOp, ttnn::TypecastOp>,
           ElementwiseOpConversionPattern<ttir::ReciprocalOp, ttnn::ReciprocalOp>,
           ElementwiseOpConversionPattern<ttir::ExpOp, ttnn::ExpOp>,
           ElementwiseOpConversionPattern<ttir::DivOp, ttnn::DivOp>,
           ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
           ReductionOpConversionPattern<ttir::MeanOp, ttnn::MeanOp>,
           ReductionOpConversionPattern<ttir::MaxOp, ttnn::MaxOp>,
           BroadcastOpConversionPattern,
           EmbeddingOpConversionPattern,
           SoftmaxOpConversionPattern,
           TransposeOpConversionPattern,
           ConcatOpConversionPattern,
           ReshapeOpConversionPattern,
           SliceOpConversionPattern,
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           MaxPool2dOpConversionPattern,
           SubtractOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
