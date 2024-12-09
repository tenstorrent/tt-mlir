// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <mlir/IR/Operation.h>

using namespace mlir;
using namespace mlir::tt;

namespace {

class TensorEmptyConversionPattern
    : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Get the shape of the tensor, tensor layout, and data type
    //
    ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
        rewriter.getContext(),
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape());
    DataType dtype = layoutAttr.getDataType();
    ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
    if (layoutAttr.isTiled()) {
      ttnnLayoutEnum = ttnn::Layout::Tile;
    } else {
      ttnnLayoutEnum = ttnn::Layout::RowMajor;
    }
    DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

    // If the tensor is not going to device, we can create the op without
    // device-specific attributes
    //
    ttnn::TensorMemoryLayoutAttr memLayout = layoutAttr.getMemLayout();
    if (!memLayout) {
      rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
          op, this->getTypeConverter()->convertType(op.getType()), nullptr,
          shapeAttr, dTypeAttr, tensorLayoutAttr, nullptr);

      return success();
    }

    ttnn::BufferType bufferType = layoutAttr.getBufferType();

    // Create MemoryConfigAttr
    //
    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);
    llvm::SmallVector<int64_t> shardShape = layoutAttr.getShardShape();
    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        op.getContext(), ttnn::BufferTypeAttr::get(op.getContext(), bufferType),
        ttnn::ShardSpecAttr::get(
            op.getContext(), ttnn::ShapeAttr::get(op.getContext(), shardShape)),
        memLayout);

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(
        op, this->getTypeConverter()->convertType(op.getType()), device,
        shapeAttr, dTypeAttr, tensorLayoutAttr, memoryConfigAttr);

    return success();
  }
};

class ToLayoutOpConversionPattern
    : public OpConversionPattern<ttir::ToLayoutOp> {
public:
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the DPS operand and delete it's creator op, if it's tensor::emptyOp
    //
    Value dpsOperand = adaptor.getOperands().back();
    ttnn::EmptyOp emptyOp = dpsOperand.getDefiningOp<ttnn::EmptyOp>();
    if (emptyOp) {
      rewriter.eraseOp(emptyOp);
    }

    auto outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        op.getResult().getType().getEncoding());

    // Determine the output data type
    DataType dtype = outputLayoutAttr.getDataType();
    DataTypeAttr outputDataType =
        DataTypeAttr::get(rewriter.getContext(), dtype);

    // Determine the output layout (tile or row major)
    ttnn::BufferType outputBufferType = outputLayoutAttr.getBufferType();

    ttnn::Layout outputLayoutEnum = outputLayoutAttr.getLayout();

    bool isOutputOnHost = (outputBufferType == ttnn::BufferType::SystemMemory);

    RankedTensorType result = mlir::cast<RankedTensorType>(op.getType());
    if (!isOutputOnHost) {
      // TODO(bug #665):
      // Binary ops fail with row major layout in ttnn, defaulting to and
      // assuming tile layout for all device tensors...
      // Note: mlir doesn't know about this, so tensors may still appear as row
      // major in the generated mlir
      // TODO(bug #875):
      // Remove the following code block once constraints modelling is
      // implemented on dialect level
      //
      // Default to Tile layout unless op supports only RowMajor layout
      //
      ttnn::Layout newOutputLayoutEnum =
          shouldForceRowMajor(op) ? ttnn::Layout::RowMajor : ttnn::Layout::Tile;

      // If the layout of the output tensor changed as a result of forcing the
      // layout update the tensor type
      if (outputLayoutEnum != newOutputLayoutEnum) {
        result =
            getLayoutForcedResultTensor(rewriter, result, newOutputLayoutEnum);
        op.getResult().setType(result);
        outputLayoutAttr =
            mlir::cast<ttnn::TTNNLayoutAttr>(result.getEncoding());
        outputLayoutEnum = newOutputLayoutEnum;
      }
    }

    ttnn::LayoutAttr outputLayout =
        ttnn::LayoutAttr::get(rewriter.getContext(), outputLayoutEnum);
    llvm::SmallVector<int64_t> outputShardShape =
        outputLayoutAttr.getShardShape();

    ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
        rewriter.getContext(),
        ttnn::BufferTypeAttr::get(rewriter.getContext(), outputBufferType),
        ttnn::ShardSpecAttr::get(
            op.getContext(),
            ttnn::ShapeAttr::get(rewriter.getContext(), outputShardShape)),
        outputLayoutAttr.getMemLayout());

    rewriter.replaceOpWithNewOp<ttnn::ToLayoutOp>(
        op, this->getTypeConverter()->convertType(result), adaptor.getInput(),
        outputLayout, outputDataType, outputMemConfigAttr,
        isOutputOnHost ? nullptr
                       : ::ttnn::utils::getOrInsertDevice(rewriter, op));

    return success();
  }

private:
  bool shouldForceRowMajor(ttir::ToLayoutOp op) const {
    for (mlir::Operation *user : op.getResult().getUsers()) {
      if (isa<ttir::Conv2dOp>(user) || isa<ttir::MaxPool2dOp>(user) ||
          isa<ttir::SliceOp>(user) || isa<ttir::EmbeddingOp>(user)) {
        return true;
      }
    }

    return false;
  }

  RankedTensorType
  getLayoutForcedResultTensor(ConversionPatternRewriter &rewriter,
                              RankedTensorType oldOutput,
                              ttnn::Layout newOutputLayoutEnum) const {
    auto oldOutputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(oldOutput.getEncoding());
    DataType outputDtype = oldOutputLayoutAttr.getDataType();
    SmallVector<std::int64_t> oldShardShape =
        oldOutputLayoutAttr.getShardShape();
    size_t shardShapeSize = oldShardShape.size();
    assert(shardShapeSize >= 2 && "expected at least 2D shape");

    if (newOutputLayoutEnum == ttnn::Layout::RowMajor) {
      // Set shard shape to match convention of row major layout
      auto tileType =
          mlir::cast<TileType>(oldOutputLayoutAttr.getElementType());
      llvm::SmallVector<int64_t> newShardShape(oldShardShape.begin(),
                                               oldShardShape.end());
      newShardShape[shardShapeSize - 2] =
          oldShardShape[shardShapeSize - 2] * tileType.getHeight();
      newShardShape[shardShapeSize - 1] =
          oldShardShape[shardShapeSize - 1] * tileType.getWidth();
      Type newElementType = ttnn::utils::createRowMajorTypeFromDtype(
          rewriter.getContext(), outputDtype);
      RankedTensorType result = RankedTensorType::get(
          oldOutput.getShape(), oldOutput.getElementType(),
          oldOutputLayoutAttr
              .withElementType(rewriter.getContext(), newElementType)
              .withShardShape(rewriter.getContext(), newShardShape));
      return result;
    }

    if (newOutputLayoutEnum == ttnn::Layout::Tile) {
      TileType tileType =
          TileType::get(rewriter.getContext(),
                        {ttnn::TILE_HEIGHT, ttnn::TILE_WIDTH}, outputDtype);
      RankedTensorType result = RankedTensorType::get(
          oldOutput.getShape(), oldOutput.getElementType(),
          oldOutputLayoutAttr.withElementType(rewriter.getContext(), tileType));
      return result;
    }

    llvm_unreachable("Unreachable code path. Unexpected output layout enum");
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
        adaptor.getInput(), adaptor.getOutput(), adaptor.getWeight());

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

class ClampOpConversionPattern : public OpConversionPattern<ttir::ClampOp> {
public:
  using OpConversionPattern<ttir::ClampOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::ClampOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getMin(), adaptor.getMax());
    return success();
  }
};

class UpdateCacheOpConversionPattern
    : public OpConversionPattern<ttir::UpdateCacheOp> {
public:
  using OpConversionPattern<ttir::UpdateCacheOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UpdateCacheOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The TTIR version of this op is pure. In TTNN this op is in-place.
    // We need to replace uses of the result ot the TTIR op with uses
    // of the cache argument.
    //
    // The presence of the MemWrite trait of this op should preserve
    // the order of this op relative to the cache arguments uses, preserving
    // program correctness.

    // This op can only work if it is the final use of the cache tensor in the
    // order of execution. For now, checking that there is only one user (this
    // op) of the cache tensor will suffice.
    std::vector<mlir::Operation *> users(op.getCache().getUsers().begin(),
                                         op.getCache().getUsers().end());
    if (users.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "UpdateCacheOp must have exactly one user");
    }

    rewriter.create<ttnn::UpdateCacheOp>(
        op.getLoc(), adaptor.getCache(), adaptor.getInput(),
        adaptor.getUpdateIndex(), adaptor.getBatchOffset());

    rewriter.replaceOp(op, adaptor.getCache());
    return success();
  }
};

class FillCacheOpConversionPattern
    : public OpConversionPattern<ttir::FillCacheOp> {
public:
  using OpConversionPattern<ttir::FillCacheOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FillCacheOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The TTIR version of this op is pure. In TTNN this op is in-place.
    // We need to replace uses of the result ot the TTIR op with uses
    // of the cache argument.
    //
    // The presence of the MemWrite trait of this op should preserve
    // the order of this op relative to the cache arguments uses, preserving
    // program correctness.

    // This op can only work if it is the final use of the cache tensor in the
    // order of execution. For now, checking that there is only one user (this
    // op) of the cache tensor will suffice.
    std::vector<mlir::Operation *> users(op.getCache().getUsers().begin(),
                                         op.getCache().getUsers().end());
    if (users.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "FillCacheOp must have exactly one user");
    }

    rewriter.create<ttnn::FillCacheOp>(op.getLoc(), adaptor.getCache(),
                                       adaptor.getInput(),
                                       adaptor.getBatchOffset());

    rewriter.replaceOp(op, adaptor.getCache());
    return success();
  }
};

template <typename TTIROpTy, typename TTNNOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseUnaryWithFloatParameterOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOpTy>(
        op, this->getTypeConverter()->convertType(op.getType(0)),
        adaptor.getInputs(), adaptor.getOutputs(), adaptor.getParameter());
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
      Value device = ::ttnn::utils::getOrInsertDevice(rewriter, op);
      float fillValue =
          valueAttr.getElementType().isInteger()
              ? getIntegerValue(valueAttr)
              : valueAttr.getSplatValue<mlir::APFloat>().convertToFloat();

      ::mlir::FloatAttr fillValueAttr = rewriter.getF32FloatAttr(fillValue);
      rewriter.replaceOpWithNewOp<ttnn::FullOp>(
          op, this->getTypeConverter()->convertType(op.getType()), device,
          fillValueAttr);

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

  float getIntegerValue(mlir::ElementsAttr valueAttr) const {
    size_t bitWidth = valueAttr.getElementType().getIntOrFloatBitWidth();
    switch (bitWidth) {
    case 1:
      return static_cast<float>(valueAttr.getSplatValue<bool>());
    case 8:
      return static_cast<float>(valueAttr.getSplatValue<int8_t>());
    case 16:
      return static_cast<float>(valueAttr.getSplatValue<int16_t>());
    case 32:
      return static_cast<float>(valueAttr.getSplatValue<int>());
    case 64:
      return static_cast<float>(valueAttr.getSplatValue<int64_t>());
    }
    assert(false && "Unsupported integer type.");
  }
};

class LinearOpConversionPattern : public OpConversionPattern<ttir::LinearOp> {
public:
  using OpConversionPattern<ttir::LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::LinearOp>(
        op, this->getTypeConverter()->convertType(op.getType()), adaptor.getA(),
        adaptor.getB(), adaptor.getBias(), adaptor.getOutput());
    return success();
  }
};

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

static ttnn::ReshapeOp generateReshape(Value input, ArrayRef<int64_t> newShape,
                                       PatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto outputType = inputType.cloneWith(newShape, inputType.getElementType());

  std::vector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  return rewriter.create<ttnn::ReshapeOp>(
      input.getLoc(), outputType, input, rewriter.getI32ArrayAttr(newShapeI32));
}

static ttnn::ReshapeOp generateNHWFlatten(Value input,
                                          PatternRewriter &rewriter) {
  std::vector<int64_t> shape =
      mlir::cast<RankedTensorType>(input.getType()).getShape().vec();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  std::vector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                   shape[3]};
  return generateReshape(input, newShape, rewriter);
}

class Conv2dOpConversionPattern : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = ::ttnn::utils::getOrInsertDevice(rewriter, op);
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

    std::vector<int64_t> flattenedInputShape = {
        1, 1, input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]};
    Value flattenedInput = generateNHWFlatten(adaptor.getInput(), rewriter);

    std::vector<int64_t> flattenedOutputShape = {
        1, 1, output_shape[0] * output_shape[1] * output_shape[2],
        output_shape[3]};

    output_ty = mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
        output_ty.cloneWith(flattenedOutputShape, output_ty.getElementType())));

    // Using a tensor::EmptyOp so that the rewriter for EmptyOp can handle the
    // attribute determination
    auto convDPSOutput = rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        adaptor.getOutput().getDefiningOp(), flattenedOutputShape,
        output_ty.getElementType());

    // Must set the type to the output type to maintain the layout attributes
    convDPSOutput.getResult().setType(output_ty);

    ttnn::Conv2dOp new_conv = rewriter.create<ttnn::Conv2dOp>(
        op.getLoc(), output_ty, flattenedInput, adaptor.getWeight(),
        adaptor.getBias(), convDPSOutput, device, in_channels, out_channels,
        batch_size, input_height, input_width, kernel_height, kernel_width,
        stride_height, stride_width, padding_height, padding_width,
        dilation_height, dilation_width, groups);

    Value output = generateReshape(new_conv, output_shape, rewriter);

    rewriter.replaceOp(op, output);
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

    auto device = mlir::tt::ttnn::utils::getOrInsertDevice(rewriter, op);
    auto input_ty = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<std::int64_t> input_shape = input_ty.getShape();

    auto batch_size =
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 4]);
    auto channels =
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 1]);

    Value flattenedInput = generateNHWFlatten(adaptor.getInput(), rewriter);

    auto output_ty =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<std::int64_t> output_shape = output_ty.getShape();

    std::vector<int64_t> flattenedOutputShape = {
        1, 1, output_shape[0] * output_shape[1] * output_shape[2],
        output_shape[3]};

    output_ty = mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
        output_ty.cloneWith(flattenedOutputShape, output_ty.getElementType())));

    // Using a tensor::EmptyOp so that the rewriter for EmptyOp can handle the
    // attribute determination
    auto poolDPSOutput = rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        adaptor.getOutput().getDefiningOp(), flattenedOutputShape,
        output_ty.getElementType());

    // Must set the type to the output type to maintain the layout attributes
    poolDPSOutput.getResult().setType(output_ty);

    auto new_pool = rewriter.create<ttnn::MaxPool2dOp>(
        op.getLoc(), output_ty, flattenedInput, poolDPSOutput, device,
        batch_size,
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 3]),
        rewriter.getSI32IntegerAttr(input_shape[input_shape.size() - 2]),
        channels, adaptor.getKernelHeightAttr(), adaptor.getKernelWidthAttr(),
        adaptor.getStrideHeightAttr(), adaptor.getStrideWidthAttr(),
        adaptor.getDilationHeightAttr(), adaptor.getDilationWidthAttr(),
        adaptor.getCeilModeAttr(), adaptor.getPaddingTopAttr(),
        adaptor.getPaddingRightAttr());

    Value output = generateReshape(new_pool, output_shape, rewriter);

    rewriter.replaceOp(op, output);

    return success();
  }
};

class TypecastOpConversionPattern
    : public OpConversionPattern<ttir::TypecastOp> {
  using OpConversionPattern<ttir::TypecastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::TypecastOp op, ttir::TypecastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(
        *op.getInputs().begin());
    auto result = ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(
        *op.getResults().begin());

    ttnn::TTNNLayoutAttr outputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(result.getType().getEncoding());

    DataType outputDataType = outputLayoutAttr.getDataType();

    rewriter.replaceOpWithNewOp<ttnn::TypecastOp>(
        op, this->getTypeConverter()->convertType(op.getType(0)), input,
        outputDataType);
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
      Value device = ::ttnn::utils::getOrInsertDevice(rewriter, srcOp);
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

class AllGatherOpConversionPattern
    : public OpConversionPattern<ttir::AllGatherOp> {
public:
  using OpConversionPattern<ttir::AllGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AllGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::AllGatherOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getDim());
    return success();
  }
};

class ArangeOpConversionPattern : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());
    assert(static_cast<int64_t>(adaptor.getArangeDimension()) ==
               outputType.getRank() - 1 &&
           "Arange dimension must be the final dimension of the output tensor "
           "to convert to ttnn.arange");

    // Get ttnn::TTNNLayoutAttr of the result type
    //
    ttnn::TTNNLayoutAttr layoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());

    DataTypeAttr dtypeAttr = rewriter.getAttr<DataTypeAttr>(
        elementTypeToDataType(outputType.getElementType()));
    Value device = mlir::tt::ttnn::utils::getOrInsertDevice(rewriter, op);

    ttnn::MemoryConfigAttr memConfigAttr =
        rewriter.getAttr<ttnn::MemoryConfigAttr>(
            rewriter.getAttr<ttnn::BufferTypeAttr>(layoutAttr.getBufferType()),
            rewriter.getAttr<ttnn::ShardSpecAttr>(
                rewriter.getAttr<ttnn::ShapeAttr>(layoutAttr.getShardShape())),
            layoutAttr.getMemLayout());

    rewriter.replaceOpWithNewOp<ttnn::ArangeOp>(
        op, outputType, adaptor.getStart(), adaptor.getEnd(), adaptor.getStep(),
        dtypeAttr, device, memConfigAttr);

    return success();
  }
};

class ScatterOpConversionPattern : public OpConversionPattern<ttir::ScatterOp> {
public:
  using OpConversionPattern<ttir::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The ttnn interface has the inverse inputs of the TTIR dialect op (which
    // matches torch ops).
    rewriter.replaceOpWithNewOp<ttnn::ScatterOp>(
        op, adaptor.getUpdate(), adaptor.getInput(), adaptor.getOutput());

    return success();
  }
};
} // namespace

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
           ElementwiseOpConversionPattern<ttir::CbrtOp, ttnn::CbrtOp>,
           ElementwiseOpConversionPattern<ttir::FloorOp, ttnn::FloorOp>,
           ElementwiseOpConversionPattern<ttir::IsFiniteOp, ttnn::IsFiniteOp>,
           ElementwiseOpConversionPattern<ttir::LogicalAndOp, ttnn::LogicalAndOp>,
           ElementwiseOpConversionPattern<ttir::LogicalOrOp, ttnn::LogicalOrOp>,
           ElementwiseOpConversionPattern<ttir::LogicalNotOp, ttnn::LogicalNotOp>,
           ElementwiseOpConversionPattern<ttir::LogicalXorOp, ttnn::LogicalXorOp>,
           ElementwiseOpConversionPattern<ttir::MultiplyOp, ttnn::MultiplyOp>,
           ElementwiseOpConversionPattern<ttir::EqualOp, ttnn::EqualOp>,
           ElementwiseOpConversionPattern<ttir::NotEqualOp, ttnn::NotEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
           ElementwiseOpConversionPattern<ttir::GreaterThanOp, ttnn::GreaterThanOp>,
           ElementwiseOpConversionPattern<ttir::LessEqualOp, ttnn::LessEqualOp>,
           ElementwiseOpConversionPattern<ttir::LessThanOp, ttnn::LessThanOp>,
           ElementwiseOpConversionPattern<ttir::MaximumOp, ttnn::MaximumOp>,
           ElementwiseOpConversionPattern<ttir::MinimumOp, ttnn::MinimumOp>,
           ElementwiseOpConversionPattern<ttir::NegOp, ttnn::NegOp>,
           ElementwiseOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
           ElementwiseOpConversionPattern<ttir::GeluOp, ttnn::GeluOp>,
           ElementwiseOpConversionPattern<ttir::SqrtOp, ttnn::SqrtOp>,
           ElementwiseOpConversionPattern<ttir::RsqrtOp, ttnn::RsqrtOp>,
           ElementwiseOpConversionPattern<ttir::SignOp, ttnn::SignOp>,
           ElementwiseOpConversionPattern<ttir::SigmoidOp, ttnn::SigmoidOp>,
           ElementwiseOpConversionPattern<ttir::Log1pOp, ttnn::Log1pOp>,
           ElementwiseOpConversionPattern<ttir::ReciprocalOp, ttnn::ReciprocalOp>,
           ElementwiseOpConversionPattern<ttir::ExpOp, ttnn::ExpOp>,
           ElementwiseOpConversionPattern<ttir::LogOp, ttnn::LogOp>,
           ElementwiseOpConversionPattern<ttir::DivOp, ttnn::DivOp>,
           ElementwiseOpConversionPattern<ttir::CeilOp, ttnn::CeilOp>,
           ElementwiseOpConversionPattern<ttir::SinOp, ttnn::SinOp>,
           ElementwiseOpConversionPattern<ttir::CosOp, ttnn::CosOp>,
           ElementwiseOpConversionPattern<ttir::Expm1Op, ttnn::Expm1Op>,
           ElementwiseOpConversionPattern<ttir::RemainderOp, ttnn::RemainderOp>,
           ElementwiseOpConversionPattern<ttir::WhereOp, ttnn::WhereOp>,
           ElementwiseOpConversionPattern<ttir::TanOp, ttnn::TanOp>,
           ElementwiseOpConversionPattern<ttir::TanhOp, ttnn::TanhOp>,
           ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
           ReductionOpConversionPattern<ttir::MeanOp, ttnn::MeanOp>,
           ReductionOpConversionPattern<ttir::MaxOp, ttnn::MaxOp>,
	   ElementwiseUnaryWithFloatParameterOpConversionPattern<ttir::LeakyReluOp, ttnn::LeakyReluOp>,
           EmbeddingOpConversionPattern,
           SoftmaxOpConversionPattern,
           TransposeOpConversionPattern,
           TypecastOpConversionPattern,
           ClampOpConversionPattern,
           ConcatOpConversionPattern,
           ReshapeOpConversionPattern,
           SliceOpConversionPattern,
           SqueezeOpConversionPattern,
           UnsqueezeOpConversionPattern,
           ConstantOpConversionPattern,
           LinearOpConversionPattern,
           MatmulOpConversionPattern,
           Conv2dOpConversionPattern,
           MaxPool2dOpConversionPattern,
           SubtractOpConversionPattern,
           AllGatherOpConversionPattern,
           ArangeOpConversionPattern,
           UpdateCacheOpConversionPattern,
           FillCacheOpConversionPattern,
           ScatterOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
