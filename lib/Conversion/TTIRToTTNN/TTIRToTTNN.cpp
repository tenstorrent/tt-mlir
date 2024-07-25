// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

static Value findDevice(Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::OpenDeviceOp>(op)) {
      return deviceOp.getResult();
    }
  }
  assert(false && "No device found");
  return nullptr;
}

class TensorEmptyToFullConversionPattern
    : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = findDevice(op);
    rewriter.replaceOpWithNewOp<ttnn::FullOp>(
        op, this->getTypeConverter()->convertType(op.getType()), device,
        rewriter.getF32FloatAttr(0.0));
    return success();
  }
};

class LayoutOpConversionPattern : public OpConversionPattern<ttir::LayoutOp> {
public:
  using OpConversionPattern<ttir::LayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::ToMemoryConfigOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getOutput());
    return success();
  }
};

template <typename TTIROp, typename TTNNOp,
          typename OpAdaptor = typename TTIROp::Adaptor>
class ElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TTNNOp>(op, resultTypes, adaptor.getInputs(),
                                        adaptor.getOutputs());
    return success();
  }
};

template <typename TTIROp, typename TTNNOp,
          typename OpAdaptor = typename TTIROp::Adaptor>
class ReductionOpConversionPattern : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTNNOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getOutput(), adaptor.getKeepDim(),
        adaptor.getDimArg().value_or(nullptr));
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
        adaptor.getInput(), adaptor.getOutput(), adaptor.getDimension());
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

namespace mlir::tt {

void populateTTIRToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // clang-format off
  // ANCHOR: adding_an_op_matmul_rewrite_pattern_set
  patterns
      .add<TensorEmptyToFullConversionPattern,
           LayoutOpConversionPattern,
           ElementwiseBinaryOpConversionPattern<ttir::AddOp, ttnn::AddOp>,
           ElementwiseBinaryOpConversionPattern<ttir::SubtractOp, ttnn::SubtractOp>,
           ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp, ttnn::MultiplyOp>,
           ElementwiseBinaryOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
           ElementwiseBinaryOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
           ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
           SoftmaxOpConversionPattern,
           MatmulOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: adding_an_op_matmul_rewrite_pattern_set
  // clang-format on
}

} // namespace mlir::tt
