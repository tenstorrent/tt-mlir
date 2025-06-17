// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Conversion/TTNNToEmitPy/Utils.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::tt;
using namespace ttnn_to_emitpy::utils;

// Eltwise Unary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseUnaryOpConversionPattern : public OpConversionPattern<TTNNOpTy> {
public:
  using OpConversionPattern<TTNNOpTy>::OpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 1> attrs = createIndexArray<1>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        op.getOperationName(), adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseBinaryOpConversionPattern : public OpConversionPattern<TTNNOpTy> {
public:
  using OpConversionPattern<TTNNOpTy>::OpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 2> attrs = createIndexArray<2>(rewriter);
    ;
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        op.getOperationName(), adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

namespace {
class MatmulOpConversionPattern : public OpConversionPattern<ttnn::MatmulOp> {
public:
  using OpConversionPattern<ttnn::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 2> attrs = createIndexArray<2>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()), "ttnn.matmul",
        adaptor.getOperands(), rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

namespace {
class SoftmaxOpConversionPattern : public OpConversionPattern<ttnn::SoftmaxOp> {
public:
  using OpConversionPattern<ttnn::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()), "ttnn.softmax",
        adaptor.getOperands());
    return success();
  }
};
} // namespace

// arith::ConstantOp conversion pattern
//
namespace {
class ArithConstantOpConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()), "constant",
        ValueRange(), rewriter.getArrayAttr(op.getValueAttr()));
    return success();
  }
};
} // namespace

// DeallocateOp conversion pattern
//
namespace {
class DeallocateOpConversionPattern
    : public OpConversionPattern<ttnn::DeallocateOp> {

public:
  using OpConversionPattern<ttnn::DeallocateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::DeallocateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 1> attrs = createIndexArray<1>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, mlir::TypeRange{}, "ttnn.deallocate", adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// tt::DeviceOp conversion pattern
//
namespace {
struct TTDeviceOpConversionPattern : public OpConversionPattern<tt::DeviceOp> {
  using OpConversionPattern<tt::DeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::DeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// GetDeviceOp conversion pattern
//
namespace {
class GetDeviceOpConversionPattern
    : public OpConversionPattern<ttnn::GetDeviceOp> {

public:
  using OpConversionPattern<ttnn::GetDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::GetDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        "my_get_device.DeviceGetter.get_device", adaptor.getOperands());
    return success();
  }
};
} // namespace

// ToDeviceOp conversion pattern
//
namespace {
class ToDeviceOpConversionPattern
    : public OpConversionPattern<ttnn::ToDeviceOp> {

public:
  using OpConversionPattern<ttnn::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ToDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 2> attrs = createIndexArray<2>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        "ttnn.to_device", adaptor.getOperands(), rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// NamedFullOp conversion pattern for operations like ttnn::zeros or ttnn::ones
//
namespace {
template <typename SourceOp>
class NamedFullOpConversionPattern : public OpConversionPattern<SourceOp> {

public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn::ShapeAttr shapeAttr = op.getShapeAttr();
    llvm::SmallVector<Attribute, 2> attrs;
    emitpy::OpaqueAttr shapeAttrConverted = convertShape(rewriter, shapeAttr);
    emitpy::OpaqueAttr layoutAttr =
        convertLayoutAttr(rewriter, op.getLayoutAttr());
    attrs.push_back(shapeAttrConverted);
    attrs.push_back(layoutAttr);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        op.getOperationName(), ValueRange(), rewriter.getArrayAttr(attrs));

    return success();
  }
};
} // namespace

// Module Op conversion pattern
//
// This conversion pattern removes attributes from the ModuleOp. Previously,
// ttmlir-translate would complain when translating to C++ if there were any
// attributes from "unregistered" dialects.
//
namespace {
class ModuleOpConversionPattern : public OpConversionPattern<mlir::ModuleOp> {

public:
  using OpConversionPattern<mlir::ModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&]() {
      for (const NamedAttribute &attr : op->getAttrs()) {
        op->removeAttr(attr.getName());
      }
    });

    return success();
  }
};
} // namespace

namespace mlir::tt {

void populateTTNNToEmitPyPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  // Device ops
  //
  patterns.add<TTDeviceOpConversionPattern>(typeConverter, ctx);
  patterns.add<GetDeviceOpConversionPattern>(typeConverter, ctx);
  // Tensor ops
  //
  // clang-format off
  patterns.add<NamedFullOpConversionPattern<tt::ttnn::ZerosOp>, NamedFullOpConversionPattern<tt::ttnn::OnesOp>>(typeConverter, ctx);
  // Arith ops
  //
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);
  // Matmul ops
  //
  patterns.add<MatmulOpConversionPattern>(typeConverter, ctx);
  // Eltwise unary ops
  //
  patterns.add<EltwiseUnaryOpConversionPattern<ttnn::ReluOp>>(typeConverter,
                                                              ctx);
  // Eltwise binary ops
  //
  patterns.add<EltwiseBinaryOpConversionPattern<ttnn::AddOp>>(typeConverter,
                                                              ctx);
  // Memory ops
  //
  patterns.add<ToDeviceOpConversionPattern, DeallocateOpConversionPattern>(
      typeConverter, ctx);
  // Other ops
  //
  patterns.add<SoftmaxOpConversionPattern>(typeConverter, ctx);
  // Module op
  //
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
