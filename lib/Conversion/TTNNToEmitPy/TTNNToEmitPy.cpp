// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/Utils.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include <string>

using namespace mlir;
using namespace mlir::tt;

// Eltwise Unary Op conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseUnaryOpConversionPattern : public OpConversionPattern<TTNNOpTy> {
public:
  using OpConversionPattern<TTNNOpTy>::OpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy eltwiseUnaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 1> attrs =
        ttnn_to_emitpy::utils::createIndexArray<1>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        eltwiseUnaryOp,
        this->getTypeConverter()->convertType(eltwiseUnaryOp.getType()),
        eltwiseUnaryOp.getOperationName(), adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// Eltwise Binary Op conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseBinaryOpConversionPattern : public OpConversionPattern<TTNNOpTy> {
public:
  using OpConversionPattern<TTNNOpTy>::OpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy eltwiseBinaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 2> attrs =
        ttnn_to_emitpy::utils::createIndexArray<2>(rewriter);
    ;
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        eltwiseBinaryOp,
        this->getTypeConverter()->convertType(eltwiseBinaryOp.getType()),
        eltwiseBinaryOp.getOperationName(), adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// MatmulOp conversion pattern
//
namespace {
class MatmulOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttnn::MatmulOp> {
public:
  using OpConversionPattern<mlir::tt::ttnn::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 2> attrs =
        ttnn_to_emitpy::utils::createIndexArray<2>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        matmulOp, this->getTypeConverter()->convertType(matmulOp.getType()),
        "ttnn.matmul", adaptor.getOperands(), rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

namespace {
class SoftmaxOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttnn::SoftmaxOp> {
public:
  using OpConversionPattern<mlir::tt::ttnn::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SoftmaxOp softmaxOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        softmaxOp, this->getTypeConverter()->convertType(softmaxOp.getType()),
        "ttnn.softmax", adaptor.getOperands());
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
  matchAndRewrite(arith::ConstantOp constantOp,
                  arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        constantOp, this->getTypeConverter()->convertType(constantOp.getType()),
        "constant", ValueRange(),
        rewriter.getArrayAttr(constantOp.getValueAttr()));
    return success();
  }
};
} // namespace

// GetTupleElementOp conversion pattern
//
namespace {
class GetTupleElementOpConversionPattern
    : public OpConversionPattern<mlir::tt::GetTupleElementOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::GetTupleElementOp getTupleElementOp,
                  mlir::tt::GetTupleElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SubscriptOp requires a Value object as index, which is created by
    // invoking the emitpy::LiteralOp.
    //
    Value indexAsVal = rewriter.create<emitpy::LiteralOp>(
        getTupleElementOp->getLoc(), rewriter.getIndexType(),
        std::to_string(adaptor.getIndex()));

    // SubscriptOp returns an emitpy::LValueType, so we wrap the
    // OpaqueType with LValueType.
    //
    emitpy::LValueType lvalueReturnType =
        emitpy::LValueType::get(emitpy::OpaqueType::get(
            rewriter.getContext(), ttnn_to_emitpy::TypeNameV<::ttnn::Tensor>));

    Value subscript = rewriter.create<emitpy::SubscriptOp>(
        getTupleElementOp->getLoc(), lvalueReturnType, adaptor.getOperand(),
        indexAsVal);

    // As SubscriptOp returns an LValueType, we need to convert it to an
    // OpaqueType - this is done by invoking the emitpy::LoadOp.
    //
    rewriter.replaceOpWithNewOp<emitpy::LoadOp>(
        getTupleElementOp,
        emitpy::OpaqueType::get(getContext(),
                                ttnn_to_emitpy::TypeNameV<::ttnn::Tensor>),
        subscript);
    return success();
  }
};
} // namespace

// TupleOp conversion pattern
//
namespace {
class TupleOpConversionPattern : public OpConversionPattern<mlir::tt::TupleOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::TupleOp tupleOp, mlir::tt::TupleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        tupleOp, this->getTypeConverter()->convertType(tupleOp.getType()),
        "util_create_list", adaptor.getOperands());
    return success();
  }
};
} // namespace

// DeallocateOp conversion pattern
//
namespace {
class DeallocateOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttnn::DeallocateOp> {

public:
  using OpConversionPattern<mlir::tt::ttnn::DeallocateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::DeallocateOp deallocateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 1> attrs =
        ttnn_to_emitpy::utils::createIndexArray<1>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        deallocateOp, mlir::TypeRange{}, "ttnn.deallocate",
        adaptor.getOperands(), rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// mlir::tt::DeviceOp conversion pattern
//
namespace {
struct TTDeviceOpConversionPattern
    : public OpConversionPattern<mlir::tt::DeviceOp> {
  using OpConversionPattern<mlir::tt::DeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::DeviceOp deviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(deviceOp);
    return success();
  }
};
} // namespace

// GetDeviceOp conversion pattern
//
namespace {
class GetDeviceOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttnn::GetDeviceOp> {

public:
  using OpConversionPattern<mlir::tt::ttnn::GetDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::GetDeviceOp getDeviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        getDeviceOp,
        this->getTypeConverter()->convertType(getDeviceOp.getType()),
        "my_get_device.DeviceGetter.get_device", adaptor.getOperands());
    return success();
  }
};
} // namespace

// ToDeviceOp conversion pattern
//
namespace {
class ToDeviceOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttnn::ToDeviceOp> {

public:
  using OpConversionPattern<mlir::tt::ttnn::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToDeviceOp toDeviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Attribute, 2> attrs =
        ttnn_to_emitpy::utils::createIndexArray<2>(rewriter);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        toDeviceOp, this->getTypeConverter()->convertType(toDeviceOp.getType()),
        "ttnn.to_device", adaptor.getOperands(), rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// NamedFullOp conversion pattern for operations like ttnn::zeros or ttnn::ones
//
// TODO (amilovanovic) : add support for other attributes
// Currently, only shape and layout attributes are supported
//
namespace {
template <typename SourceOp>
class NamedFullOpConversionPattern : public OpConversionPattern<SourceOp> {

public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp namedFullOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::tt::ttnn::ShapeAttr shapeAttr = namedFullOp.getShapeAttr();
    llvm::SmallVector<Attribute, 2> attrs;
    emitpy::OpaqueAttr shapeAttrConverted =
        ttnn_to_emitpy::utils::convertShape(rewriter, shapeAttr);
    emitpy::OpaqueAttr layoutAttr = ttnn_to_emitpy::utils::convertLayoutAttr(
        rewriter, namedFullOp.getLayoutAttr());
    attrs.push_back(shapeAttrConverted);
    attrs.push_back(layoutAttr);
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        namedFullOp,
        this->getTypeConverter()->convertType(namedFullOp.getType()),
        namedFullOp.getOperationName(), ValueRange(),
        rewriter.getArrayAttr(attrs));

    return success();
  }
};
} // namespace

// ModuleOp conversion pattern
//
// This conversion pattern removes attributes from the ModuleOp.
//
namespace {
class ModuleOpConversionPattern : public OpConversionPattern<mlir::ModuleOp> {

public:
  using OpConversionPattern<mlir::ModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ModuleOp moduleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(moduleOp, [&]() {
      for (const NamedAttribute &attr : moduleOp->getAttrs()) {
        moduleOp->removeAttr(attr.getName());
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
  patterns.add<NamedFullOpConversionPattern<mlir::tt::ttnn::ZerosOp>, NamedFullOpConversionPattern<mlir::tt::ttnn::OnesOp>>(typeConverter, ctx);
  // Arith ops
  //
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);
  // Tuple ops
  //
  patterns.add<GetTupleElementOpConversionPattern>(typeConverter, ctx);
  patterns.add<TupleOpConversionPattern>(typeConverter, ctx);
  // Matmul ops
  //
  patterns.add<MatmulOpConversionPattern>(typeConverter, ctx);
  // Eltwise unary ops
  //
  patterns.add<EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::ReluOp>>(typeConverter,
                                                              ctx);
  // Eltwise binary ops
  //
  patterns.add<EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::AddOp>>(typeConverter,
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
