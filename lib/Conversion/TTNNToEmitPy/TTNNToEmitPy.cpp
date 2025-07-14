// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/Utils.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "llvm/ADT/SmallVector.h"

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
        matmulOp.getOperationName(), adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
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
        softmaxOp.getOperationName(), adaptor.getOperands());
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
  matchAndRewrite(arith::ConstantOp constOp, arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = this->getTypeConverter()->convertType(constOp.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(constOp, "type conversion failed");
    }

    rewriter.replaceOpWithNewOp<emitpy::ConstantOp>(constOp, newTy,
                                                    adaptor.getValue());
    return success();
  }
};
} // namespace

// GetTupleElementOp conversion pattern
//
namespace {
class GetTupleElementOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttcore::GetTupleElementOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::GetTupleElementOp getTupleElementOp,
                  mlir::tt::ttcore::GetTupleElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SubscriptOp requires a Value object as index, which is created by
    // invoking the emitpy::LiteralOp.
    //
    Value indexAsVal = rewriter.create<emitpy::LiteralOp>(
        getTupleElementOp->getLoc(), rewriter.getIndexType(),
        std::to_string(adaptor.getIndex()));

    Value subscript = rewriter.create<emitpy::SubscriptOp>(
        getTupleElementOp->getLoc(),
        this->getTypeConverter()->convertType(getTupleElementOp.getType()),
        adaptor.getOperand(), indexAsVal);

    rewriter.replaceOpWithNewOp<emitpy::AssignOp>(
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
class TupleOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttcore::TupleOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::TupleOp tupleOp,
                  mlir::tt::ttcore::TupleOp::Adaptor adaptor,
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
        deallocateOp, mlir::TypeRange{}, deallocateOp.getOperationName(),
        adaptor.getOperands(), rewriter.getArrayAttr(attrs));
    return success();
  }
};
} // namespace

// mlir::tt::ttcore::DeviceOp conversion pattern
//
namespace {
struct TTDeviceOpConversionPattern
    : public OpConversionPattern<mlir::tt::ttcore::DeviceOp> {
  using OpConversionPattern<mlir::tt::ttcore::DeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttcore::DeviceOp deviceOp, OpAdaptor adaptor,
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
        toDeviceOp.getOperationName(), adaptor.getOperands(),
        rewriter.getArrayAttr(attrs));
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
  // clang-format off
  patterns.add<TTDeviceOpConversionPattern,
               GetDeviceOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Tensor ops
  //
  // clang-format off
  patterns.add<NamedFullOpConversionPattern<mlir::tt::ttnn::ZerosOp>,
               NamedFullOpConversionPattern<mlir::tt::ttnn::OnesOp>>(typeConverter, ctx);
  // clang-format on

  // Arith ops
  //
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);

  // Tuple ops
  //
  // clang-format off
  patterns.add<GetTupleElementOpConversionPattern,
               TupleOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Matmul ops
  //
  patterns.add<MatmulOpConversionPattern>(typeConverter, ctx);

  // Eltwise unary ops
  //
  // clang-format off
  patterns.add<EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::ReluOp>>(typeConverter, ctx);
  // clang-format on

  // Eltwise binary ops
  //
  // clang-format off
  patterns.add<EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::AddOp>>(typeConverter, ctx);
  // clang-format on

  // Memory ops
  //
  // clang-format off
  patterns.add<ToDeviceOpConversionPattern,
               DeallocateOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Other ops
  //
  patterns.add<SoftmaxOpConversionPattern>(typeConverter, ctx);

  // Module op
  //
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
