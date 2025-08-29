// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "mlir/IR/Attributes.h"
#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

using namespace mlir;
using namespace mlir::tt;
using ttnn_to_emitpy::operator|;

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseUnaryOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseUnaryOp.getInput()),
        emitter.emit(std::nullopt, "memory_config") |
            emitter.getMemoryConfig(eltwiseUnaryOp.getResult()),
    };

    emitter.replaceOp(*this, args);

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseBinaryOp,
                                                        adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseBinaryOp.getLhs()),
        emitter.emit(eltwiseBinaryOp.getRhs()),
        emitter.emit(eltwiseBinaryOp.getDtype(), "dtype"),
        emitter.emit(std::nullopt, "memory_config") |
            emitter.getMemoryConfig(eltwiseBinaryOp.getResult()),
    };

    emitter.replaceOp(*this, args);

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::MatmulOp> emitter(
        matmulOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(matmulOp.getA()),
        emitter.emit(matmulOp.getB()),
        emitter.emit(matmulOp.getTransposeA(), "transpose_a"),
        emitter.emit(matmulOp.getTransposeB(), "transpose_b"),
        emitter.emit(std::nullopt, "memory_config") |
            emitter.getMemoryConfig(matmulOp.getResult()),
    };

    emitter.replaceOp(*this, args);

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::SoftmaxOp> emitter(
        softmaxOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(softmaxOp.getInput()),
        emitter.emit(softmaxOp.getDimension()),
        emitter.emit(std::nullopt, "memory_config") |
            emitter.getMemoryConfig(softmaxOp.getResult()),
    };

    emitter.replaceOp(*this, args);

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::DeallocateOp> emitter(
        deallocateOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(deallocateOp.getInput()),
        emitter.emit(deallocateOp.getForce()),
    };

    emitter.replaceOp(*this, args);

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::GetDeviceOp> emitter(
        getDeviceOp, adaptor, rewriter);

    emitter.replaceOp(*this, {});

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

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToDeviceOp> emitter(
        toDeviceOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(toDeviceOp.getInput()),
        emitter.emit(toDeviceOp.getDevice(), "device"),
        emitter.emit(toDeviceOp.getMemoryConfig(), "memory_config") |
            emitter.getMemoryConfig(toDeviceOp.getResult()),
    };

    emitter.replaceOp(*this, args);

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
  matchAndRewrite(SourceOp namedFullOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<SourceOp> emitter(namedFullOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(namedFullOp.getShape(), "shape"),
        emitter.emit(namedFullOp.getDtype(), "dtype"),
        emitter.emit(namedFullOp.getLayout(), "layout"),
        emitter.emit(namedFullOp.getDevice(), "device"),
        emitter.emit(namedFullOp.getMemoryConfig(), "memory_config") |
            emitter.getMemoryConfig(namedFullOp.getResult()),
    };

    emitter.replaceOp(*this, args);

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
