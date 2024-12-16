// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

emitc::OpaqueAttr createNullDevicePointer(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>(
      "static_cast<::ttnn::Device *>(nullptr)");
}

// Base class for TTNN to EmitC OpConversionPattern
//
template <typename SourceOp>
class TTNNToEmitCBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  TTNNToEmitCBaseOpConversionPattern(const TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit) {}

private:
  std::string virtual getPrefixSearchPattern() const { return "ttnn."; }
  std::string virtual getPrefixSwapPattern() const { return "ttnn::"; }

public:
  // Converts op name by removing the dialect prefix ("ttnn.") and replacing
  // with namespace prefix ("ttnn::")
  //
  std::string convertOpName(SourceOp op) const {
    auto name = op.getOperationName();
    assert(
        name.starts_with(getPrefixSearchPattern()) &&
        "DefaultOpConversionPattern only supports ops from the TTNN dialect");

    return name.str().replace(0, getPrefixSearchPattern().size(),
                              getPrefixSwapPattern());
  }
};

// Default op conversion pattern, used to convert most ops
//
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class DefaultOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  DefaultOpConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<SourceOp>(typeConverter, context,
                                                     benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int numReturnTypes = srcOp->getResultTypes().size();
    assert(numReturnTypes <= 1 &&
           "DefaultOpConversionPattern does not support multiple return types");

    // If srcOp has a return type, cast it before converting
    //
    if (numReturnTypes == 1) {
      auto resultTy = cast<emitc::OpaqueType>(
          this->getTypeConverter()->convertType(srcOp->getResult(0).getType()));
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, resultTy, this->convertOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    } else {
      // No return type, only convert the op
      //
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp), nullptr,
          nullptr, adaptor.getOperands());
    }

    return success();
  }
};

// Eltwise Binary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (output dtype, memcfg)
//
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class EltwiseBinaryOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  EltwiseBinaryOpConversionPattern(const TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<SourceOp>(typeConverter, context,
                                                     benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this
    //
    llvm::SmallVector<Attribute, 5> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 1));
    attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 2));

    ArrayAttr arrayAttrs = ArrayAttr::get(srcOp->getContext(), attrs);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType(0)),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// GetDeviceOp conversion pattern
//
class GetDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::GetDeviceOp> {

private:
  std::string getPrefixSearchPattern() const override {
    return "ttnn.get_device";
  }
  std::string getPrefixSwapPattern() const override {
    return "ttnn::DeviceGetter::getInstance";
  }

public:
  GetDeviceOpConversionPattern(const TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::GetDeviceOp>(
            typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::GetDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), nullptr, nullptr, adaptor.getOperands());

    return success();
  }
};

// ToDeviceOp conversion pattern
//
class ToDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ToDeviceOp> {

public:
  ToDeviceOpConversionPattern(const TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::ToDeviceOp>(typeConverter,
                                                             context, benefit) {
  }

  LogicalResult
  matchAndRewrite(ttnn::ToDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<Attribute, 2> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 1));
    llvm::SmallVector<Value, 2> operands(adaptor.getOperands());

    if (srcOp.getMemoryConfig()) {
      // Create ArrayAttr object holding MemoryConfig attributes
      //
      ArrayAttr arrayAttrs = rewriter.getArrayAttr(
          {ttnn_to_emitc::utils::convertTensorMemoryLayout(
               rewriter, srcOp.getMemoryConfig()->getTensorMemoryLayout()),
           ttnn_to_emitc::utils::convertBufferType(
               rewriter, srcOp.getMemoryConfig()->getBufferType())});

      // Create MemoryConfig object first, then pass it to the op
      //
      emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
          srcOp->getLoc(),
          emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
          "ttnn::MemoryConfig", arrayAttrs, nullptr, ValueRange());
      operands.append(1, memCfgOp.getResult(0));
      attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 2));
    } else {
      attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    }

    ArrayAttr finalAttrs = ArrayAttr::get(srcOp->getContext(), attrs);
    // Concat operands and MemoryConfig object
    //

    // Convert ToDeviceOp
    //
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), finalAttrs, nullptr, operands);

    return success();
  }
};

// FromDeviceOp conversion pattern
//
class FromDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::FromDeviceOp> {

public:
  FromDeviceOpConversionPattern(const TypeConverter &typeConverter,
                                MLIRContext *context,
                                PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::FromDeviceOp>(
            typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::FromDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), nullptr, nullptr, adaptor.getOperands());

    return success();
  }
};

// TypecastOp conversion pattern
//
class TypecastOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::TypecastOp> {

public:
  TypecastOpConversionPattern(const TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::TypecastOp>(typeConverter,
                                                             context, benefit) {
  }

  LogicalResult
  matchAndRewrite(ttnn::TypecastOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), nullptr, nullptr, adaptor.getOperands());

    return success();
  }
};

// ToMemoryConfig conversion pattern
//
class ToMemoryConfigOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ToMemoryConfigOp> {

public:
  ToMemoryConfigOpConversionPattern(const TypeConverter &typeConverter,
                                    MLIRContext *context,
                                    PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::ToMemoryConfigOp>(
            typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::ToMemoryConfigOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create ArrayAttr object holding MemoryConfig attributes
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {ttnn_to_emitc::utils::convertTensorMemoryLayout(
             rewriter, srcOp.getMemoryConfig().getTensorMemoryLayout()),
         ttnn_to_emitc::utils::convertBufferType(
             rewriter, srcOp.getMemoryConfig().getBufferType())});

    // Create MemoryConfig object first, then pass it to the op
    //
    emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
        srcOp->getLoc(),
        emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
        "ttnn::MemoryConfig", arrayAttrs, nullptr, ValueRange());

    // Concat operands and MemoryConfig object
    //
    llvm::SmallVector<Value, 2> operands(adaptor.getOperands());
    operands.append(1, memCfgOp.getResult(0));

    llvm::SmallVector<Attribute, 3> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 1));
    attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));

    ArrayAttr finalAttrs = ArrayAttr::get(srcOp->getContext(), attrs);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), finalAttrs, nullptr, operands);

    return success();
  }
};

// ToLayoutOp conversion pattern
//
class ToLayoutOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ToLayoutOp> {

public:
  ToLayoutOpConversionPattern(const TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::ToLayoutOp>(typeConverter,
                                                             context, benefit) {
  }

  LogicalResult
  matchAndRewrite(ttnn::ToLayoutOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<Attribute, 5> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(ttnn_to_emitc::utils::convertLayoutAttr(
        rewriter, srcOp.getLayoutAttr()));
    attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    attrs.push_back(createNullDevicePointer(rewriter));

    ArrayAttr arrayAttrs = ArrayAttr::get(srcOp->getContext(), attrs);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// EmptyOp conversion pattern
//
class EmptyOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::EmptyOp> {

public:
  EmptyOpConversionPattern(const TypeConverter &typeConverter,
                           MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::EmptyOp>(typeConverter,
                                                          context, benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::EmptyOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn::ShapeAttr shapeAttr = srcOp.getShapeAttr();
    tt::DataTypeAttr dataTypeAttr = srcOp.getDtypeAttr();
    ttnn::LayoutAttr layoutAttr = srcOp.getLayoutAttr();

    // Create ttnn::Shape() call
    //
    emitc::ExpressionOp shapeExpressionOp = ttnn_to_emitc::utils::createShapeOp(
        rewriter, shapeAttr, srcOp->getBlock(), srcOp.getLoc());

    llvm::SmallVector<Value, 3> operands{
        shapeExpressionOp->getResult(0),
    };

    // If there is a device operand, create tensor on device
    //
    ArrayAttr arrayAttr;
    if (adaptor.getDevice()) {
      operands.append(1, adaptor.getDevice());

      // Create MemoryConfig object first, then pass it to the op
      //
      emitc::CallOpaqueOp memCfgOp = ttnn_to_emitc::utils::createMemoryConfigOp(
          rewriter, srcOp.getMemoryConfig().value(), srcOp.getLoc());

      // Concat operands and MemoryConfig object
      //
      operands.append(1, memCfgOp.getResult(0));

      // Create ArrayAttr object holding attributes and pointers to operands
      //
      arrayAttr = rewriter.getArrayAttr({
          rewriter.getIndexAttr(0), // ttnn::Shape
          ttnn_to_emitc::utils::convertDType(rewriter, dataTypeAttr),
          ttnn_to_emitc::utils::convertLayoutAttr(rewriter, layoutAttr),
          rewriter.getIndexAttr(1), // ttnn::Device
          rewriter.getIndexAttr(2), // ttnn::MemoryConfig
      });
    } else {
      arrayAttr = rewriter.getArrayAttr({
          rewriter.getIndexAttr(0), // ttnn::Shape
          ttnn_to_emitc::utils::convertDType(rewriter, dataTypeAttr),
          ttnn_to_emitc::utils::convertLayoutAttr(rewriter, layoutAttr),
      });
    }

    // Finally, convert ttir::EmptyOp to ttnn::EmptyOp
    //
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttr, nullptr, operands);

    return success();
  }
};

// OnesOp conversion pattern
//
class OnesOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::OnesOp> {

public:
  OnesOpConversionPattern(const TypeConverter &typeConverter,
                          MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::OnesOp>(typeConverter, context,
                                                         benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::OnesOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // ttnn:OnesOp has 5 input params:
    //
    // let arguments = (ins TTNN_ShapeAttr:$shape,
    //                      OptionalAttr<TT_DataTypeAttr>:$dtype,
    //                      OptionalAttr<TTNN_LayoutAttr>:$layout,
    //                      Optional<TT_Device>:$device,
    //                      OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    //
    // Some of them are Attrs, some are Values. ShapeAttr is required, while
    // others are optional. Additionally, in the context of C++, some of the
    // Attrs (like shape) need to be instantiated into objects before being
    // passed to the op. Therefore:
    //
    // We first create a ttnn::Shape object (SSA) by calling createShapeOp() and
    // add it to the operands vector, but also add an IndexAttr in ArrayAttr to
    // reference it (this is an EmitC mechanism that allows for combining Attrs
    // and Values when calling an OpaqueOp).
    // All the other input params are optional, so we create them on-the-fly
    // into the ArrayAttr, whether they are an actual Attr, or a Value pointed
    // to by IndexAttr. If they are present, we create the object and pass it to
    // the op. If not, we pass std::nullopt.

    // Create ttnn::Shape() call
    //
    emitc::ExpressionOp shapeExpressionOp = ttnn_to_emitc::utils::createShapeOp(
        rewriter, srcOp.getShapeAttr(), srcOp->getBlock(), srcOp.getLoc());

    llvm::SmallVector<Value, 3> operands{
        shapeExpressionOp->getResult(0),
    };

    // Create ArrayAttr object holding attributes and pointers to operands
    //
    // Params that are Values are added to the operands vector on-the-fly, and a
    // corresponding IndexAttr is added to the ArrayAttr to reference them.
    //
    size_t operandIndex = 0;
    ArrayAttr arrayAttr = rewriter.getArrayAttr({
        rewriter.getIndexAttr(operandIndex++), // ttnn::Shape
        srcOp.getDtype().has_value()
            ? ttnn_to_emitc::utils::convertDType(rewriter, srcOp.getDtypeAttr())
            : ttnn_to_emitc::utils::createStdNullopt(
                  rewriter), // ttnn::DataType
        srcOp.getLayout().has_value()
            ? ttnn_to_emitc::utils::convertLayoutAttr(rewriter,
                                                      srcOp.getLayoutAttr())
            : ttnn_to_emitc::utils::createStdNullopt(rewriter), // ttnn::Layout
        adaptor.getDevice()
            ? (operands.append(1, adaptor.getDevice()),
               mlir::cast<Attribute>(rewriter.getIndexAttr(operandIndex++)))
            : ttnn_to_emitc::utils::createStdNullopt(rewriter), // ttnn::Device
        srcOp.getMemoryConfig().has_value()
            ? (operands.append(
                   1, ttnn_to_emitc::utils::createMemoryConfigOp(
                          rewriter, srcOp.getMemoryConfigAttr(), srcOp.getLoc())
                          ->getResult(0)),
               mlir::cast<Attribute>(rewriter.getIndexAttr(operandIndex++)))
            : ttnn_to_emitc::utils::createStdNullopt(
                  rewriter), // ttnn::MemoryConfig
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttr, nullptr, operands);

    return success();
  }
};

// DeallocateOp conversion pattern
//
class DeallocateOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::DeallocateOp> {

public:
  DeallocateOpConversionPattern(const TypeConverter &typeConverter,
                                MLIRContext *context,
                                PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::DeallocateOp>(
            typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::DeallocateOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ArrayAttr arrayAttr = rewriter.getArrayAttr({
        rewriter.getIndexAttr(0),
        ttnn_to_emitc::utils::convertBoolAttr(rewriter, srcOp.getForceAttr()),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp), arrayAttr,
        nullptr, adaptor.getOperands());

    return success();
  }
};

// arith::ConstantOp conversion pattern
//
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

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, newTy,
                                                   adaptor.getValue());
    return success();
  }
};

class GetTupleElementOpConversionPattern
    : public OpConversionPattern<tt::GetTupleElementOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::GetTupleElementOp getTupleElementOp,
                  tt::GetTupleElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SubscriptOp requires a Value object as index, which is created by
    // invoking the emitc::LiteralOp
    //
    Value indexAsVal = rewriter.create<emitc::LiteralOp>(
        getTupleElementOp->getLoc(), rewriter.getIndexType(),
        std::to_string(adaptor.getIndex()));

    // SubscriptOp also returns an emitc::LValueType, so we wrap the OpaqueType
    // with LValueType
    //
    emitc::LValueType lvalueReturnType = emitc::LValueType::get(
        emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Tensor"));
    Value subscript = rewriter.create<emitc::SubscriptOp>(
        getTupleElementOp->getLoc(), lvalueReturnType, adaptor.getOperand(),
        indexAsVal);

    // As SubscriptOp returns an LValueType, we need to convert it to an
    // OpaqueType - this is done by invoking the emitc::LoadOp
    //
    rewriter.replaceOpWithNewOp<emitc::LoadOp>(
        getTupleElementOp, emitc::OpaqueType::get(getContext(), "ttnn::Tensor"),
        subscript);
    return success();
  }
};

// Module Op conversion pattern
//
// This conversion pattern removes attributes from the ModuleOp. Previously,
// ttmlir-translate would complain when translating to C++ if there were any
// attributes from "unregistered" dialects.
//
class ModuleOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<mlir::ModuleOp> {

public:
  ModuleOpConversionPattern(const TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<mlir::ModuleOp>(typeConverter,
                                                           context, benefit) {}

  LogicalResult
  matchAndRewrite(mlir::ModuleOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.modifyOpInPlace(srcOp, [&]() {
      for (const NamedAttribute &attr : srcOp->getAttrs()) {
        srcOp->removeAttr(attr.getName());
      }
    });

    return success();
  }
};

} // namespace

namespace mlir::tt {

void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // Device ops
  //
  patterns.add<GetDeviceOpConversionPattern>(typeConverter, ctx);

  // Memory ops
  //
  // clang-format off
  patterns.add<ToLayoutOpConversionPattern,
               ToMemoryConfigOpConversionPattern,
               TypecastOpConversionPattern,
               ToDeviceOpConversionPattern,
               FromDeviceOpConversionPattern,
               DeallocateOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Tensor ops
  //
  // clang-format off
  patterns.add<EmptyOpConversionPattern,
               OnesOpConversionPattern,
               DefaultOpConversionPattern<ttnn::FullOp>,
               DefaultOpConversionPattern<ttnn::ArangeOp>>(typeConverter, ctx);
  // clang-format on

  // Eltwise unary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AbsOp>,
               DefaultOpConversionPattern<ttnn::CbrtOp>,
               DefaultOpConversionPattern<ttnn::ClampOp>,
               DefaultOpConversionPattern<ttnn::FloorOp>,
               DefaultOpConversionPattern<ttnn::IsFiniteOp>,
               DefaultOpConversionPattern<ttnn::LogicalNotOp>,
               DefaultOpConversionPattern<ttnn::NegOp>,
               DefaultOpConversionPattern<ttnn::ReluOp>,
               DefaultOpConversionPattern<ttnn::LeakyReluOp>,
               DefaultOpConversionPattern<ttnn::GeluOp>,
               DefaultOpConversionPattern<ttnn::SqrtOp>,
               DefaultOpConversionPattern<ttnn::RsqrtOp>,
               DefaultOpConversionPattern<ttnn::SignOp>,
               DefaultOpConversionPattern<ttnn::SigmoidOp>,
               DefaultOpConversionPattern<ttnn::Log1pOp>,
               DefaultOpConversionPattern<ttnn::ReciprocalOp>,
               DefaultOpConversionPattern<ttnn::ExpOp>,
               DefaultOpConversionPattern<ttnn::CeilOp>,
               DefaultOpConversionPattern<ttnn::SinOp>,
               DefaultOpConversionPattern<ttnn::CosOp>,
               DefaultOpConversionPattern<ttnn::Expm1Op>,
               DefaultOpConversionPattern<ttnn::TanOp>,
               DefaultOpConversionPattern<ttnn::TanhOp>,
               DefaultOpConversionPattern<ttnn::LogOp>>(typeConverter, ctx);

  // Eltwise binary ops
  //
  patterns.add<EltwiseBinaryOpConversionPattern<ttnn::AddOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LogicalAndOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LogicalOrOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LogicalXorOp>,
               EltwiseBinaryOpConversionPattern<ttnn::SubtractOp>,
               EltwiseBinaryOpConversionPattern<ttnn::MultiplyOp>,
               DefaultOpConversionPattern<ttnn::EqualOp>,
               DefaultOpConversionPattern<ttnn::NotEqualOp>,
               DefaultOpConversionPattern<ttnn::GreaterEqualOp>,
               DefaultOpConversionPattern<ttnn::GreaterThanOp>,
               DefaultOpConversionPattern<ttnn::LessEqualOp>,
               DefaultOpConversionPattern<ttnn::LessThanOp>,
               DefaultOpConversionPattern<ttnn::MaximumOp>,
               DefaultOpConversionPattern<ttnn::MinimumOp>,
               DefaultOpConversionPattern<ttnn::DivOp>,
               DefaultOpConversionPattern<ttnn::ScatterOp>,
               DefaultOpConversionPattern<ttnn::RemainderOp>>(typeConverter,
                                                              ctx);

  // Tensor manipulation ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::TransposeOp>,
               DefaultOpConversionPattern<ttnn::ConcatOp>,
               DefaultOpConversionPattern<ttnn::ReshapeOp>,
               DefaultOpConversionPattern<ttnn::SliceOp>>(typeConverter, ctx);

  // Matmul ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::LinearOp>,
               DefaultOpConversionPattern<ttnn::MatmulOp>>(typeConverter, ctx);

  // Reduction ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::SumOp>,
               DefaultOpConversionPattern<ttnn::MeanOp>,
               DefaultOpConversionPattern<ttnn::MaxOp>>(typeConverter, ctx);

  // Conv ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::Conv2dOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MaxPool2dOp>>(typeConverter,
                                                              ctx);

  // Other ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::SoftmaxOp>,
               DefaultOpConversionPattern<ttnn::EmbeddingOp>,
               DefaultOpConversionPattern<ttnn::EmbeddingBackwardOp>,
               DefaultOpConversionPattern<ttnn::WhereOp>>(typeConverter, ctx);

  // CCL ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AllGatherOp>>(typeConverter,
                                                              ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ReduceScatterOp>>(typeConverter,
                                                                  ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MeshShardOp>>(typeConverter,
                                                              ctx);

  // KV Cache ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::UpdateCacheOp>>(typeConverter,
                                                                ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::FillCacheOp>>(typeConverter,
                                                              ctx);

  // Arith ops
  //
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);

  // Module op
  //
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);

  // Tuple ops
  //
  patterns.add<GetTupleElementOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
