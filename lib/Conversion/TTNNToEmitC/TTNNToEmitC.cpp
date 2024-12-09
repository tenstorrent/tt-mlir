// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

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

// Create emitc::OpaqueAttr for std::nullopt
//
emitc::OpaqueAttr createStdNullopt(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>("std::nullopt");
}

emitc::OpaqueAttr createNullDevicePointer(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>(
      "static_cast<::ttnn::Device *>(nullptr)");
}

// Create emitc::OpaqueAttr for ttnn::Layout
//
emitc::OpaqueAttr convertLayoutAttr(Builder &builder, ttnn::LayoutAttr attr) {
  switch (attr.getValue()) {
  case ttnn::Layout::RowMajor:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::ROW_MAJOR");
  case ttnn::Layout::Tile:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::TILE");
  case ttnn::Layout::Invalid:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::INVALID");
  }

  llvm_unreachable("Unknown ttnn::Layout");
}

emitc::OpaqueAttr convertBoolAttr(Builder &builder, BoolAttr attr) {
  return builder.getType<emitc::OpaqueAttr>(attr.getValue() ? "true" : "false");
}

// Create emitc::OpaqueAttr for ttnn::TensorMemoryLayout
//
emitc::OpaqueAttr convertTensorMemoryLayout(Builder &builder,
                                            ttnn::TensorMemoryLayoutAttr attr) {
  switch (attr.getValue()) {
  case ttnn::TensorMemoryLayout::BlockSharded:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::BLOCK_SHARDED");
  case ttnn::TensorMemoryLayout::HeightSharded:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::HEIGHT_SHARDED");
  case ttnn::TensorMemoryLayout::Interleaved:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::INTERLEAVED");
  case ttnn::TensorMemoryLayout::SingleBank:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::SINGLE_BANK");
  case ttnn::TensorMemoryLayout::WidthSharded:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::WIDTH_SHARDED");
  }
}

// Create emitc::OpaqueAttr for ttnn::BufferType
//
emitc::OpaqueAttr convertBufferType(Builder &builder,
                                    ttnn::BufferTypeAttr attr) {
  switch (attr.getValue()) {
  case ttnn::BufferType::DRAM:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::DRAM");
  case ttnn::BufferType::L1:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::L1");
  case ttnn::BufferType::L1Small:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::L1_SMALL");
  case ttnn::BufferType::SystemMemory:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::BufferType::SYSTEM_MEMORY");
  case ttnn::BufferType::Trace:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::TRACE");
  }

  llvm_unreachable("Unknown ttnn::BufferType");
}

// Create emitc::OpaqueAttr for ttnn::Shape
//
emitc::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr) {
  llvm::ArrayRef shape = attr.getShape();
  std::ostringstream oss;
  std::copy(shape.begin(), shape.end(), std::ostream_iterator<int>(oss, ", "));
  return builder.getType<emitc::OpaqueAttr>("{" + oss.str() + "}");
}

// Create emitc::OpaqueAttr for ttnn::DataType
//
emitc::OpaqueAttr convertDType(Builder &builder, tt::DataTypeAttr attr) {
  switch (attr.getValue()) {
  case tt::DataType::BFloat16:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT16");
  case tt::DataType::Float32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::FLOAT32");
  case tt::DataType::UInt32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT32");
  case tt::DataType::BFP_BFloat8:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT8_B");
  case tt::DataType::BFP_BFloat4:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT4_B");
  case tt::DataType::UInt8:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT8");
  case tt::DataType::UInt16:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT16");
  // TODO(svuckovic):
  // Add support for INT32
  //
  // case tt::DataType::Int32:
  //   return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::INT32");
  case tt::DataType::Float16:
  case tt::DataType::BFP_Float2:
  case tt::DataType::BFP_Float4:
  case tt::DataType::BFP_Float8:
  case tt::DataType::BFP_BFloat2:
    llvm_unreachable("Unsupported ttnn::DataType");
  }

  llvm_unreachable("Unkonwn tt::DataType");
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
    attrs.push_back(createStdNullopt(rewriter));
    attrs.push_back(createStdNullopt(rewriter));
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
          {convertTensorMemoryLayout(
               rewriter, srcOp.getMemoryConfig()->getTensorMemoryLayout()),
           convertBufferType(rewriter,
                             srcOp.getMemoryConfig()->getBufferType())});

      // Create MemoryConfig object first, then pass it to the op
      //
      emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
          srcOp->getLoc(),
          emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
          "ttnn::MemoryConfig", arrayAttrs, nullptr, ValueRange());
      operands.append(1, memCfgOp.getResult(0));
      attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 2));
    } else {
      attrs.push_back(createStdNullopt(rewriter));
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
        {convertTensorMemoryLayout(
             rewriter, srcOp.getMemoryConfig().getTensorMemoryLayout()),
         convertBufferType(rewriter, srcOp.getMemoryConfig().getBufferType())});

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
    attrs.push_back(createStdNullopt(rewriter));

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
    attrs.push_back(convertLayoutAttr(rewriter, srcOp.getLayoutAttr()));
    attrs.push_back(createStdNullopt(rewriter));
    attrs.push_back(createStdNullopt(rewriter));
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
    // ttnn:Shape has a couple constructors, but they are explicit and require
    // specific datatypes on input. However, one of the constructors takes in a
    // tt_metal::Shape - given that it's much easier to construct a
    // tt_metal::Shape, we opted to do that here. The call looks like this:
    // ttnn::Shape(tt::tt_metal::LegacyShape{dim0, dim1, dim2, ...});
    //
    // To make it easier on the eyes, these two calls are packed into one, using
    // EmitC's ExpressionOp.
    //
    emitc::ExpressionOp shapeExpressionOp =
        rewriter.create<emitc::ExpressionOp>(
            srcOp->getLoc(),
            emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Shape"),
            false);
    mlir::Block &bodyBlock = shapeExpressionOp.getBodyRegion().emplaceBlock();
    auto currentPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPointToStart(&bodyBlock);
    emitc::CallOpaqueOp metalShapeOp = rewriter.create<emitc::CallOpaqueOp>(
        srcOp->getLoc(),
        emitc::OpaqueType::get(rewriter.getContext(),
                               "tt::tt_metal::LegacyShape"),
        rewriter.getStringAttr("tt::tt_metal::LegacyShape"),
        rewriter.getArrayAttr(convertShape(rewriter, shapeAttr)), nullptr,
        ValueRange());
    emitc::CallOpaqueOp ttnnShapeOp = rewriter.create<emitc::CallOpaqueOp>(
        srcOp->getLoc(),
        emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Shape"),
        rewriter.getStringAttr("ttnn::Shape"), nullptr, nullptr,
        metalShapeOp->getResults());
    rewriter.create<emitc::YieldOp>(srcOp->getLoc(), ttnnShapeOp->getResult(0));
    rewriter.setInsertionPoint(srcOp->getBlock(), currentPoint);

    llvm::SmallVector<Value, 3> operands{
        shapeExpressionOp->getResult(0),
    };

    // If there is a device operand, create tensor on device
    //
    ArrayAttr arrayAttr;
    if (adaptor.getDevice()) {
      operands.append(1, adaptor.getDevice());

      // Create ArrayAttr object holding MemoryConfig attributes
      //
      ArrayAttr memCfgArrayAttrs = rewriter.getArrayAttr(
          {convertTensorMemoryLayout(
               rewriter, srcOp.getMemoryConfig()->getTensorMemoryLayout()),
           convertBufferType(rewriter,
                             srcOp.getMemoryConfig()->getBufferType())});

      // Create MemoryConfig object first, then pass it to the op
      //
      emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
          srcOp->getLoc(),
          emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
          "ttnn::MemoryConfig", memCfgArrayAttrs, nullptr, ValueRange());

      // Concat operands and MemoryConfig object
      //
      operands.append(1, memCfgOp.getResult(0));

      // Create ArrayAttr object holding attributes and pointers to operands
      //
      arrayAttr = rewriter.getArrayAttr({
          rewriter.getIndexAttr(0), // ttnn::Shape
          convertDType(rewriter, dataTypeAttr),
          convertLayoutAttr(rewriter, layoutAttr),
          rewriter.getIndexAttr(1), // ttnn::Device
          rewriter.getIndexAttr(2), // ttnn::MemoryConfig
      });
    } else {
      arrayAttr = rewriter.getArrayAttr({
          rewriter.getIndexAttr(0), // ttnn::Shape
          convertDType(rewriter, dataTypeAttr),
          convertLayoutAttr(rewriter, layoutAttr),
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
        convertBoolAttr(rewriter, srcOp.getForceAttr()),
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
  patterns.add<ToLayoutOpConversionPattern, ToMemoryConfigOpConversionPattern,
               TypecastOpConversionPattern, ToDeviceOpConversionPattern,
               FromDeviceOpConversionPattern, DeallocateOpConversionPattern>(
      typeConverter, ctx);

  // Tensor ops
  //
  patterns
      .add<EmptyOpConversionPattern, DefaultOpConversionPattern<ttnn::FullOp>,
           DefaultOpConversionPattern<ttnn::ArangeOp>>(typeConverter, ctx);

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
               DefaultOpConversionPattern<ttnn::WhereOp>>(typeConverter, ctx);

  // CCL ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AllGatherOp>>(typeConverter,
                                                              ctx);

  // Module op
  //
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);

  // KV Cache ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::UpdateCacheOp>>(typeConverter,
                                                                ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::FillCacheOp>>(typeConverter,
                                                              ctx);

  // Arith ops
  //
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
