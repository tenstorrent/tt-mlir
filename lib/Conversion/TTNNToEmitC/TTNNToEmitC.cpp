// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

emitc::OpaqueAttr createStdNullopt(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>("std::nullopt");
}

emitc::OpaqueAttr convertLayoutAttr(Builder &builder, ttnn::LayoutAttr attr) {
  switch (attr.getValue()) {
  case ttnn::Layout::RowMajor:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::ROW_MAJOR");
  case ttnn::Layout::Tile:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::TILE");
  case ttnn::Layout::Invalid:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::INVALID");
  }
  llvm_unreachable("Unknown ttnn::TensorMemoryLayout");
  return nullptr;
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

// MultiplyOp conversion pattern
//
// TODO(bug #623):
// Convert all DPS-supported ttnn ops to this conversion pattern (nullopts added
// for correct signature)
//
class MultiplyOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::MultiplyOp> {

public:
  MultiplyOpConversionPattern(const TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::MultiplyOp>(typeConverter,
                                                             context, benefit) {
  }

  LogicalResult
  matchAndRewrite(ttnn::MultiplyOp srcOp, OpAdaptor adaptor,
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

public:
  GetDeviceOpConversionPattern(const TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::GetDeviceOp>(
            typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ttnn::GetDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy =
        this->getTypeConverter()->convertType(srcOp->getResult(0).getType());

    rewriter.replaceOpWithNewOp<emitc::VariableOp>(
        srcOp, resultTy, rewriter.getAttr<emitc::OpaqueAttr>("device"));

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

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), nullptr, nullptr, adaptor.getOperands());

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

    llvm::StringRef amp = llvm::StringRef("&");
    emitc::ApplyOp ampApplyOp = rewriter.create<emitc::ApplyOp>(
        srcOp->getLoc(),
        emitc::PointerType::get(
            emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Device")),
        amp, adaptor.getOperands().back());

    llvm::SmallVector<Value, 4> operands(adaptor.getOperands().begin(),
                                         adaptor.getOperands().end());
    operands.pop_back(); // erase device
    operands.push_back(ampApplyOp);

    llvm::SmallVector<Attribute, 5> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(convertLayoutAttr(rewriter, srcOp.getLayoutAttr()));
    attrs.push_back(createStdNullopt(rewriter));
    attrs.push_back(createStdNullopt(rewriter));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 1));

    ArrayAttr arrayAttrs = ArrayAttr::get(srcOp->getContext(), attrs);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, operands);

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
  patterns
      .add<ToLayoutOpConversionPattern,
           DefaultOpConversionPattern<ttnn::ToMemoryConfigOp>,
           ToDeviceOpConversionPattern /*, MemoryConfigOpConversionPattern*/>(
          typeConverter, ctx);

  // Tensor ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::EmptyOp>,
               DefaultOpConversionPattern<ttnn::FullOp>>(typeConverter, ctx);

  // Eltwise unary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AbsOp>,
               DefaultOpConversionPattern<ttnn::ReluOp>,
               DefaultOpConversionPattern<ttnn::SqrtOp>,
               DefaultOpConversionPattern<ttnn::SigmoidOp>,
               DefaultOpConversionPattern<ttnn::ReciprocalOp>,
               DefaultOpConversionPattern<ttnn::ExpOp>>(typeConverter, ctx);

  // Eltwise binary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AddOp>,
               DefaultOpConversionPattern<ttnn::SubtractOp>,
               MultiplyOpConversionPattern,
               DefaultOpConversionPattern<ttnn::GreaterEqualOp>,
               DefaultOpConversionPattern<ttnn::MaximumOp>,
               DefaultOpConversionPattern<ttnn::DivOp>>(typeConverter, ctx);

  // Tensor manipulation ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::TransposeOp>,
               DefaultOpConversionPattern<ttnn::ConcatOp>,
               DefaultOpConversionPattern<ttnn::ReshapeOp>>(typeConverter, ctx);

  // Matmul ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::MatmulOp>>(typeConverter, ctx);

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
               DefaultOpConversionPattern<ttnn::EmbeddingOp>>(typeConverter,
                                                              ctx);
}

} // namespace mlir::tt
