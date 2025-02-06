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

emitc::OpaqueAttr createNullDevicePointer(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>(
      "static_cast<::ttnn::IDevice *>(nullptr)");
}

// Base class for TTNN to EmitC OpConversionPattern.
//
namespace {
template <typename SourceOp>
class TTNNToEmitCBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

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
} // namespace

// Default op conversion pattern, used to convert most ops.
//
namespace {
template <typename SourceOp>
class DefaultOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int numReturnTypes = srcOp->getResultTypes().size();
    assert(numReturnTypes <= 1 &&
           "DefaultOpConversionPattern does not support multiple return types");

    // If srcOp has a return type, cast it before converting.
    //
    if (numReturnTypes == 1) {
      auto resultTy = cast<emitc::OpaqueType>(
          this->getTypeConverter()->convertType(srcOp->getResult(0).getType()));
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, resultTy, this->convertOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    } else {
      // No return type, only convert the op.
      //
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp), nullptr,
          nullptr, adaptor.getOperands());
    }

    return success();
  }
};
} // namespace

// Eltwise Unary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseUnaryOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    llvm::SmallVector<Attribute, 5> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 1));

    ArrayAttr arrayAttrs = ArrayAttr::get(srcOp->getContext(), attrs);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType(0)),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};
} // namespace

// EltwiseUnaryWithFastAndApproximateModeOp conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (parameter, memcfg).
//
template <typename SourceOp>
class EltwiseUnaryWithFastAndApproximateModeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
         ttnn_to_emitc::utils::convertBoolAttr(
             rewriter, BoolAttr::get(rewriter.getContext(), false)),
         ttnn_to_emitc::utils::createStdNullopt(rewriter),
         mlir::IntegerAttr::get(rewriter.getIndexType(), 1)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType(0)),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// EltwiseUnaryCompositeOp conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (parameter, memcfg).
//
template <typename SourceOp>
class EltwiseUnaryCompositeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
         ttnn_to_emitc::utils::createStdNullopt(rewriter)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType(0)),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// Eltwise Binary op conversion pattern
//
// Currently, it has to insert nullopts for some parameters that are not
// modelled in the dialect (output dtype, memcfg).
//
namespace {
template <typename SourceOp>
class EltwiseBinaryOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      SourceOp>::TTNNToEmitCBaseOpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

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
} // namespace

// Linear op conversion pattern
//
class LinearOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::LinearOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::LinearOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::LinearOp linearOp, ttnn::LinearOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr({
        mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
        mlir::IntegerAttr::get(rewriter.getIndexType(), 1),
        mlir::IntegerAttr::get(rewriter.getIndexType(), 2),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        linearOp, this->getTypeConverter()->convertType(linearOp.getType()),
        this->convertOpName(linearOp), arrayAttrs, nullptr,
        adaptor.getOperands());

    return success();
  }
};

// Matmul op conversion pattern
//
// ANCHOR: adding_an_op_matmul_op_rewriter_emitc
namespace {
class MatmulOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::MatmulOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::MatmulOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::MatmulOp matmulOp, ttnn::MatmulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // ANCHOR: adding_an_op_matmul_ttnn_to_emitc_array_attrs
    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr({
        mlir::IntegerAttr::get(rewriter.getIndexType(),
                               0), // points to operand 0
        mlir::IntegerAttr::get(rewriter.getIndexType(),
                               1), // points to operand 1
        ttnn_to_emitc::utils::convertBoolAttr(
            rewriter,
            BoolAttr::get(
                rewriter.getContext(),
                false)), // bool attr denoting transposeA is set to false
        ttnn_to_emitc::utils::convertBoolAttr(
            rewriter, BoolAttr::get(rewriter.getContext(), false)),
        ttnn_to_emitc::utils::createStdNullopt(rewriter), // std::nullopt
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        mlir::IntegerAttr::get(rewriter.getIndexType(),
                               2), // points to operand 2
    });
    // ANCHOR_END: adding_an_op_matmul_ttnn_to_emitc_array_attrs

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        matmulOp, this->getTypeConverter()->convertType(matmulOp.getType()),
        this->convertOpName(matmulOp), arrayAttrs, nullptr,
        adaptor.getOperands());

    return success();
  }
};
} // namespace
// ANCHOR_END: adding_an_op_matmul_op_rewriter_emitc

// Softmax op conversion pattern
//
class SoftmaxOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::SoftmaxOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::SoftmaxOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::SoftmaxOp softmaxOp, ttnn::SoftmaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr({
        mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
        softmaxOp.getDimensionAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        softmaxOp, this->getTypeConverter()->convertType(softmaxOp.getType()),
        this->convertOpName(softmaxOp), arrayAttrs, nullptr,
        adaptor.getOperands());

    return success();
  }
};

// Embedding op conversion pattern
//
class EmbeddingOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::EmbeddingOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::EmbeddingOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::EmbeddingOp embeddingOp,
                  ttnn::EmbeddingOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr({
        mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
        mlir::IntegerAttr::get(rewriter.getIndexType(), 1),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        embeddingOp,
        this->getTypeConverter()->convertType(embeddingOp.getType()),
        this->convertOpName(embeddingOp), arrayAttrs, nullptr,
        adaptor.getOperands());

    return success();
  }
};

// Moreh CumSum op conversion pattern
//
class MorehCumSumOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::MorehCumSumOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::MorehCumSumOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                  ttnn::MorehCumSumOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr({
        mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
        srcOp.getDimAttr(),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
        ttnn_to_emitc::utils::createStdNullopt(rewriter),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// MeanOp conversion pattern
//
class MeanOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::MeanOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::MeanOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::MeanOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {rewriter.getIndexAttr(0),
         srcOp.getDimArg().has_value()
             ? ttnn_to_emitc::utils::convertArrayAttrToTTNNSmallVector(
                   rewriter, srcOp.getDimArgAttr())
             : ttnn_to_emitc::utils::createStdNullopt(rewriter),
         ttnn_to_emitc::utils::convertBoolAttr(rewriter,
                                               srcOp.getKeepDimAttr())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// ReshapeOp conversion pattern
//
class ReshapeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ReshapeOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::ReshapeOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ReshapeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {rewriter.getIndexAttr(0), ttnn_to_emitc::utils::convertArrayAttrToSpan(
                                       rewriter, srcOp.getShapeAttr())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// TransposeOp conversion pattern
//
class TransposeOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::TransposeOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::TransposeOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::TransposeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // emitc::CallOpaqueOp needs to know positions of operands vs attributes, so
    // an ArrayAttr object holding IndexTypes is created to denote this.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {rewriter.getIndexAttr(0), srcOp.getDim0Attr(), srcOp.getDim1Attr()});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};

// ConcatOp conversion pattern
//
class ConcatOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ConcatOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::ConcatOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ConcatOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // ttnn::concat op requires a `std::vector<>` of `Tensor` objects, but we
    // can't really create a `std::vector<>` with `Value` objects without
    // introducing an EmitC op that takes in these `Value` objects. We do this
    // by creating a utility function within the IR that converts a list of
    // `Tensor` objects into a `std::vector<ttnn::Tensor>`.

    ttnn_to_emitc::utils::insertVecCreateFnIfNotExists(rewriter, srcOp);

    mlir::emitc::CallOpaqueOp vectorOp = rewriter.create<emitc::CallOpaqueOp>(
        srcOp.getLoc(),
        emitc::OpaqueType::get(rewriter.getContext(),
                               "std::vector<ttnn::Tensor>"),
        ttnn_to_emitc::utils::kCreateVectorFunctionName, nullptr, nullptr,
        adaptor.getInputs());

    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
         srcOp.getDimAttr()});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr,
        ValueRange(vectorOp->getResults()));

    return success();
  }
};

// Repeat op conversion pattern
//
class RepeatOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::RepeatOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::RepeatOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::RepeatOp repeatOp, ttnn::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn::ShapeAttr repeatDims = repeatOp.getRepeatDimsAttr();

    // Create ttnn::Shape() call
    //
    emitc::CallOpaqueOp shapeOp = ttnn_to_emitc::utils::createShapeOp(
        rewriter, repeatDims, repeatOp.getLoc());

    // Create operands vector
    //
    llvm::SmallVector<Value, 2> operands{
        adaptor.getOperands()[0], // input tensor
        shapeOp->getResult(0)};

    // Create ArrayAttr object holding attributes and pointers to operands
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr({
        rewriter.getIndexAttr(0), // input tensor
        rewriter.getIndexAttr(1)  // ttnn::Shape
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        repeatOp, this->getTypeConverter()->convertType(repeatOp.getType()),
        this->convertOpName(repeatOp), arrayAttrs, nullptr, operands);

    return success();
  }
};

// GetDeviceOp conversion pattern
//
namespace {
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
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::GetDeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::GetDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), nullptr, nullptr, adaptor.getOperands());

    return success();
  }
};
} // namespace

// ToDeviceOp conversion pattern
//
namespace {
class ToDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ToDeviceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::ToDeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ToDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<Attribute, 2> attrs;
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 1));
    llvm::SmallVector<Value, 2> operands(adaptor.getOperands());

    if (srcOp.getMemoryConfig()) {
      // Create ArrayAttr object holding MemoryConfig attributes.
      //
      ArrayAttr arrayAttrs = rewriter.getArrayAttr(
          {ttnn_to_emitc::utils::convertTensorMemoryLayout(
               rewriter, srcOp.getMemoryConfig()->getTensorMemoryLayout()),
           ttnn_to_emitc::utils::convertBufferType(
               rewriter, srcOp.getMemoryConfig()->getBufferType())});

      // Create MemoryConfig object first, then pass it to the op.
      //
      emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
          srcOp->getLoc(),
          emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
          "ttnn::MemoryConfig", arrayAttrs, nullptr, ValueRange());

      // Concat operands and MemoryConfig object.
      //
      operands.append(1, memCfgOp.getResult(0));

      attrs.push_back(mlir::IntegerAttr::get(rewriter.getIndexType(), 2));
    } else {
      attrs.push_back(ttnn_to_emitc::utils::createStdNullopt(rewriter));
    }

    ArrayAttr finalAttrs = ArrayAttr::get(srcOp->getContext(), attrs);

    // Convert ToDeviceOp
    //
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), finalAttrs, nullptr, operands);

    return success();
  }
};
} // namespace

// FromDeviceOp conversion pattern
//
namespace {
class FromDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::FromDeviceOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::FromDeviceOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::FromDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), nullptr, nullptr, adaptor.getOperands());

    return success();
  }
};
} // namespace

// TypecastOp conversion pattern
//
namespace {
class TypecastOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::TypecastOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::TypecastOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::TypecastOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getIndexType(), 0),
         ttnn_to_emitc::utils::convertDType(rewriter, srcOp.getDtypeAttr())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttrs, nullptr, adaptor.getOperands());

    return success();
  }
};
} // namespace

// ToMemoryConfig conversion pattern
//
namespace {
class ToMemoryConfigOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ToMemoryConfigOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::ToMemoryConfigOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ToMemoryConfigOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create ArrayAttr object holding MemoryConfig attributes.
    //
    ArrayAttr arrayAttrs = rewriter.getArrayAttr(
        {ttnn_to_emitc::utils::convertTensorMemoryLayout(
             rewriter, srcOp.getMemoryConfig().getTensorMemoryLayout()),
         ttnn_to_emitc::utils::convertBufferType(
             rewriter, srcOp.getMemoryConfig().getBufferType())});

    // Create MemoryConfig object first, then pass it to the op.
    //
    emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
        srcOp->getLoc(),
        emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
        "ttnn::MemoryConfig", arrayAttrs, nullptr, ValueRange());

    // Concat operands and MemoryConfig object.
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
} // namespace

// ToLayoutOp conversion pattern
//
namespace {
class ToLayoutOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ToLayoutOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::ToLayoutOp>::TTNNToEmitCBaseOpConversionPattern;

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
} // namespace

// EmptyOp conversion pattern
//
namespace {
class EmptyOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::EmptyOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::EmptyOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::EmptyOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn::ShapeAttr shapeAttr = srcOp.getShapeAttr();
    tt::DataTypeAttr dataTypeAttr = srcOp.getDtypeAttr();
    ttnn::LayoutAttr layoutAttr = srcOp.getLayoutAttr();

    // Find the GetDeviceOp.
    //
    ttnn::GetDeviceOp getDeviceOp;
    srcOp->getParentOp()->walk(
        [&getDeviceOp](ttnn::GetDeviceOp currGetDeviceOp) {
          getDeviceOp = currGetDeviceOp;
        });

    // Create ttnn::Shape() call.
    //
    emitc::CallOpaqueOp shapeOp = ttnn_to_emitc::utils::createShapeOp(
        rewriter, shapeAttr, srcOp.getLoc());

    // Create operands vector.
    //
    llvm::SmallVector<Value, 3> operands{shapeOp->getResult(0),
                                         adaptor.getDevice()};

    // Create MemoryConfig object first, then pass it to the op.
    //
    emitc::CallOpaqueOp memCfgOp = ttnn_to_emitc::utils::createMemoryConfigOp(
        rewriter, srcOp.getMemoryConfig(), srcOp.getLoc());

    // Concat operands and MemoryConfig object.
    //
    operands.append(1, memCfgOp.getResult(0));

    // Create ArrayAttr object holding attributes and pointers to operands.
    //
    ArrayAttr arrayAttr = rewriter.getArrayAttr({
        rewriter.getIndexAttr(0), // ttnn::Shape
        ttnn_to_emitc::utils::convertDType(rewriter, dataTypeAttr),
        ttnn_to_emitc::utils::convertLayoutAttr(rewriter, layoutAttr),
        rewriter.getIndexAttr(1), // ttnn::Device
        rewriter.getIndexAttr(2), // ttnn::MemoryConfig
    });

    // Finally, convert ttir::EmptyOp to ttnn::EmptyOp.
    //
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, this->getTypeConverter()->convertType(srcOp.getType()),
        this->convertOpName(srcOp), arrayAttr, nullptr, operands);

    return success();
  }
};
} // namespace

// ZerosOp conversion pattern
//
class ZerosOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::ZerosOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::ZerosOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttnn::ZerosOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // ttnn:ZerosOp has 5 input params:
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
    // We first create a ttnn::SimpleShape object (SSA) by calling
    // createShapeOp() and add it to the operands vector, but also add an
    // IndexAttr in ArrayAttr to reference it (this is an EmitC mechanism that
    // allows for combining Attrs and Values when calling an OpaqueOp). All the
    // other input params are optional, so we create them on-the-fly into the
    // ArrayAttr, whether they are an actual Attr, or a Value pointed to by
    // IndexAttr. If they are present, we create the object and pass it to the
    // op. If not, we pass std::nullopt.

    // Create ttnn::SimpleShape() call
    //
    emitc::CallOpaqueOp shapeOp = ttnn_to_emitc::utils::createShapeOp(
        rewriter, srcOp.getShapeAttr(), srcOp.getLoc());

    llvm::SmallVector<Value, 3> operands{
        shapeOp->getResult(0),
    };

    // Create ArrayAttr object holding attributes and pointers to operands
    //
    // Params that are Values are added to the operands vector on-the-fly, and
    // a corresponding IndexAttr is added to the ArrayAttr to reference them.
    //
    size_t operandIndex = 0;
    ArrayAttr arrayAttr = rewriter.getArrayAttr({
        rewriter.getIndexAttr(operandIndex++), // ttnn::SimpleShape
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

// OnesOp conversion pattern
//
namespace {
class OnesOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::OnesOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::OnesOp>::TTNNToEmitCBaseOpConversionPattern;

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
    // We first create a ttnn::Shape object (SA) by calling
    // createShapeOp() and add it to the operands vector, but also add an
    // IndexAttr in ArrayAttr to reference it (this is an EmitC mechanism that
    // allows for combining Attrs and Values when calling an OpaqueOp). All the
    // other input params are optional, so we create them on-the-fly into the
    // ArrayAttr, whether they are an actual Attr, or a Value pointed to by
    // IndexAttr. If they are present, we create the object and pass it to the
    // op. If not, we pass std::nullopt.

    // Create ttnn::Shape() call
    //
    emitc::CallOpaqueOp shapeOp = ttnn_to_emitc::utils::createShapeOp(
        rewriter, srcOp.getShapeAttr(), srcOp.getLoc());

    llvm::SmallVector<Value, 3> operands{
        shapeOp->getResult(0),
    };

    // Create ArrayAttr object holding attributes and pointers to operands
    //
    // Params that are Values are added to the operands vector on-the-fly, and
    // a corresponding IndexAttr is added to the ArrayAttr to reference them.
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
} // namespace

// DeallocateOp conversion pattern
//
namespace {
class DeallocateOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::DeallocateOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      ttnn::DeallocateOp>::TTNNToEmitCBaseOpConversionPattern;

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

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, newTy,
                                                   adaptor.getValue());
    return success();
  }
};
} // namespace

namespace {
class GetTupleElementOpConversionPattern
    : public OpConversionPattern<tt::GetTupleElementOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::GetTupleElementOp getTupleElementOp,
                  tt::GetTupleElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SubscriptOp requires a Value object as index, which is created by
    // invoking the emitc::LiteralOp.
    //
    Value indexAsVal = rewriter.create<emitc::LiteralOp>(
        getTupleElementOp->getLoc(), rewriter.getIndexType(),
        std::to_string(adaptor.getIndex()));

    // SubscriptOp also returns an emitc::LValueType, so we wrap the
    // OpaqueType with LValueType.
    //
    emitc::LValueType lvalueReturnType = emitc::LValueType::get(
        emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Tensor"));
    Value subscript = rewriter.create<emitc::SubscriptOp>(
        getTupleElementOp->getLoc(), lvalueReturnType, adaptor.getOperand(),
        indexAsVal);

    // As SubscriptOp returns an LValueType, we need to convert it to an
    // OpaqueType - this is done by invoking the emitc::LoadOp.
    //
    rewriter.replaceOpWithNewOp<emitc::LoadOp>(
        getTupleElementOp, emitc::OpaqueType::get(getContext(), "ttnn::Tensor"),
        subscript);
    return success();
  }
};
} // namespace

namespace {
class TupleOpConversionPattern : public OpConversionPattern<tt::TupleOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::TupleOp tupleOp, tt::TupleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // EmitC doesn't offer a way to create a vector from a list of values, so
    // we need to create a utility function that does this. This is achieved
    // by using EmitC's VerbatimOp.

    // Try to find if utility vec creation function is already defined in the
    // module. If not, insert it.
    //
    ttnn_to_emitc::utils::insertVecCreateFnIfNotExists(rewriter, tupleOp);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        tupleOp, this->getTypeConverter()->convertType(tupleOp.getType()),
        ttnn_to_emitc::utils::kCreateVectorFunctionName, nullptr, nullptr,
        adaptor.getOperands());
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
class ModuleOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<mlir::ModuleOp> {

public:
  using TTNNToEmitCBaseOpConversionPattern<
      mlir::ModuleOp>::TTNNToEmitCBaseOpConversionPattern;

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

// ANCHOR: op_rewriter_pattern_set_emitc
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
               ZerosOpConversionPattern,
               OnesOpConversionPattern,
               DefaultOpConversionPattern<ttnn::FullOp>,
               DefaultOpConversionPattern<ttnn::ArangeOp>>(typeConverter, ctx);
  // clang-format on

  // Eltwise unary ops
  //
  patterns.add<
      EltwiseUnaryOpConversionPattern<ttnn::AbsOp>,
      EltwiseUnaryCompositeOpConversionPattern<ttnn::CbrtOp>,
      EltwiseUnaryOpConversionPattern<ttnn::ClampOp>,
      EltwiseUnaryOpConversionPattern<ttnn::FloorOp>,
      EltwiseUnaryOpConversionPattern<ttnn::IsFiniteOp>,
      EltwiseUnaryOpConversionPattern<ttnn::LogicalNotOp>,
      EltwiseUnaryOpConversionPattern<ttnn::BitwiseNotOp>,
      EltwiseUnaryOpConversionPattern<ttnn::NegOp>,
      EltwiseUnaryOpConversionPattern<ttnn::ReluOp>,
      EltwiseUnaryOpConversionPattern<ttnn::LeakyReluOp>,
      EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<ttnn::GeluOp>,
      EltwiseUnaryOpConversionPattern<ttnn::SqrtOp>,
      EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<ttnn::RsqrtOp>,
      EltwiseUnaryOpConversionPattern<ttnn::SignOp>,
      EltwiseUnaryOpConversionPattern<ttnn::SigmoidOp>,
      EltwiseUnaryCompositeOpConversionPattern<ttnn::Log1pOp>,
      EltwiseUnaryOpConversionPattern<ttnn::ReciprocalOp>,
      EltwiseUnaryWithFastAndApproximateModeOpConversionPattern<ttnn::ExpOp>,
      EltwiseUnaryOpConversionPattern<ttnn::CeilOp>,
      EltwiseUnaryOpConversionPattern<ttnn::SinOp>,
      EltwiseUnaryOpConversionPattern<ttnn::CosOp>,
      EltwiseUnaryOpConversionPattern<ttnn::Expm1Op>,
      EltwiseUnaryOpConversionPattern<ttnn::TanOp>,
      EltwiseUnaryOpConversionPattern<ttnn::TanhOp>,
      EltwiseUnaryOpConversionPattern<ttnn::LogOp>>(typeConverter, ctx);

  // Eltwise binary ops
  //
  patterns.add<EltwiseBinaryOpConversionPattern<ttnn::AddOp>,
               EltwiseBinaryOpConversionPattern<ttnn::SubtractOp>,
               EltwiseBinaryOpConversionPattern<ttnn::MultiplyOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LogicalAndOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LogicalOrOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LogicalXorOp>,
               EltwiseBinaryOpConversionPattern<ttnn::BitwiseAndOp>,
               EltwiseBinaryOpConversionPattern<ttnn::BitwiseOrOp>,
               EltwiseBinaryOpConversionPattern<ttnn::BitwiseXorOp>,
               EltwiseBinaryOpConversionPattern<ttnn::EqualOp>,
               EltwiseBinaryOpConversionPattern<ttnn::NotEqualOp>,
               EltwiseBinaryOpConversionPattern<ttnn::GreaterEqualOp>,
               EltwiseBinaryOpConversionPattern<ttnn::GreaterThanOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LessEqualOp>,
               EltwiseBinaryOpConversionPattern<ttnn::LessThanOp>,
               EltwiseBinaryOpConversionPattern<ttnn::MaximumOp>,
               EltwiseBinaryOpConversionPattern<ttnn::MinimumOp>,
               EltwiseBinaryOpConversionPattern<ttnn::DivOp>,
               EltwiseBinaryOpConversionPattern<ttnn::ScatterOp>,
               EltwiseBinaryOpConversionPattern<ttnn::RemainderOp>,
               EltwiseBinaryOpConversionPattern<ttnn::PowerOp>>(typeConverter,
                                                                ctx);

  // Tensor manipulation ops
  //
  patterns.add<TransposeOpConversionPattern, ConcatOpConversionPattern,
               ReshapeOpConversionPattern, RepeatOpConversionPattern,
               DefaultOpConversionPattern<ttnn::RepeatInterleaveOp>,
               DefaultOpConversionPattern<ttnn::SliceOp>,
               DefaultOpConversionPattern<ttnn::PermuteOp>>(typeConverter, ctx);

  // Matmul ops
  //
  patterns.add<LinearOpConversionPattern, MatmulOpConversionPattern>(
      typeConverter, ctx);

  // Reduction ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::SumOp>, MeanOpConversionPattern,
               DefaultOpConversionPattern<ttnn::MaxOp>,
               DefaultOpConversionPattern<ttnn::MinOp>,
               DefaultOpConversionPattern<ttnn::ProdOp>>(typeConverter, ctx);

  // Conv ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::Conv2dOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ConvTranspose2dOp>>(
      typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MaxPool2dOp>>(typeConverter,
                                                              ctx);

  // Other ops
  //
  patterns.add<SoftmaxOpConversionPattern, EmbeddingOpConversionPattern,
               DefaultOpConversionPattern<ttnn::EmbeddingBackwardOp>,
               DefaultOpConversionPattern<ttnn::WhereOp>,
               MorehCumSumOpConversionPattern>(typeConverter, ctx);

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

  // Tuple ops
  //
  patterns.add<GetTupleElementOpConversionPattern>(typeConverter, ctx);
  patterns.add<TupleOpConversionPattern>(typeConverter, ctx);

  // Module op
  //
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);
}
// ANCHOR_END: op_rewriter_pattern_set_emitc

} // namespace mlir::tt
