// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

using namespace mlir;
using namespace mlir::tt;
using ttnn_to_emitpy::operator|;

// Base class for TTNN to EmitPy OpConversionPattern.
//
namespace {
template <typename SourceOp>
class TTNNToEmitPyBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

private:
  std::string virtual getPrefixSearchPattern() const { return "ttnn."; }
  std::string virtual getPrefixSwapPattern() const { return "ttnn."; }

public:
  // Converts op name by removing the prefixSearchPattern with
  // prefixSwapPattern, e.g. "ttnn." with "tt_metal."
  //
  std::string convertOpName(SourceOp op) const {
    auto name = op.getOperationName();
    assert(
        name.starts_with(getPrefixSearchPattern()) &&
        "DefaultOpConversionPattern only supports ops from the TTNN dialect");

    if (getPrefixSearchPattern() == getPrefixSwapPattern()) {
      return name.str();
    }

    // Exchange search with swap pattern.
    //
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
    : public TTNNToEmitPyBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      SourceOp>::TTNNToEmitPyBaseOpConversionPattern;
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
      auto resultTy = cast<emitpy::OpaqueType>(
          this->getTypeConverter()->convertType(srcOp->getResult(0).getType()));
      rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
          srcOp, resultTy, this->convertOpName(srcOp), adaptor.getOperands(),
          nullptr, nullptr);
    } else {
      // No return type, only convert the op.
      //
      rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp),
          adaptor.getOperands(), nullptr, nullptr);
    }

    return success();
  }
};
} // namespace

// EltwiseUnaryOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseUnaryOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(TTNNOpTy eltwiseUnaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseUnaryOp, adaptor,
                                                        rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseUnaryOp.getInput()),
        emitter.emit(eltwiseUnaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseUnaryOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EltwiseBinaryOp conversion pattern
//
namespace {
template <typename TTNNOpTy, typename OpAdaptor = typename TTNNOpTy::Adaptor>
class EltwiseBinaryOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<TTNNOpTy> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      TTNNOpTy>::TTNNToEmitPyBaseOpConversionPattern;
  LogicalResult

  matchAndRewrite(TTNNOpTy eltwiseBinaryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<TTNNOpTy> emitter(eltwiseBinaryOp,
                                                        adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(eltwiseBinaryOp.getLhs()),
        emitter.emit(eltwiseBinaryOp.getRhs()),
        emitter.emit(eltwiseBinaryOp.getDtype(), "dtype"),
        emitter.emit(eltwiseBinaryOp.getMemoryConfig() |
                         emitter.getMemoryConfig(eltwiseBinaryOp.getResult()),
                     "memory_config"),
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
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::MatmulOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::MatmulOp>::TTNNToEmitPyBaseOpConversionPattern;

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
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(matmulOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// MaxPool2dOp conversion pattern
//
namespace {
class MaxPool2dOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::MaxPool2dOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::MaxPool2dOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::MaxPool2dOp maxPool2dOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::MaxPool2dOp> emitter(
        maxPool2dOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(maxPool2dOp.getInput()),
        emitter.emit(maxPool2dOp.getBatchSize()),
        emitter.emit(maxPool2dOp.getInputHeight()),
        emitter.emit(maxPool2dOp.getInputWidth()),
        emitter.emit(maxPool2dOp.getChannels()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getKernelSizeAttr()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getStrideAttr()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getPaddingAttr()),
        emitter.template emit<std::vector<uint32_t>>(
            maxPool2dOp.getDilationAttr()),
        emitter.emit(emitter.getMemoryConfig(maxPool2dOp.getResult()),
                     "memory_config"),
        emitter.emit(maxPool2dOp.getAppliedShardScheme(),
                     "applied_shard_scheme"),
        emitter.emit(maxPool2dOp.getCeilMode(), "ceil_mode"),
        emitter.emit(maxPool2dOp.getInPlaceHalo(), "in_place_halo"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

namespace {
class SoftmaxOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::SoftmaxOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::SoftmaxOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SoftmaxOp softmaxOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::SoftmaxOp> emitter(
        softmaxOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(softmaxOp.getInput()),
        emitter.emit(softmaxOp.getDimension()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(softmaxOp.getResult()),
                     "memory_config"),
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

// DeallocateOp conversion pattern
//
namespace {
class DeallocateOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::DeallocateOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::DeallocateOp>::TTNNToEmitPyBaseOpConversionPattern;

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
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttcore::DeviceOp> {
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttcore::DeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

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
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::GetDeviceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::GetDeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

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
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ToDeviceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToDeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToDeviceOp toDeviceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToDeviceOp> emitter(
        toDeviceOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(toDeviceOp.getInput()),
        emitter.emit(toDeviceOp.getDevice(), "device"),
        emitter.emit(toDeviceOp.getMemoryConfig() |
                         emitter.getMemoryConfig(toDeviceOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// FromDeviceOp conversion pattern
//
namespace {
class FromDeviceOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::FromDeviceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::FromDeviceOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::FromDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::FromDeviceOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// TypecastOp conversion pattern
//
namespace {
class TypecastOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::TypecastOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::TypecastOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::TypecastOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::TypecastOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDtype()),
        emitter.emit(std::nullopt | emitter.getMemoryConfig(srcOp.getResult())),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ToDTypeOp conversion pattern
//
namespace {
class ToDTypeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ToDTypeOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToDTypeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToDTypeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToDTypeOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getDtype()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ToLayoutOp conversion pattern
//
namespace {
class ToLayoutOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ToLayoutOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ToLayoutOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ToLayoutOp toLayoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ToLayoutOp> emitter(
        toLayoutOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(toLayoutOp.getInput()),
        emitter.emit(toLayoutOp.getLayout()),
        emitter.emit(toLayoutOp.getDtype()),
        emitter.emit(toLayoutOp.getMemoryConfig() |
                         emitter.getMemoryConfig(toLayoutOp.getResult()),
                     "memory_config"),
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
class NamedFullOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<SourceOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      SourceOp>::TTNNToEmitPyBaseOpConversionPattern;
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
        /*  emitter.emit(namedFullOp.getMemoryConfig() |
                          emitter.getMemoryConfig(namedFullOp.getResult()),
                      "memory_config"), */
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ReductionOp conversion pattern
//
namespace {
template <typename ReductionOp>
class ReductionOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<ReductionOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      ReductionOp>::TTNNToEmitPyBaseOpConversionPattern;
  using Adaptor = typename ReductionOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReductionOp reductionOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<ReductionOp> emitter(reductionOp, adaptor,
                                                           rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(reductionOp.getInput()),
        emitter.template emit<::ttsl::SmallVector<int32_t>>(
            reductionOp.getDimArg()),
        emitter.emit(reductionOp.getKeepDim()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(reductionOp.getResult()),
                     "memory_config")};

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// Conv2dOp conversion pattern
//
namespace {
class Conv2dOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::Conv2dOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::Conv2dOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::Conv2dOp conv2dOp,
                  mlir::tt::ttnn::Conv2dOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::Conv2dOp> emitter(
        conv2dOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(conv2dOp.getInput(), "input_tensor"),
        emitter.emit(conv2dOp.getWeight(), "weight_tensor"),
        emitter.emit(conv2dOp.getDevice(), "device"),
        emitter.emit(conv2dOp.getInChannels(), "in_channels"),
        emitter.emit(conv2dOp.getOutChannels(), "out_channels"),
        emitter.emit(conv2dOp.getBatchSize(), "batch_size"),
        emitter.emit(conv2dOp.getInputHeight(), "input_height"),
        emitter.emit(conv2dOp.getInputWidth(), "input_width"),
        emitter.template emit<std::vector<uint32_t>>(
            conv2dOp.getKernelSizeAttr(), "kernel_size"),
        emitter.template emit<std::vector<uint32_t>>(conv2dOp.getStrideAttr(),
                                                     "stride"),
        emitter.template emit<std::vector<uint32_t>>(conv2dOp.getPaddingAttr(),
                                                     "padding"),
        emitter.template emit<std::vector<uint32_t>>(conv2dOp.getDilationAttr(),
                                                     "dilation"),
        emitter.emit(conv2dOp.getGroups(), "groups"),
        emitter.emit(conv2dOp.getBias(), "bias_tensor"),
        emitter.emit(conv2dOp.getConv2dConfig(), "conv_config"),
        emitter.emit(std::nullopt, "compute_config"),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(conv2dOp.getResult()),
                     "memory_config"),
    };
    /* llvm::SmallVector<mlir::StringRef> keyword_args{
        "input_tensor",   "weight_tensor", "bias_tensor", "device",
        "in_channels",    "out_channels",  "batch_size",  "input_height",
        "input_width",    "kernel_size",   "stride",      "padding",
        "dilation",       "groups",        "dtype",       "conv_config",
        "compute_config", "memory_config"};
 */
    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ReshapeOp conversion pattern
//
namespace {
class ReshapeOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ReshapeOp> {

public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ReshapeOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ReshapeOp reshapeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ReshapeOp> emitter(
        reshapeOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(reshapeOp.getInput()),
        emitter.emit<std::vector<int32_t>>(reshapeOp.getShape()),
        emitter.emit(reshapeOp.getMemoryConfig() |
                         emitter.getMemoryConfig(reshapeOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// RepeatOp conversion pattern
//
namespace {
class RepeatOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::RepeatOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::RepeatOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RepeatOp repeatOp,
                  mlir::tt::ttnn::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::RepeatOp> emitter(
        repeatOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(repeatOp.getInput()),
        emitter.emit(repeatOp.getRepeatDims()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// ConcatOp conversion pattern
//
namespace {
class ConcatOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::ConcatOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::ConcatOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ConcatOp concatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::ConcatOp> emitter(
        concatOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(concatOp.getInputs()),
        emitter.emit(concatOp.getDim()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(concatOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// PermuteOp conversion pattern
//
namespace {
class PermuteOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::PermuteOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::PermuteOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::PermuteOp permuteOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::PermuteOp> emitter(
        permuteOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(permuteOp.getInput()),
        emitter.emit(permuteOp.getPermutation()),
        emitter.emit(std::nullopt |
                         emitter.getMemoryConfig(permuteOp.getResult()),
                     "memory_config"),
        emitter.emit(permuteOp.getPadValue(), "pad_value"),
    };

    emitter.replaceOp(*this, args);

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

    rewriter.replaceOpWithNewOp<emitpy::SubscriptOp>(
        getTupleElementOp,
        this->getTypeConverter()->convertType(getTupleElementOp.getType()),
        adaptor.getOperand(), indexAsVal);

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
  patterns.add<GetDeviceOpConversionPattern,
               TTDeviceOpConversionPattern
              >(typeConverter, ctx);
  // clang-format on

  // Tensor ops
  //
  // clang-format off
  patterns.add<NamedFullOpConversionPattern<mlir::tt::ttnn::OnesOp>,
               NamedFullOpConversionPattern<mlir::tt::ttnn::ZerosOp>
              >(typeConverter, ctx);
  // clang-format on

  // Arith ops
  //
  // clang-format off
  patterns.add<ArithConstantOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Matmul ops
  //
  // clang-format off
  patterns.add<MatmulOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Reduction ops
  //
  // clang-format off
  patterns.add<ReductionOpConversionPattern<mlir::tt::ttnn::MaxOp>,
               ReductionOpConversionPattern<mlir::tt::ttnn::MeanOp>,
               ReductionOpConversionPattern<mlir::tt::ttnn::MinOp>,
               ReductionOpConversionPattern<mlir::tt::ttnn::SumOp>
              >(typeConverter, ctx);
  // clang-format on

  // Eltwise unary ops
  //
  // clang-format off
  patterns.add<EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::AbsOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::AtanOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::BitwiseNotOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::CeilOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::CosOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::Expm1Op>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::FloorOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::IsFiniteOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::LogicalNotOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::LogOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::NegOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::ReciprocalOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::ReluOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SignOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SinOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::SqrtOp>,
               EltwiseUnaryOpConversionPattern<mlir::tt::ttnn::TanOp>
              >(typeConverter, ctx);
  // clang-format on

  // Eltwise binary ops
  //
  // clang-format off
  patterns.add<EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::AddOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::DivideOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::EqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::GreaterEqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::GreaterThanOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LessEqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LessThanOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalAndOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalOrOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::LogicalXorOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::MultiplyOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::NotEqualOp>,
               EltwiseBinaryOpConversionPattern<mlir::tt::ttnn::SubtractOp>
              >(typeConverter, ctx);
  // clang-format on

  // Tensor manipulation ops
  //
  // clang-format off
  patterns.add<ConcatOpConversionPattern,
               PermuteOpConversionPattern,
               ReshapeOpConversionPattern,
               RepeatOpConversionPattern
              >(typeConverter, ctx);
  // clang-format on

  // Memory ops
  //
  // clang-format off
  patterns.add<DeallocateOpConversionPattern,
               FromDeviceOpConversionPattern,
               ToDeviceOpConversionPattern,
               ToDTypeOpConversionPattern,
               ToLayoutOpConversionPattern,
               TypecastOpConversionPattern
              >(typeConverter, ctx);
  // clang-format on

  // Pooling ops
  //
  // clang-format off
  patterns.add<MaxPool2dOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Convolution ops
  //
  // clang-format off
  patterns.add<Conv2dOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Other ops
  //
  // clang-format off
  patterns.add<SoftmaxOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Tuple ops
  //
  // clang-format off
  patterns.add<GetTupleElementOpConversionPattern,
               TupleOpConversionPattern>(typeConverter, ctx);
  // clang-format on

  // Default (catch-all)
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::ConstantOp>>(
      typeConverter, ctx);

  // Module op
  //
  // clang-format off
  patterns.add<ModuleOpConversionPattern>(typeConverter, ctx);
  // clang-format on
}

} // namespace mlir::tt
